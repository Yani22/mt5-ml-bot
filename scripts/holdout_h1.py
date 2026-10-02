"""True holdout for the frozen USDJPY H1 strategy: years the design never saw (before 2024).

PRE-REGISTERED (written before running): frozen spec (train 6000 rows, test 1000, seed 0, offset 0.06, 2.0 x ATR14 stop,
24-bar hold, same costs: recorded spread floored at the current median spread + 0.1 pip slippage per side), data from
the start of the export through 2023-12-31. PASS if mean net pips/trade > 0 with the 95% bootstrap (over folds) lower
bound > 0 AND >= 60% of folds positive. Swap is reported separately (unresolved for this account).

Usage: python scripts/holdout_h1.py --symbol USDJPY [--end 2023-12-31]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import walkforward as wf  # noqa: E402
from walkforward_h1_stop import simulate_stop  # noqa: E402
from src.h1_features import build_h1  # noqa: E402
from src import h1_strategy as hs  # noqa: E402


def main(a):
    wf.set_point(a.symbol, a.raw)
    h1 = pd.read_csv(f"{a.data}/{a.symbol}_H1.csv", index_col=0, parse_dates=True)
    sp = pd.read_csv(f"{a.data}/{a.symbol}_H1_spread.csv", index_col=0, parse_dates=True)["spread"]
    h1.index, sp.index = h1.index.tz_localize(None), h1.index.tz_localize(None)
    h1, sp = h1[h1.index <= pd.Timestamp(a.end) + pd.Timedelta("23h")], sp[sp.index <= pd.Timestamp(a.end) + pd.Timedelta("23h")]
    spread_px = np.maximum(sp.to_numpy(), a.spread_floor_pts) * wf.POINT
    print(f"{a.symbol}: {len(h1)} bars {h1.index[0]} -> {h1.index[-1]}; share of bars with recorded spread below floor: "
          f"{(sp.to_numpy() < a.spread_floor_pts).mean():.0%}")
    X_all = build_h1(h1)
    atr = X_all["atr_14"].reindex(h1.index).to_numpy()
    o, h, l_ = h1["open"].to_numpy(), h1["high"].to_numpy(), h1["low"].to_numpy()
    hz = hs.HORIZON
    fwd = (h1["close"].shift(-hz) - h1["close"]).dropna()
    yl, ys = (fwd > 0).astype(int), (fwd < 0).astype(int)
    common = X_all.index.intersection(fwd.index)
    X, yl, ys = X_all.loc[common], yl.loc[common], ys.loc[common]
    pos = h1.index.get_indexer(common)
    rows, start, nfold = [], hs.TRAIN_ROWS, 0
    while start + hs.RETRAIN_EVERY <= len(X):
        tr, te = slice(start - hs.TRAIN_ROWS, start - hz), slice(start, start + hs.RETRAIN_EVERY)
        pl = LGBMClassifier(**hs.LGBM_PARAMS).fit(X.iloc[tr], yl.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
        ps = LGBMClassifier(**hs.LGBM_PARAMS).fit(X.iloc[tr], ys.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
        bl, bs = yl.iloc[tr].mean(), ys.iloc[tr].mean()
        for net, stopped, side, eb, held in simulate_stop(pl - bl + 0.5, ps - bs + 0.5, o, h, l_, spread_px, atr, pos[te],
                                                          0.5 + hs.OFFSET, hz, a.slip * wf.PIP, hs.STOP_ATR):
            rows.append((nfold, h1.index[eb], side, held, net / wf.PIP))
        start += hs.RETRAIN_EVERY
        nfold += 1
        if nfold % 10 == 0:
            print(f"  fold {nfold}", flush=True)
    t = pd.DataFrame(rows, columns=["fold", "entry", "side", "held", "net"])
    t.to_csv(f"results/holdout_{a.symbol}.csv", index=False)
    g = t.groupby("fold").net.agg(["sum", "size"])
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        s = g.iloc[rng.integers(0, len(g), len(g))]
        boots.append(s["sum"].sum() / s["size"].sum())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    fp = (g["sum"] > 0).mean() * 100
    mean = t.net.mean()
    print(f"\nHOLDOUT {a.symbol} {t.entry.min():%Y-%m} -> {t.entry.max():%Y-%m}: folds {len(g)}, trades {len(t)}")
    print(f"mean net {mean:.2f} pips/trade, 95% CI [{lo:.2f}, {hi:.2f}], folds positive {fp:.0f}%")
    print("by side:", t.groupby("side").net.agg(["size", "mean"]).round(2).to_dict("index"))
    print("by year:", t.groupby(t.entry.dt.year).net.agg(["size", "mean"]).round(2).to_dict("index"))
    print("PASS" if (mean > 0 and lo > 0 and fp >= 60) else "FAIL", "(pre-registered bar: mean>0, CI low>0, folds+>=60%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="USDJPY")
    ap.add_argument("--data", default="data/historical_old")
    ap.add_argument("--raw", default="data/raw_export/old")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--slip", type=float, default=0.1)
    ap.add_argument("--spread-floor-pts", type=float, default=9.0, help="USDJPY median spread today: 9 points")
    main(ap.parse_args())
