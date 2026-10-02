"""Leak-free walk-forward test of the signal edge, net of spread and slippage (CSV data, no MT5 needed).

For each fold: train long/short LightGBM on the past `--train` bars (last `horizon` rows purged), then trade the next
`--test` bars out-of-sample. A signal at the close of bar t enters at the open of bar t+1 and exits at the open of
bar t+1+horizon. One position at a time. Long pays the entry-bar spread, short pays it at exit (approximated with the
same bar's spread). Reports gross and net pips per trade for a grid of probability thresholds.

Usage: python scripts/walkforward.py [--symbols EURUSD GBPUSD AUDUSD] [--train 60000] [--test 10000]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import FeatureCfg, MtaCfg, PriceActionCfg  # noqa: E402
from src.features import add_contextual_features, build_dynamic_features, build_static_features  # noqa: E402
from src.labels import generate_long_short_labels  # noqa: E402

PIP = POINT = 0.0  # set per symbol in run() from the price digits (pip = 10 points)
WARMUP = 1000  # drop the start where indicators are warming up (and dynamic features bfill)


def load(symbol: str, d: str):
    m5 = pd.read_csv(f"{d}/{symbol}_M5.csv", index_col=0, parse_dates=True)
    h1 = pd.read_csv(f"{d}/{symbol}_H1.csv", index_col=0, parse_dates=True)
    sp = pd.read_csv(f"{d}/{symbol}_M5_spread.csv", index_col=0, parse_dates=True)["spread"]
    return m5, h1, sp.reindex(m5.index).ffill()


def build(m5, h1, cfg):
    static = build_static_features(m5, "WF", pa_cfg=PriceActionCfg())
    X = build_dynamic_features(m5, static, cfg, "WF")
    X = add_contextual_features(X, mta_df=h1, mta_cfg=MtaCfg())
    return X.iloc[WARMUP:]


def simulate(pl, ps, o, spread_px, idx, th, horizon, slip_px):
    """Returns list of (gross_px, net_px) per trade over positions idx (signal bars)."""
    out = []
    i = 0
    n = len(idx)
    while i < n:
        t = idx[i]
        if t + 1 + horizon >= len(o):
            break
        side = 0
        if pl[i] > th and pl[i] > ps[i]:
            side = 1
        elif ps[i] > th and ps[i] > pl[i]:
            side = -1
        if side == 0:
            i += 1
            continue
        gross = side * (o[t + 1 + horizon] - o[t + 1])
        net = gross - spread_px[t + 1] - 2 * slip_px
        out.append((gross, net))
        i += 1 + horizon  # position occupies the next `horizon` bars
    return out


def set_point(symbol, raw_dir):
    """Point size from the number of decimals in the raw MT5 export (5 -> 1e-5, 3 -> 1e-3, 2 -> 1e-2)."""
    global PIP, POINT
    name = {"GOLD": "GOLD#"}.get(symbol, symbol + "#")
    raw = f"{raw_dir}/{name}_M5.csv"
    if not os.path.exists(raw):
        raw = f"{raw_dir}/{name}_H1.csv"  # H1-only exports
    row = open(raw).readlines()[1].split("\t")[5]
    POINT = 10.0 ** -len(row.split(".")[1])
    PIP = POINT * 10


def run(symbol, args):
    set_point(symbol, args.raw)
    m5, h1, sp = load(symbol, args.data)
    cfg = FeatureCfg()
    hz = args.horizon
    X = build(m5, h1, cfg)
    y_long, y_short = generate_long_short_labels(m5, hz, cfg.min_pct_change)
    common = X.index.intersection(y_long.index)
    X, yl, ys = X.loc[common], y_long.loc[common], y_short.loc[common]
    pos = m5.index.get_indexer(common)  # positions of X rows in m5
    o = m5["open"].to_numpy()
    spread_px = (sp.to_numpy() * POINT)
    ths = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62]
    rows, aucs = [], []
    start = args.train
    while start + args.test <= len(X):
        tr = slice(start - args.train, start - hz)  # purge: labels of last hz train rows look into the test window
        te = slice(start, start + args.test)
        params = dict(n_estimators=200, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.8, num_leaves=31, min_child_samples=200, n_jobs=4, verbose=-1)
        ml = LGBMClassifier(**params).fit(X.iloc[tr], yl.iloc[tr])
        ms = LGBMClassifier(**params).fit(X.iloc[tr], ys.iloc[tr])
        pl, ps = ml.predict_proba(X.iloc[te])[:, 1], ms.predict_proba(X.iloc[te])[:, 1]
        aucs.append((roc_auc_score(yl.iloc[te], pl), roc_auc_score(ys.iloc[te], ps)))
        for th in ths:
            for g, nt in simulate(pl, ps, o, spread_px, pos[te], th, hz, args.slip * PIP):
                rows.append((start, th, g / PIP, nt / PIP))
        start += args.test
    df = pd.DataFrame(rows, columns=["fold", "th", "gross", "net"])
    summ = df.groupby("th").agg(trades=("net", "size"), gross_pips=("gross", "mean"), net_pips=("net", "mean"),
                                win_pct=("net", lambda s: 100 * (s > 0).mean()), total_net=("net", "sum"))
    fold_pos = df.groupby(["th", "fold"])["net"].sum().gt(0).groupby("th").mean() * 100
    summ["folds_pos_pct"] = fold_pos
    a = np.array(aucs)
    print(f"\n=== {symbol}: {len(aucs)} folds, mean OOS AUC long {a[:, 0].mean():.4f} short {a[:, 1].mean():.4f}, "
          f"median spread {np.median(spread_px) / PIP:.2f} pips ===")
    print(summ.round(3).to_string())
    os.makedirs("results", exist_ok=True)
    summ.to_csv(f"results/walkforward_{symbol}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "GOLD"])
    ap.add_argument("--data", default="data/historical_data")
    ap.add_argument("--raw", default="data/raw_export")
    ap.add_argument("--train", type=int, default=60000)
    ap.add_argument("--test", type=int, default=10000)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--slip", type=float, default=0.1, help="slippage in pips per side")
    a = ap.parse_args()
    for s in a.symbols:
        run(s, a)
