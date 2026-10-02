"""Low-frequency walk-forward: H1 bars, holds of several hours to a day, so the spread is a small share of each trade.

Same leak-free setup as walkforward.py (train on the past, purge `horizon` rows, test on the next block, enter at the
next open, exit `horizon` bars later, pay the exported spread + slippage), but on H1 data with the bot's own features
plus H4/D1 trend context (no M5 data needed).

Pre-registered success criterion (decided before looking at results): at one (horizon, offset) setting, net pips/trade
> 0 with >= 60% of folds positive on at least 4 of the 6 symbols.

Usage: python scripts/walkforward_h1.py [--symbols ...] [--horizons 4 8 24]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import walkforward as wf  # noqa: E402
from cross_features import cross_features  # noqa: E402
from src.h1_features import build_h1  # noqa: E402

OFFSETS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.18]


def run(symbol, args):
    wf.set_point(symbol, args.raw)
    h1 = pd.read_csv(f"{args.data}/{symbol}_H1.csv", index_col=0, parse_dates=True)
    sp = pd.read_csv(f"{args.data}/{symbol}_H1_spread.csv", index_col=0, parse_dates=True)["spread"]
    spread_px = (sp.reindex(h1.index).ffill() * wf.POINT).to_numpy()
    X_all = build_h1(h1)
    if getattr(args, "cross", False):
        X_all = X_all.join(cross_features(X_all.index, symbol, args.data))
    o = h1["open"].to_numpy()
    rows, aucs = [], {}
    for hz in args.horizons:
        fwd = (h1["close"].shift(-hz) - h1["close"]).dropna()
        yl, ys = (fwd > 0).astype(int), (fwd < 0).astype(int)
        common = X_all.index.intersection(fwd.index)
        X, yl, ys = X_all.loc[common], yl.loc[common], ys.loc[common]
        pos = h1.index.get_indexer(common)
        start, fold_aucs = args.train, []
        while start + args.test <= len(X):
            tr, te = slice(start - args.train, start - hz), slice(start, start + args.test)
            params = dict(n_estimators=200, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, num_leaves=15, min_child_samples=50, n_jobs=4, verbose=-1, random_state=args.seed)
            pl = LGBMClassifier(**params).fit(X.iloc[tr], yl.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
            ps = LGBMClassifier(**params).fit(X.iloc[tr], ys.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
            fold_aucs.append(roc_auc_score(yl.iloc[te], pl))
            bl, bs = yl.iloc[tr].mean(), ys.iloc[tr].mean()
            for off in OFFSETS:
                for g, nt in wf.simulate(pl - bl + 0.5, ps - bs + 0.5, o, spread_px, pos[te], 0.5 + off, hz,
                                         args.slip * wf.PIP):
                    rows.append((symbol, hz, off, start, g / wf.PIP, nt / wf.PIP))
            start += args.test
        aucs[hz] = np.mean(fold_aucs)
        print(f"{symbol} hz={hz} done, mean AUC(long) {aucs[hz]:.3f}", flush=True)
    return pd.DataFrame(rows, columns=["symbol", "hz", "off", "fold", "gross", "net"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "GBPJPY", "GOLD"])
    ap.add_argument("--data", default="data/historical_data")
    ap.add_argument("--raw", default="data/raw_export")
    ap.add_argument("--train", type=int, default=6000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--horizons", type=int, nargs="+", default=[4, 8, 24])
    ap.add_argument("--slip", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    allr = pd.concat([run(s, a) for s in a.symbols])
    os.makedirs("results", exist_ok=True)
    allr.to_csv(f"results/walkforward_h1{a.tag}_trades.csv", index=False)
    per = allr.groupby(["symbol", "hz", "off"]).agg(trades=("net", "size"), gross=("gross", "mean"), net=("net", "mean"))
    per["folds_pos"] = allr.groupby(["symbol", "hz", "off", "fold"])["net"].sum().gt(0).groupby(
        ["symbol", "hz", "off"]).mean() * 100
    per = per.reset_index()
    per["ok"] = (per.trades >= 50) & (per.net > 0) & (per.folds_pos >= 60)
    crit = per.groupby(["hz", "off"]).agg(
        symbols_ok=("ok", "sum"), mean_net=("net", "mean"), mean_gross=("gross", "mean"), trades=("trades", "sum"))
    print("\nPer (hold, offset) across symbols; criterion needs symbols_ok >= 4:")
    print(crit.round(3).to_string())
    print("\nBest single cells (noise-prone):")
    print(per[per.trades >= 50].sort_values("net", ascending=False).head(8).round(3).to_string(index=False))
    per.to_csv(f"results/walkforward_h1{a.tag}_summary.csv", index=False)
