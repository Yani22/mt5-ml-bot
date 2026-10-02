"""Walk-forward over longer horizons and cost-aware labels (extends scripts/walkforward.py).

label=dir : y_long = price up after `horizon` bars, y_short = price down (the bot's current target).
label=cost: y_long = price up by more than K * spread after `horizon` bars, y_short = mirror. Rows that move less
            than that are "no trade" for both sides, so the model only learns moves that can pay for the spread.
Probability thresholds are set relative to each fold's train base rate (base + offset) so the two label types are
comparable. Entry/exit/costs are identical to walkforward.py.

Usage: python scripts/walkforward_horizons.py --symbols USDJPY GBPUSD EURUSD --horizons 24 48 96 --k 2
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
from extra_features import extra_features  # noqa: E402
from src.config import FeatureCfg  # noqa: E402

OFFSETS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.14]


def labels(close, spread_px, hz, label, k):
    fwd = (close.shift(-hz) - close).dropna()  # last hz rows unknown: dropped, not zero
    if label == "dir":
        return (fwd > 0).astype(int), (fwd < 0).astype(int)
    cost = k * spread_px.reindex(fwd.index)
    return (fwd > cost).astype(int), (fwd < -cost).astype(int)


def run(symbol, args):
    wf.set_point(symbol, args.raw)
    m5, h1, sp = wf.load(symbol, args.data)
    X_all = wf.build(m5, h1, FeatureCfg())
    if args.extra:
        X_all = X_all.join(extra_features(m5, h1).loc[X_all.index])
    o = m5["open"].to_numpy()
    spread_px = sp * wf.POINT
    out = []
    for hz in args.horizons:
        for label in args.labels:
            yl, ys = labels(m5["close"], spread_px, hz, label, args.k)
            common = X_all.index.intersection(yl.index)
            X, yl, ys = X_all.loc[common], yl.loc[common], ys.loc[common]
            pos = m5.index.get_indexer(common)
            start, aucs, rows = args.train, [], []
            while start + args.test <= len(X):
                tr = slice(start - args.train, start - hz)  # purge labels that look into the test window
                te = slice(start, start + args.test)
                params = dict(n_estimators=200, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, num_leaves=31, min_child_samples=200, n_jobs=4, verbose=-1)
                ml = LGBMClassifier(**params).fit(X.iloc[tr], yl.iloc[tr])
                ms = LGBMClassifier(**params).fit(X.iloc[tr], ys.iloc[tr])
                pl, ps = ml.predict_proba(X.iloc[te])[:, 1], ms.predict_proba(X.iloc[te])[:, 1]
                if yl.iloc[te].nunique() > 1 and ys.iloc[te].nunique() > 1:
                    aucs.append((roc_auc_score(yl.iloc[te], pl), roc_auc_score(ys.iloc[te], ps)))
                bl, bs = yl.iloc[tr].mean(), ys.iloc[tr].mean()
                for off in OFFSETS:
                    # per-side thresholds: base rate of that side + offset (simulate() takes one threshold,
                    # so rescale probabilities so both sides share it)
                    th = 0.5 + off
                    tr_l = pl - bl + 0.5
                    tr_s = ps - bs + 0.5
                    for g, nt in wf.simulate(tr_l, tr_s, o, spread_px.to_numpy(), pos[te], th, hz,
                                             args.slip * wf.PIP):
                        rows.append((start, off, g / wf.PIP, nt / wf.PIP))
                start += args.test
            df = pd.DataFrame(rows, columns=["fold", "off", "gross", "net"])
            if df.empty:
                continue
            s = df.groupby("off").agg(trades=("net", "size"), gross=("gross", "mean"), net=("net", "mean"),
                                      win=("net", lambda x: 100 * (x > 0).mean()))
            s["folds_pos"] = df.groupby(["off", "fold"])["net"].sum().gt(0).groupby("off").mean() * 100
            s["auc"] = np.mean(aucs) if aucs else np.nan
            s.insert(0, "label", label)
            s.insert(0, "hz", hz)
            s.insert(0, "symbol", symbol)
            out.append(s.reset_index())
            print(f"{symbol} hz={hz} label={label} done", flush=True)
    res = pd.concat(out)
    os.makedirs("results", exist_ok=True)
    res.to_csv(f"results/walkforward_horizons{args.tag}_{symbol}.csv", index=False)
    print(f"\n=== {symbol} (spread {np.median(spread_px) / wf.PIP:.2f} pips, k={args.k}) ===")
    print(res.drop(columns="symbol").round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["USDJPY", "GBPUSD", "EURUSD"])
    ap.add_argument("--data", default="data/historical_data")
    ap.add_argument("--raw", default="data/raw_export")
    ap.add_argument("--train", type=int, default=60000)
    ap.add_argument("--test", type=int, default=10000)
    ap.add_argument("--horizons", type=int, nargs="+", default=[24, 48, 96])
    ap.add_argument("--k", type=float, default=2.0, help="cost label: move must exceed k * spread")
    ap.add_argument("--slip", type=float, default=0.1)
    ap.add_argument("--extra", action="store_true", help="add scripts/extra_features.py features")
    ap.add_argument("--tag", default="", help="suffix for the results file name")
    ap.add_argument("--labels", nargs="+", default=["dir", "cost"])
    a = ap.parse_args()
    for s in a.symbols:
        run(s, a)
