"""Screen symbols on the H1 24h hold with a pre-registered bar.

Per symbol, run the base walk-forward and 5 variants (train 4000/8000, test 500/1500, seed 7). A variant passes if the
mean over offsets 0.02-0.14 has net pips/trade > 0 and >= 60% of folds positive. A symbol passes if >= 4 of the 5
variants pass. Nothing is tuned per symbol.

Usage: python scripts/screen_h1.py --symbols EURJPY AUDJPY ...
"""
import argparse
import os
import sys
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from walkforward_h1 import run  # noqa: E402

VARIANTS = {"base": dict(train=6000, test=1000, seed=0), "tr4000": dict(train=4000, test=1000, seed=0),
            "tr8000": dict(train=8000, test=1000, seed=0), "te500": dict(train=6000, test=500, seed=0),
            "te1500": dict(train=6000, test=1500, seed=0), "seed7": dict(train=6000, test=1000, seed=7)}


def summarize(df):
    d = df[df.off.between(0.019, 0.141)]
    net = d.groupby("off").net.mean().mean()
    folds = d.groupby(["off", "fold"]).net.sum().gt(0).groupby("off").mean().mean() * 100
    return net, folds, d.groupby("off").size().mean()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--data", default="data/historical_data")
    ap.add_argument("--raw", default="data/raw_export")
    ap.add_argument("--cross", action="store_true", help="add cross-asset features")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    rows = []
    for sym in a.symbols:
        for name, v in VARIANTS.items():
            args = SimpleNamespace(data=a.data, raw=a.raw, horizons=[24], slip=0.1, cross=a.cross, **v)
            net, folds, trades = summarize(run(sym, args))
            rows.append((sym, name, net, folds, trades, net > 0 and folds >= 60))
            print(f"{sym} {name} net {net:.2f} folds+ {folds:.0f}% trades {trades:.0f}", flush=True)
    r = pd.DataFrame(rows, columns=["symbol", "variant", "net", "folds_pos", "trades", "pass"])
    r.to_csv(f"results/screen_h1{a.tag}_variants.csv", index=False)
    v = r[r.variant != "base"]
    out = v.groupby("symbol").agg(variants_pass=("pass", "sum"), mean_net=("net", "mean"), mean_folds=("folds_pos", "mean"))
    out["base_net"] = r[r.variant == "base"].set_index("symbol").net
    out["PASS"] = out.variants_pass >= 4
    print("\n", out.sort_values("variants_pass", ascending=False).round(2).to_string())
