"""H1 walk-forward (as walkforward_h1.py) with a hard ATR stop-loss inside the hold.

Entry at the open of bar t+1; stop at entry -/+ m * ATR14(t). While holding, a bar whose low (long) / high (short)
crosses the stop exits there (a bar that opens beyond the stop exits at its open). Otherwise exit at the open of
bar t+1+horizon. Spread and slippage are charged as in the other scripts. A stopped trade frees the slot, so the next
signal can enter right after it.

Usage: python scripts/walkforward_h1_stop.py --symbols USDJPY GBPJPY --horizon 24
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import walkforward as wf  # noqa: E402
from walkforward_h1 import build_h1  # noqa: E402

OFFSETS = [0.02, 0.06, 0.10, 0.14]
STOPS = [None, 4.0, 3.0, 2.0, 1.5, 1.0]


def simulate_stop(pl, ps, o, h, l_, spread_px, atr, idx, th, hz, slip_px, m):
    """Returns (net_px, stopped, side, entry_bar, bars_held) per trade."""
    out = []
    i, n = 0, len(idx)
    while i < n:
        t = idx[i]
        if t + 1 + hz >= len(o):
            break
        side = 1 if (pl[i] > th and pl[i] > ps[i]) else -1 if (ps[i] > th and ps[i] > pl[i]) else 0
        if side == 0:
            i += 1
            continue
        entry = o[t + 1]
        exit_px, held, stopped = o[t + 1 + hz], hz, False
        if m is not None:
            stop = entry - side * m * atr[t]
            for k in range(1, hz + 1):  # bars t+1 .. t+hz; exit bar t+1+hz is the planned open exit
                b = t + k
                if side == 1 and l_[b] <= stop:
                    exit_px, held, stopped = min(o[b], stop), k, True
                    break
                if side == -1 and h[b] >= stop:
                    exit_px, held, stopped = max(o[b], stop), k, True
                    break
        net = side * (exit_px - entry) - spread_px[t + 1] - 2 * slip_px
        out.append((net, stopped, side, t + 1, held))
        i += 1 + held
    return out


def run(symbol, args):
    wf.set_point(symbol, args.raw)
    h1 = pd.read_csv(f"{args.data}/{symbol}_H1.csv", index_col=0, parse_dates=True)
    sp = pd.read_csv(f"{args.data}/{symbol}_H1_spread.csv", index_col=0, parse_dates=True)["spread"]
    spread_px = (sp.reindex(h1.index).ffill() * wf.POINT).to_numpy()
    X_all = build_h1(h1)
    atr = X_all["atr_14"].reindex(h1.index).to_numpy()
    o, h, l_ = h1["open"].to_numpy(), h1["high"].to_numpy(), h1["low"].to_numpy()
    hz = args.horizon
    fwd = (h1["close"].shift(-hz) - h1["close"]).dropna()
    yl, ys = (fwd > 0).astype(int), (fwd < 0).astype(int)
    common = X_all.index.intersection(fwd.index)
    X, yl, ys = X_all.loc[common], yl.loc[common], ys.loc[common]
    pos = h1.index.get_indexer(common)
    rows, start = [], args.train
    while start + args.test <= len(X):
        tr, te = slice(start - args.train, start - hz), slice(start, start + args.test)
        params = dict(n_estimators=200, learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                      num_leaves=15, min_child_samples=50, n_jobs=4, verbose=-1, random_state=args.seed)
        pl = LGBMClassifier(**params).fit(X.iloc[tr], yl.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
        ps = LGBMClassifier(**params).fit(X.iloc[tr], ys.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
        bl, bs = yl.iloc[tr].mean(), ys.iloc[tr].mean()
        for off in OFFSETS:
            for m in STOPS:
                sims = simulate_stop(pl - bl + 0.5, ps - bs + 0.5, o, h, l_, spread_px, atr, pos[te], 0.5 + off, hz,
                                     args.slip * wf.PIP, m)
                for net, stopped, side, eb, held in sims:
                    rows.append((symbol, off, m if m else 99.0, start, net / wf.PIP, stopped, side, h1.index[eb], held))
        start += args.test
    return pd.DataFrame(rows, columns=["symbol", "off", "stop_atr", "fold", "net", "stopped", "side", "entry", "held"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["USDJPY", "GBPJPY"])
    ap.add_argument("--data", default="data/historical_data")
    ap.add_argument("--raw", default="data/raw_export")
    ap.add_argument("--train", type=int, default=6000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--slip", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    allr = pd.concat([run(s, a) for s in a.symbols])
    allr.to_csv("results/walkforward_h1_stop_trades.csv", index=False)
    g = allr.groupby(["symbol", "stop_atr", "off"])
    res = g.agg(trades=("net", "size"), net=("net", "mean"), stop_hit=("stopped", "mean"), worst=("net", "min"))
    res["folds_pos"] = allr.groupby(["symbol", "stop_atr", "off", "fold"])["net"].sum().gt(0).groupby(
        ["symbol", "stop_atr", "off"]).mean() * 100
    # max drawdown of cumulative net pips over the whole test sequence
    res["max_dd"] = g["net"].apply(lambda s: float((s.cumsum().cummax() - s.cumsum()).max()))
    res = res.reset_index()
    summ = res.groupby(["symbol", "stop_atr"]).agg(
        net=("net", "mean"), folds_pos=("folds_pos", "mean"), stop_hit=("stop_hit", "mean"),
        worst=("worst", "min"), max_dd=("max_dd", "mean"), trades=("trades", "mean"))
    print("\nAveraged over offsets 0.02-0.14 (stop_atr 99 = no stop):")
    print(summ.round(2).to_string())
