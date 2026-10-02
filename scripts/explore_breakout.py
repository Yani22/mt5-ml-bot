"""Exploration 1: rule-based H4 breakout / trend-following. No model, no fitted parameters.

Signal at the close of H4 bar t: long if close > highest high of the previous N bars (short mirrored). Enter at the
open of bar t+1, hard stop 2 x ATR14(t), time exit after H bars (a stop frees the slot). One position at a time.
Spread = exported H1 spread at the entry time, slippage 0.1 pip per side.

Pre-registered bar (fixed before running): grid N in {20,40,80} x H in {12,30}; a variant passes if net pips/trade > 0
and >= 4 of 6 equal time blocks are positive; a symbol passes if >= 5 of the 6 variants pass.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import ta  # type: ignore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import walkforward as wf  # noqa: E402

NS, HS, STOP = (20, 40, 80), (12, 30), 2.0


def h4_bars(symbol, data):
    h1 = pd.read_csv(f"{data}/{symbol}_H1.csv", index_col=0, parse_dates=True)
    sp = pd.read_csv(f"{data}/{symbol}_H1_spread.csv", index_col=0, parse_dates=True)["spread"]
    b = h1.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    b["spread"] = sp.reindex(b.index).ffill() * wf.POINT
    b["atr"] = ta.volatility.AverageTrueRange(b["high"], b["low"], b["close"], window=14).average_true_range()
    return b


def trades(b, n, hold, slip_px):
    o, h, l_, c = (b[k].to_numpy() for k in ("open", "high", "low", "close"))
    hi = b["high"].shift(1).rolling(n).max().to_numpy()
    lo = b["low"].shift(1).rolling(n).min().to_numpy()
    atr, sp, idx = b["atr"].to_numpy(), b["spread"].to_numpy(), b.index
    out, t = [], n + 14
    while t < len(b) - hold - 2:
        side = 1 if c[t] > hi[t] else -1 if c[t] < lo[t] else 0
        if side == 0 or np.isnan(atr[t]):
            t += 1
            continue
        entry = o[t + 1]
        stop = entry - side * STOP * atr[t]
        exit_px, held = o[t + 1 + hold], hold
        for k in range(1, hold + 1):
            j = t + k
            if (side == 1 and l_[j] <= stop) or (side == -1 and h[j] >= stop):
                exit_px, held = (min(o[j], stop) if side == 1 else max(o[j], stop)), k
                break
        out.append((idx[t + 1], side * (exit_px - entry) - sp[t + 1] - 2 * slip_px))
        t += 1 + held
    return pd.DataFrame(out, columns=["time", "net"])


def run(symbol, args):
    wf.set_point(symbol, args.raw)
    b = h4_bars(symbol, args.data)
    rows = []
    for n in NS:
        for hold in HS:
            tr = trades(b, n, hold, args.slip * wf.PIP)
            if len(tr) < 30:
                rows.append((symbol, n, hold, len(tr), np.nan, 0, False))
                continue
            blocks = tr.groupby(pd.cut(tr.time, 6, labels=False)).net.sum()
            ok = tr.net.mean() > 0 and (blocks > 0).sum() >= 4
            rows.append((symbol, n, hold, len(tr), tr.net.mean() / wf.PIP, int((blocks > 0).sum()), ok))
    return pd.DataFrame(rows, columns=["symbol", "N", "hold", "trades", "net_pips", "blocks_pos", "pass"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--data", default="data/historical_data")
    ap.add_argument("--raw", default="data/raw_export")
    ap.add_argument("--slip", type=float, default=0.1)
    a = ap.parse_args()
    r = pd.concat([run(s, a) for s in a.symbols])
    r.to_csv("results/explore_breakout.csv", index=False)
    s = r.groupby("symbol").agg(variants_pass=("pass", "sum"), mean_net=("net_pips", "mean"), trades=("trades", "mean"))
    s["PASS"] = s.variants_pass >= 5
    print(s.sort_values("variants_pass", ascending=False).round(2).to_string())
