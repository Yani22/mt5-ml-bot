"""Robustness of the USDCHF 'short 13:00 -> 21:00 UTC' rule. Pre-registered bars (fixed before running), judged on the
2010-2023 holdout: (A) net > 0 with a 2.0 pip spread and >= 3/5 blocks positive; (B) net > 0 in >= 7 of the 9 windows
entry in {12,13,14} x exit in {20,21,22}; (C) net > 0 for >= 2 of the ATR stops {2,3,4}. 2024+ is shown for information.
"""
import os
import sys

import numpy as np
import pandas as pd
import ta  # type: ignore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explore_calendar as ec  # noqa: E402
import walkforward as wf  # noqa: E402

SLIP = 0.1


def prep(floor_pts):
    wf.set_point("USDCHF", "data/raw_export/old")
    h1 = ec.load("USDCHF", "data/historical_old")
    h1["spread"] = np.maximum(h1.spread.to_numpy(), floor_pts * wf.POINT)
    h1["atr"] = ta.volatility.AverageTrueRange(h1.high, h1.low, h1.close, window=14).average_true_range()
    return h1


def trades(h1, a, b, k=None):
    rows = []
    for _, g in h1.groupby("udate"):
        ent, ext = g[g.uhour == a], g[g.uhour == b]
        if ent.empty or ext.empty:
            continue
        e, x = ent.iloc[0], ext.iloc[0]
        if x.utc - e.utc != pd.Timedelta(hours=b - a):
            continue
        exit_px = x.open
        if k is not None:
            stop = e.open + k * e.atr  # short: stop above entry
            for _, bar in g[(g.utc >= e.utc) & (g.utc < x.utc)].iterrows():
                if bar.high >= stop:
                    exit_px = max(bar.open, stop) if bar.utc > e.utc else stop
                    break
        rows.append((e.name, e.open - exit_px - e.spread - 2 * SLIP * wf.PIP))
    return pd.DataFrame(rows, columns=["time", "net"])


def summ(tr, split="2024-01-01"):
    out = {}
    for name, d in (("holdout<=2023", tr[tr.time < split]), ("2024+", tr[tr.time >= split])):
        if len(d) < 50:
            out[name] = (len(d), np.nan, 0)
            continue
        blocks = d.groupby(pd.cut(d.time, 5, labels=False)).net.sum()
        out[name] = (len(d), d.net.mean() / wf.PIP, int((blocks > 0).sum()))
    return out


if __name__ == "__main__":
    print("A. spread sensitivity (window 13-21):  floor pips -> holdout net (blocks+/5) | 2024+ net")
    okA = False
    for pips in (1.2, 1.5, 2.0, 2.5, 3.0):
        s = summ(trades(prep(pips * 10), 13, 21))
        h, r = s["holdout<=2023"], s["2024+"]
        print(f"  {pips:.1f}: {h[1]:+.2f} ({h[2]}/5) n={h[0]} | {r[1]:+.2f}")
        okA = okA or (pips == 2.0 and h[1] > 0 and h[2] >= 3)
    base = prep(12)
    print("B. window shifts (entry-exit UTC): holdout net (blocks+) | 2024+ net")
    pos = 0
    for a in (12, 13, 14):
        for b in (20, 21, 22):
            s = summ(trades(base, a, b))
            h, r = s["holdout<=2023"], s["2024+"]
            pos += h[1] > 0
            print(f"  {a}-{b}: {h[1]:+.2f} ({h[2]}/5) | {r[1]:+.2f}")
    print("C. ATR stop on the 13-21 short: k -> holdout net, worst-case n | 2024+ net")
    okC = 0
    for k in (2, 3, 4):
        s = summ(trades(base, 13, 21, k))
        h, r = s["holdout<=2023"], s["2024+"]
        okC += h[1] > 0
        print(f"  {k}x ATR: {h[1]:+.2f} ({h[2]}/5) | {r[1]:+.2f}")
    t = trades(base, 13, 21)
    t = t[t.time < "2024-01-01"]
    print("by weekday net pips:", t.groupby(t.time.dt.dayofweek).net.mean().div(wf.PIP).round(2).to_dict())
    yr = t.groupby(t.time.dt.year).net.sum()
    print("block-bootstrap by year, 95% CI of mean net:", np.round(np.percentile(
        [t.net[t.time.dt.year.isin(np.random.default_rng(i).choice(yr.index, len(yr)))].mean() / wf.PIP for i in range(1000)],
        [2.5, 97.5]), 2))
    print(f"\nA(2.0 pip) {'PASS' if okA else 'FAIL'} | B windows positive {pos}/9 {'PASS' if pos >= 7 else 'FAIL'} | "
          f"C stops positive {okC}/3 {'PASS' if okC >= 2 else 'FAIL'}")
