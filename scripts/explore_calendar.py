"""Exploration 3: documented calendar/time-of-day effects, no model, no fitted parameters.

Server time -> UTC: the server day rolls at 17:00 New York (00:00 server), so NY wall clock = server - 7h and UTC is
that NY time converted with US daylight saving. Windows are in UTC hours [a, b): enter at the open of the bar whose UTC
hour is a, exit at the open of the bar whose UTC hour is b. Cost = bar spread + 2 x slippage (0.1 pip each side).

Hypotheses fixed in advance (sign from Breedon & Ranaldo 2011: a currency depreciates in its own trading hours):
  Asia [0,8) UTC, Europe [7,15), US [13,21).  USDJPY: Asia long, US short.  EURUSD/GBPUSD: Europe short, US long.
  USDCHF: Europe long, US short.  AUDUSD: Asia short, US long.   Gotobi (USDJPY): long [0,1) UTC on the 5th/10th/15th/
  20th/25th/month-end (weekend dates move to the prior Friday).
A hypothesis passes if net pips/trade > 0 and >= 3 of 5 half-year blocks are positive.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import walkforward as wf  # noqa: E402

WINDOWS = {"Asia": (0, 8), "Europe": (7, 15), "US": (13, 21)}
HYPOTHESES = [("USDJPY", "Asia", 1), ("USDJPY", "US", -1), ("EURUSD", "Europe", -1), ("EURUSD", "US", 1),
              ("GBPUSD", "Europe", -1), ("GBPUSD", "US", 1), ("USDCHF", "Europe", 1), ("USDCHF", "US", -1),
              ("AUDUSD", "Asia", -1), ("AUDUSD", "US", 1)]


def load(symbol, data):
    h1 = pd.read_csv(f"{data}/{symbol}_H1.csv", index_col=0, parse_dates=True)
    sp = pd.read_csv(f"{data}/{symbol}_H1_spread.csv", index_col=0, parse_dates=True)["spread"]
    h1.index = h1.index.tz_localize(None) if h1.index.tz is not None else h1.index
    sp.index = h1.index
    ny = (h1.index - pd.Timedelta("7h")).tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward")
    h1 = h1.assign(spread=sp.to_numpy() * wf.POINT, utc=ny.tz_convert("UTC"))
    h1 = h1[h1.utc.notna()]
    h1["uhour"], h1["udate"] = h1.utc.dt.hour, h1.utc.dt.date
    return h1


def window_trades(h1, a, b, sign, slip_px, dates=None):
    """One trade per UTC date: open of hour a -> open of hour b (needs both bars that date; b==24 wraps to next)."""
    rows = []
    for d, g in h1.groupby("udate"):
        if dates is not None and d not in dates:
            continue
        ent = g[g.uhour == a]
        ext = g[g.uhour == b] if b < 24 else None
        if ent.empty or ext is None or ext.empty:
            continue
        e, x = ent.iloc[0], ext.iloc[0]
        if (x.utc - e.utc) != pd.Timedelta(hours=b - a):  # a gap (weekend/holiday) inside the window
            continue
        rows.append((e.name, sign * (x.open - e.open), sign * (x.open - e.open) - e.spread - 2 * slip_px))
    return pd.DataFrame(rows, columns=["time", "gross", "net"])


def gotobi_dates(h1):
    days = pd.Series(sorted(set(h1.udate)))
    cal = pd.to_datetime(days)
    out = set()
    for ym, g in cal.groupby([cal.dt.year, cal.dt.month]):
        y, m = ym
        last = (pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)).day
        for dd in (5, 10, 15, 20, 25, last):
            t = pd.Timestamp(y, m, dd)
            while t.weekday() >= 5:
                t -= pd.Timedelta("1D")
            out.add(t.date())
    return out


def report(name, tr, pip):
    if len(tr) < 30:
        return (name, len(tr), np.nan, np.nan, 0, False)
    blocks = tr.groupby(pd.cut(tr.time, 5, labels=False)).net.sum()
    ok = tr.net.mean() > 0 and (blocks > 0).sum() >= 3
    return (name, len(tr), tr.gross.mean() / pip, tr.net.mean() / pip, int((blocks > 0).sum()), ok)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/historical_data")
    ap.add_argument("--raw", default="data/raw_export")
    ap.add_argument("--slip", type=float, default=0.1)
    a = ap.parse_args()
    rows, cache = [], {}
    for sym, win, sign in HYPOTHESES:
        wf.set_point(sym, a.raw)
        h1 = cache.setdefault(sym, load(sym, a.data))
        lo, hi = WINDOWS[win]
        rows.append(report(f"{sym} {win} {'long' if sign > 0 else 'short'}",
                           window_trades(h1, lo, hi, sign, a.slip * wf.PIP), wf.PIP))
    wf.set_point("USDJPY", a.raw)
    h1 = cache["USDJPY"]
    rows.append(report("USDJPY gotobi long [0,1)", window_trades(h1, 0, 1, 1, a.slip * wf.PIP, gotobi_dates(h1)), wf.PIP))
    r = pd.DataFrame(rows, columns=["hypothesis", "trades", "gross_pips", "net_pips", "blocks_pos_of_5", "PASS"])
    r.to_csv("results/explore_calendar.csv", index=False)
    print(r.round(2).to_string(index=False))
