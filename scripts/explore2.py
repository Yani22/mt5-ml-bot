"""Exploration 4 (strict process): three simple intraday rules on 16 years of H1 data (2010-2026), no fitted parameters.

Costs: a long pays the spread at ENTRY, a short pays the spread at EXIT (it buys back at the ask); old bars have no
recorded spread, so the spread used is the median recorded spread for that UTC hour in 2024+ (captures rollover
widening). Slippage 0.1 pip per side.
Rules (fixed in advance):
  breakout : range of the 00-06 UTC bars; bar 07 UTC closes beyond it -> enter at the 08 UTC open in that direction,
             stop at the opposite range boundary, exit at the 16 UTC open.
  fade     : z = (close[t]-close[t-4]) / (2*ATR14[t]); |z| > 1.5 -> trade AGAINST the move, enter next open, exit 4 bars later.
  continue : same trigger, trade WITH the move.
Pass (holdout 2010-2023, BOTH pairs): mean net > 0, 95% year-bootstrap CI low > 0, >= 60% of years positive, and
2024+ mean net >= 0.
"""
import os
import sys

import numpy as np
import pandas as pd
import ta  # type: ignore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explore_calendar as ec  # noqa: E402
import walkforward as wf  # noqa: E402

SLIP = float(os.environ.get("SLIP_PIPS", "0.1"))


def prep(sym):
    wf.set_point(sym, "data/raw_export/old")
    h1 = ec.load(sym, "data/historical_old").reset_index(names="srv")
    h1["atr"] = ta.volatility.AverageTrueRange(h1.high, h1.low, h1.close, window=14).average_true_range()
    recent = h1[h1.srv >= "2024-01-01"]
    hour_sp = recent.groupby("uhour").spread.median()
    h1["sp_hr"] = h1.uhour.map(hour_sp) * float(os.environ.get("SPREAD_MULT", "1"))  # what-if: cheaper account
    return h1


def cost(side, sp_entry, sp_exit):
    return (sp_entry if side == 1 else sp_exit) + 2 * SLIP * wf.PIP


def rule_breakout(h):
    out = []
    for _, g in h.groupby("udate"):
        asia, sig = g[g.uhour <= 6], g[g.uhour == 7]
        ent, ext = g[g.uhour == 8], g[g.uhour == 16]
        if len(asia) < 6 or sig.empty or ent.empty or ext.empty:
            continue
        hi, lo, c = asia.high.max(), asia.low.min(), sig.iloc[0].close
        side = 1 if c > hi else -1 if c < lo else 0
        if side == 0:
            continue
        e, x = ent.iloc[0], ext.iloc[0]
        stop, exit_px, sp_exit = (lo if side == 1 else hi), x.open, x.sp_hr
        for _, b in g[(g.uhour >= 8) & (g.uhour < 16)].iterrows():
            if (side == 1 and b.low <= stop) or (side == -1 and b.high >= stop):
                exit_px, sp_exit = stop, b.sp_hr
                break
        out.append((e.srv, side * (exit_px - e.open) - cost(side, e.sp_hr, sp_exit)))
    return pd.DataFrame(out, columns=["time", "net"])


def rule_move(h, fade):
    o, c, atr, sp = (h[k].to_numpy() for k in ("open", "close", "atr", "sp_hr"))
    z = (pd.Series(c) - pd.Series(c).shift(4)).to_numpy() / (2 * atr)
    out, t = [], 20
    while t < len(h) - 6:
        if np.isnan(z[t]) or abs(z[t]) <= 1.5:
            t += 1
            continue
        side = int(-np.sign(z[t]) if fade else np.sign(z[t]))
        out.append((h.srv.iloc[t + 1], side * (o[t + 5] - o[t + 1]) - cost(side, sp[t + 1], sp[t + 5])))
        t += 5
    return pd.DataFrame(out, columns=["time", "net"])


def judge(tr):
    ho, rc = tr[tr.time < "2024-01-01"], tr[tr.time >= "2024-01-01"]
    yr = ho.groupby(ho.time.dt.year).net.agg(["sum", "size"])
    rng = np.random.default_rng(0)
    boots = [(lambda s: s["sum"].sum() / s["size"].sum())(yr.iloc[rng.integers(0, len(yr), len(yr))]) for _ in range(2000)]
    lo = np.percentile(boots, 2.5) / wf.PIP
    mean, pos = ho.net.mean() / wf.PIP, (yr["sum"] > 0).mean() * 100
    rmean = rc.net.mean() / wf.PIP if len(rc) else np.nan
    ok = mean > 0 and lo > 0 and pos >= 60 and rmean >= 0
    return dict(n=len(ho), net=round(mean, 2), ci_lo=round(lo, 2), years_pos=round(pos), net_2024=round(rmean, 2), ok=bool(ok))


if __name__ == "__main__":
    res = {}
    for sym in ("USDJPY", "USDCHF"):
        h = prep(sym)
        for name, fn in (("breakout", rule_breakout), ("fade", lambda d: rule_move(d, True)),
                         ("continue", lambda d: rule_move(d, False))):
            r = judge(fn(h))
            res[(name, sym)] = r
            print(f"{name:9s} {sym}: {r}", flush=True)
    for name in ("breakout", "fade", "continue"):
        print(f"{name}: {'PASS' if all(res[(name, s)]['ok'] for s in ('USDJPY', 'USDCHF')) else 'FAIL'}")
