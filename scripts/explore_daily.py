"""Exploration 5: daily/monthly trend-following on a basket of FX pairs, 2010-2026 (no fitted parameters).

PRE-REGISTERED before any real data was run (only synthetic-data unit tests existed):
  tsmom   : at each month-end close, position per pair = sign of the past L-month return (L in {3, 6, 12}); hold one month.
  donchian: long when the daily close exceeds the highest high of the past N days, short below the lowest low
            (N in {50, 100, 200}); hold until the opposite signal.
Costs: entry and exit each pay the spread (median recorded spread of the 00:00-server hour in 2024+, i.e. the rollover
hour, which is conservative) plus 0.1 pip slippage per side; returns are in % of price so pairs are comparable.
Swap is NOT included (status unresolved) and is reported as a caveat.
PASS for a variant: equal-weight basket, 2010-2023: mean monthly net return > 0 with 95% year-bootstrap CI low > 0,
>= 60% of calendar years positive, and 2024+ mean >= 0.  A family passes if >= 2 of its 3 variants pass.
Also reported: annualised Sharpe and max drawdown of the basket (in % of notional), and the number of pairs positive.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SLIP_PIPS = 0.1


def pair_cost_pct(spread_px: float, price: float, pip: float) -> float:
    """Round-trip cost as a fraction of price: entry + exit spread plus slippage on both sides."""
    return (2 * spread_px + 2 * SLIP_PIPS * pip) / price


def tsmom_positions(close: pd.Series, lookback_months: int) -> pd.Series:
    """Daily position series: sign of the past L-month return, decided at month-end close, held to the next month-end.
    The position for day d only uses closes up to the previous month-end (no look-ahead)."""
    me = close.groupby(close.index.to_period("M")).tail(1)  # last close of each month
    sig = np.sign(me / me.shift(lookback_months) - 1)
    pos = sig.reindex(close.index).ffill().shift(1)  # act from the next day
    # only update on month-ends: values already constant between month-ends because sig is sparse then ffilled
    return pos.fillna(0)


def donchian_positions(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    up = high.shift(1).rolling(n).max()
    dn = low.shift(1).rolling(n).min()
    raw = pd.Series(np.where(close > up, 1.0, np.where(close < dn, -1.0, np.nan)), index=close.index)
    return raw.ffill().shift(1).fillna(0)  # decided at today's close, applied from tomorrow


def daily_pnl_pct(open_: pd.Series, pos: pd.Series, cost_per_change: float) -> pd.Series:
    """Position held from open to next open. P&L in % of price; each position change pays cost_per_change per unit."""
    ret = open_.shift(-1) / open_ - 1
    gross = pos * ret
    turns = pos.diff().abs().fillna(pos.abs())
    return (gross - turns * cost_per_change / 2).dropna()  # cost_per_change is a round trip: half on each flip leg


def judge(daily: pd.Series, split="2024-01-01") -> dict:
    m = daily.groupby(daily.index.to_period("M")).sum()
    ho, rc = m[m.index < pd.Period(split, "M")], m[m.index >= pd.Period(split, "M")]
    yr = ho.groupby(ho.index.year).sum()
    rng = np.random.default_rng(0)
    boots = [yr.iloc[rng.integers(0, len(yr), len(yr))].mean() / 12 for _ in range(2000)]
    mean_m, lo = ho.mean(), np.percentile(boots, 2.5)
    years_pos = (yr > 0).mean() * 100
    d = daily[daily.index < split]
    sharpe = d.mean() / d.std() * np.sqrt(252) if d.std() > 0 else np.nan
    dd = float((d.cumsum().cummax() - d.cumsum()).max())
    rmean = rc.mean() if len(rc) else np.nan
    ok = bool(mean_m > 0 and lo > 0 and years_pos >= 60 and rmean >= 0)
    return dict(monthly_pct=round(mean_m * 100, 3), ci_lo=round(lo * 100, 3), years_pos=round(years_pos),
                net_2024=round(rmean * 100, 3), sharpe=round(sharpe, 2), maxdd_pct=round(dd * 100, 1), ok=ok)


def run(pairs: dict) -> None:
    """pairs: name -> (DataFrame with open/high/low/close indexed by date, spread_price, pip)."""
    for fam, params in (("tsmom", (3, 6, 12)), ("donchian", (50, 100, 200))):
        passed = 0
        for p in params:
            series = []
            for name, (d, sp, pip) in pairs.items():
                pos = (tsmom_positions(d.close, p) if fam == "tsmom"
                       else donchian_positions(d.high, d.low, d.close, p))
                c = pair_cost_pct(sp, float(d.close.median()), pip)
                series.append(daily_pnl_pct(d.open, pos, c).rename(name))
            basket = pd.concat(series, axis=1).mean(axis=1)  # equal weight over pairs that have data that day
            r = judge(basket)
            passed += r["ok"]
            print(f"{fam} {p}: {r}", flush=True)
        print(f"{fam}: {'PASS' if passed >= 2 else 'FAIL'} ({passed}/3 variants)\n")


def load_daily(n: str):
    """Daily bars from the MT5 'Daily' export, or built from the old H1 export (server-day grouping)."""
    base = "data/raw_export/old"
    for path in (f"{base}/{n}#_Daily.csv", f"{base}/{n}#_D1.csv"):
        if os.path.exists(path):
            raw = pd.read_csv(path, sep="\t")
            raw.columns = [c.strip("<>").lower() for c in raw.columns]
            raw["t"] = pd.to_datetime(raw["date"], format="%Y.%m.%d")
            d = raw.set_index("t")[["open", "high", "low", "close"]].sort_index()
            return d[~d.index.duplicated(keep="last")]
    path = f"{base}/{n}#_H1.csv"
    if not os.path.exists(path):
        return None
    raw = pd.read_csv(path, sep="\t")
    raw.columns = [c.strip("<>").lower() for c in raw.columns]
    raw["t"] = pd.to_datetime(raw["date"] + " " + raw["time"], format="%Y.%m.%d %H:%M:%S")
    h = raw.set_index("t").sort_index()
    return h.groupby(h.index.normalize()).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                                              close=("close", "last"))


if __name__ == "__main__":
    import walkforward as wf  # noqa: E402
    names = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY", "EURJPY", "GBPJPY", "AUDJPY",
             "CADJPY", "CHFJPY", "NZDJPY"]
    pairs = {}
    for n in names:
        d = load_daily(n)
        if d is None:
            print("missing daily data for", n)
            continue
        wf.set_point(n, "data/raw_export")
        sp = pd.read_csv(f"data/historical_data/{n}_H1_spread.csv", index_col=0, parse_dates=True)["spread"]
        sp = sp[sp.index.hour == 0]  # the 00:00 server bar is the rollover hour (widest spreads)
        pairs[n] = (d, float(sp.median()) * wf.POINT, wf.PIP)
    run(pairs)
