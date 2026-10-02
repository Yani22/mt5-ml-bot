"""Resad's suggested baseline (email 2025-12-19): EMA21 crosses above EMA50 = buy, below = sell, M30/H1, with the
same exits as resad_recipe.py (TP 4 x ATR14, SL 2 x ATR14, cap 120 bars). No ML.

PRE-REGISTERED (written before the first run, 2026-10-02):
  * Confirmatory: USDJPY H1 2010-01 .. 2023-12. Folds = calendar years. PASS if mean net R > 0, year-bootstrap
    95% CI low > 0, >= 60% years positive, and swap-charged mean > 0.
  * Exploratory only (window already used today): GOLD H1 and M30 (resampled from M5), USDJPY/GBPJPY/EURUSD/GBPUSD/
    BTCUSD H1 on 2024-26.
  * FRESH HOLDOUT (--set fresh; pre-registered 2026-10-02, after the USDJPY result, before the data was exported):
    GOLD, GBPJPY, EURJPY H1 2010-01 .. 2023-12 (data/historical_old), rule and exits unchanged, no parameter search.
    Same per-symbol bar as above. The rule counts as validated only if >= 2 of the 3 symbols PASS (with 3 tries, a single
    pass is weak evidence). Even then: a GOLD stop at 0.01 lot is ~$30-39 (2025-26 ATR,
    100 oz contract assumed), too large for a $250 account; a pass would mean a demo forward test, not funding.
Signal at the close of bar t (cross between t-1 and t), fill at the open of t+1; one position at a time, a cross
while in a position is ignored. Costs and fills as in resad_recipe.first_touch (shorts on the ask, SL wins ties).

Usage: python scripts/resad_trend.py [--set fresh]
"""
import os
import sys

import numpy as np
import pandas as pd
import ta  # type: ignore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resad_recipe as rr  # noqa: E402


def cross_signals(close: pd.Series, fast=21, slow=50) -> np.ndarray:
    """+1 where EMA fast crosses above slow at bar t, -1 below, else 0. Uses bars <= t only."""
    f, s = ta.trend.ema_indicator(close, fast), ta.trend.ema_indicator(close, slow)
    above = (f > s).astype(int).to_numpy()
    sig = np.zeros(len(close), dtype=int)
    valid = (f.notna() & s.notna()).to_numpy()
    sig[1:] = np.where(valid[1:] & valid[:-1], above[1:] - above[:-1], 0)
    return sig


def backtest(bars, sp, cap=120, slip_frac=0.1, swap_r_per_bar=0.0):
    o, h, l_, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
    atr = ta.volatility.AverageTrueRange(bars.high, bars.low, bars.close, window=14).average_true_range().to_numpy()
    sig = cross_signals(bars.close)
    slip = slip_frac * float(np.median(sp))
    out, t, n = [], 0, len(c)
    while t < n - cap - 2:
        side = sig[t]
        if side == 0 or not np.isfinite(atr[t]) or atr[t] <= 0:
            t += 1
            continue
        entry = o[t + 1] + (sp[t + 1] if side == 1 else 0.0)
        risk = rr.SL_ATR * atr[t]
        r = rr.first_touch(side, entry, entry + side * rr.TP_ATR * atr[t], entry - side * risk, o, h, l_, sp, t + 1, cap)
        if r is None:
            break
        exit_px, held, outcome = r
        net = side * (exit_px - entry) - 2 * slip
        out.append((bars.index[t + 1], side, net / risk, held, outcome))
        t = t + 1 + held
    tr = pd.DataFrame(out, columns=["entry", "side", "net_r", "held", "outcome"])
    tr["net_pts"] = 0.0
    tr["fold"] = tr.entry.dt.year
    return tr


def load(symbol, tf, d, end=None):
    if tf == "M30":
        m5 = pd.read_csv(f"{d}/{symbol}_M5.csv", index_col=0, parse_dates=True)
        s5 = pd.read_csv(f"{d}/{symbol}_M5_spread.csv", index_col=0, parse_dates=True)["spread"].reindex(m5.index)
        bars = m5.resample("30min").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        sps = s5.resample("30min").first().reindex(bars.index).ffill().fillna(0)
    else:
        bars = pd.read_csv(f"{d}/{symbol}_{tf}.csv", index_col=0, parse_dates=True)
        sps = pd.read_csv(f"{d}/{symbol}_{tf}_spread.csv", index_col=0, parse_dates=True)["spread"]
        sps = sps.reindex(bars.index).ffill().fillna(0)
    sps = rr.hour_spread(sps)
    if end:
        keep = bars.index <= pd.Timestamp(end, tz=bars.index.tz) + pd.Timedelta("23h59min")
        bars, sps = bars[keep], sps[keep]
    return bars, sps.to_numpy() * rr.point_size(symbol, ["data/raw_export", "data/raw_export/old"])


if __name__ == "__main__":
    fresh = sys.argv[1:] == ["--set", "fresh"]
    if fresh:
        cells = [(s, "H1", "data/historical_old", "2023-12-31", "FRESH") for s in ("GOLD", "GBPJPY", "EURJPY")]
    else:
        cells = [("USDJPY", "H1", "data/historical_old", "2023-12-31", "CONFIRMATORY")]
        cells += [(s, "H1", "data/historical_data", None, "explore") for s in
                  ("GOLD", "USDJPY", "GBPJPY", "EURUSD", "GBPUSD", "BTCUSD")]
        cells += [("GOLD", "M30", "data/historical_data", None, "explore")]
    res = []
    for sym, tf, d, end, kind in cells:
        bars, sp = load(sym, tf, d, end)
        per_day = 24 if tf == "H1" else 48
        t = backtest(bars, sp)
        j = rr.judge(t, 0.02 / per_day)
        ok = j["trades"] > 0 and j["mean_r"] > 0 and j["ci_lo"] > 0 and j["folds_pos"] >= 60 and j["mean_r_swap"] > 0
        res.append(dict(kind=kind, symbol=sym, tf=tf, verdict="PASS" if ok else "FAIL", **j))
    summ = pd.DataFrame(res).drop(columns=["pts"])
    summ.to_csv(f"results/resad_trend_{'fresh' if fresh else 'summary'}.csv", index=False)
    print(summ.round(3).to_string(index=False))
    if fresh:
        n = int((summ.verdict == "PASS").sum())
        print(f"{n} of 3 symbols pass -> rule {'VALIDATED (demo forward test next)' if n >= 2 else 'NOT validated'}")
