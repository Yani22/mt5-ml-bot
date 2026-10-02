"""Experimental extra features for walk-forward tests (not used by the live bot).

Families: session/daily structure, volatility regime, range position, multi-horizon returns scaled by ATR,
and H4/D1 trend context built from H1 bars. Every feature at M5 bar t uses only information available at t's close.
Higher-timeframe bars are indexed by OPEN time, so they are shifted to close time before the as-of join.
"""
import numpy as np
import pandas as pd
import ta  # type: ignore

from src.h1_features import htf_trend as _htf_trend

DAY = 288  # M5 bars per day


def extra_features(m5: pd.DataFrame, h1: pd.DataFrame) -> pd.DataFrame:
    c, h, l_ = m5["close"], m5["high"], m5["low"]
    X = pd.DataFrame(index=m5.index)
    atr = ta.volatility.AverageTrueRange(h, l_, c, window=14).average_true_range()

    # --- session / daily structure (server-day based, causal) ---
    day = m5.index.normalize()
    day_open = m5["open"].groupby(day).transform("first")
    day_hi = h.groupby(day).cummax()
    day_lo = l_.groupby(day).cummin()
    X["dist_day_open_atr"] = (c - day_open) / (atr + 1e-12)
    X["day_range_pos"] = (c - day_lo) / (day_hi - day_lo + 1e-12)
    X["day_range_atr"] = (day_hi - day_lo) / (atr + 1e-12)
    daily = m5.groupby(day).agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    prev = daily.shift(1)  # previous completed trading day
    for col in ("high", "low", "close"):
        X[f"dist_prev_{col}_atr"] = (c - pd.Series(day.map(prev[col]), index=m5.index)) / (atr + 1e-12)
    X["bars_into_day"] = m5.groupby(day).cumcount()
    hour = m5.index.hour
    X["sess_asia"] = ((hour >= 0) & (hour < 9)).astype(int)
    X["sess_london"] = ((hour >= 9) & (hour < 16)).astype(int)
    X["sess_ny"] = ((hour >= 16) & (hour < 24)).astype(int)

    # --- volatility regime ---
    X["atr_ratio_1d"] = atr / (atr.rolling(DAY).mean() + 1e-12)
    X["atr_ratio_1w"] = atr / (atr.rolling(5 * DAY).mean() + 1e-12)
    rv = c.pct_change().rolling(DAY // 4).std()
    X["rv_pct_rank_1w"] = rv.rolling(5 * DAY).rank(pct=True)

    # --- range position over 1d / 1w ---
    for name, n in (("1d", DAY), ("1w", 5 * DAY)):
        lo, hi = l_.rolling(n).min(), h.rolling(n).max()
        X[f"range_pos_{name}"] = (c - lo) / (hi - lo + 1e-12)

    # --- multi-horizon returns scaled by ATR ---
    for n in (12, 48, 144, DAY):
        X[f"ret_{n}_atr"] = (c - c.shift(n)) / (atr + 1e-12)

    # --- H4 / D1 trend context from H1 ---
    for rule, delta, ema, tag in (("4h", "4h", 20, "h4"), ("1D", "1D", 20, "d1")):
        htf = _htf_trend(h1, rule, delta, ema, tag)
        X = pd.merge_asof(X.sort_index(), htf.sort_index(), left_index=True, right_index=True, direction="backward")

    return X.replace([np.inf, -np.inf], np.nan)
