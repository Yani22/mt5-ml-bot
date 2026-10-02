"""H1 features for the USDJPY 24h strategy: the bot's static/dynamic features plus H4/D1 trend context.

Shared by the research scripts and the live H1 bot so both build identical inputs. Features at bar t use only bars
<= t; higher-timeframe bars are indexed by open time and shifted to close time before the as-of join.
NOTE: some inputs are running sums (OBV) or recursive (EMA/ADX), so they depend on where the series STARTS. Always
build on the same fixed-origin history (data/historical_data/<SYMBOL>_H1.csv, from 2024-01-02), never a sliding window.
"""
import numpy as np
import pandas as pd
import ta  # type: ignore

from src.config import FeatureCfg, PriceActionCfg
from src.features import build_dynamic_features, build_static_features

WARMUP = 300  # rows dropped at the start while indicators warm up


def htf_trend(h1: pd.DataFrame, rule: str, delta: str, ema: int, tag: str) -> pd.DataFrame:
    bars = h1.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    f = pd.DataFrame(index=bars.index)
    e = ta.trend.ema_indicator(bars["close"], window=ema)
    f[f"{tag}_dist_ema"] = (bars["close"] - e) / bars["close"]
    f[f"{tag}_ema_slope"] = e.pct_change(3)
    f[f"{tag}_rsi"] = ta.momentum.rsi(bars["close"], window=14)
    f[f"{tag}_range_pos"] = (bars["close"] - bars["low"].rolling(10).min()) / (
        bars["high"].rolling(10).max() - bars["low"].rolling(10).min() + 1e-12)
    f.index = f.index + pd.Timedelta(delta)  # bar close time = when its values are known
    return f


def build_h1(h1: pd.DataFrame) -> pd.DataFrame:
    cfg = FeatureCfg(timeframe_minutes=60)
    X = build_dynamic_features(h1, build_static_features(h1, "H1", pa_cfg=PriceActionCfg()), cfg, "H1")
    for rule, delta, ema, tag in (("4h", "4h", 20, "h4"), ("1D", "1D", 20, "d1")):
        htf = htf_trend(h1, rule, delta, ema, tag)
        X = pd.merge_asof(X.sort_index(), htf.sort_index(), left_index=True, right_index=True, direction="backward")
    atr = X["atr_14"] + 1e-12
    for n in (4, 12, 24, 120):
        X[f"ret_{n}_atr"] = (h1["close"] - h1["close"].shift(n)) / atr
    return X.replace([np.inf, -np.inf], np.nan).iloc[WARMUP:]
