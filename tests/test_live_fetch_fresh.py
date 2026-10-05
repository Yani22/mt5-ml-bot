"""K21: when the 200-bar MT5 fetch comes back empty, `fetch_live` used to fall back to the cached history, so the decision ran
on an old bar against a live tick. It now returns nothing (the caller skips the bar), and does the same when the feature
build drops the newest bar. Bar labels are compared, never the wall clock (bar times are server time labelled UTC)."""
from types import SimpleNamespace as NS
from unittest.mock import patch

import pandas as pd
from loguru import logger

import src.data_manager as dmod
from src.data_manager import DataManager
from src.symbol_processor import SymbolProcessor


def frame(n, start=0):
    idx = pd.date_range("2026-01-01", periods=n + start, freq="5min", tz="UTC")[start:]
    return pd.DataFrame({"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 1}, index=idx)


def make(cached, recent, drop_last_feature_row=False):
    d = object.__new__(DataManager)
    d.cfg = NS(timeframe="M5", history_bars=100, fetch=NS(save_raw_data_locally=False),
               context_features=NS(mta=NS(enabled=False), inter_market=NS(enabled=False)))
    d.load_local_history = lambda *a, **k: cached.copy()
    d._fetch_bars_from_mt5_chunked = lambda *a, **k: recent.copy()

    def build(data, *a, **k):
        X = data[["close"]].copy()
        return X.iloc[:-1] if drop_last_feature_row else X

    return d, patch.object(dmod, "build_features", build)


def run(d, patcher):
    with patcher:
        return d.fetch_live("EURUSD#", None)


def test_an_empty_mt5_fetch_does_not_fall_back_to_the_cached_history():
    lines = []
    sink = logger.add(lambda m: lines.append(str(m)), format="{level} {message}", level="DEBUG")
    try:
        data, X, y = run(*make(frame(50), pd.DataFrame()))
    finally:
        logger.remove(sink)
    assert data.empty and X.empty
    assert any(line.startswith("WARNING") and "stale" in line for line in lines)


def test_a_fetch_ending_at_the_newest_bar_returns_data_ending_there():
    cached, recent = frame(50), frame(10, start=45)
    data, X, y = run(*make(cached, recent))
    assert data.index[-1] == recent.index[-1] and X.index[-1] == recent.index[-1]


def test_a_feature_build_that_drops_the_newest_bar_returns_nothing():
    data, X, y = run(*make(frame(50), frame(10, start=45), drop_last_feature_row=True))
    assert data.empty and X.empty


def test_the_processor_skips_the_bar_when_the_fetch_returns_nothing():
    sp = object.__new__(SymbolProcessor)
    sp.symbol, sp.feature_cfg = "EURUSD#", None
    sp.data_manager = NS(fetch_live=lambda *a: (pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))
    assert sp._fetch_and_prepare_data() == (None, None, None)
