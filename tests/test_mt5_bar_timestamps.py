"""MT5-fetched bars were indexed `datetime64[s, UTC]` while the cached CSV bars load as `[us, UTC]`. Under pandas 3 a merge of
the two units raises (`incompatible merge keys`), which stopped `trainer.py` on the MT5 data path with the H1 context on. The unit
tests and any CSV-only run never see it: only the MT5 fetch produces `[s]`. The fetch now returns the unit the CSV path has."""
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch

import pandas as pd
import pytest

from src.config import MtaCfg
from src.data_manager import DataManager
from src.features import add_contextual_features

STEP = {"M5": 300, "H1": 3600}


@pytest.fixture
def dm(tmp_path):
    d = object.__new__(DataManager)
    d.cfg, d.raw_data_dir = NS(timeframe="M5"), str(tmp_path)
    return d


def terminal():
    mt5 = sys.modules["MetaTrader5"]

    def copy_rates_from_pos(symbol, tf, start, count):
        step = 3600 if tf == mt5.TIMEFRAME_H1 else 300
        return [{"time": 1_700_000_000 + i * step, "open": 1.0 + i * 1e-5, "high": 1.1, "low": 0.9, "close": 1.0 + i * 1e-5,
                 "tick_volume": 1} for i in range(count)]

    return patch.object(mt5, "copy_rates_from_pos", copy_rates_from_pos, create=True)


def test_fetched_bars_have_the_same_timestamp_unit_as_the_cached_csv(dm):
    with terminal():
        fetched = dm._fetch_bars_from_mt5_chunked("EURUSD#", "M5", 50)
    dm.append_new_bars("EURUSD#", fetched)
    cached = dm.load_local_history("EURUSD#", "M5")
    assert fetched.index.dtype == cached.index.dtype
    assert list(fetched.index) == list(cached.index)          # the same instants, written and read back unchanged


def test_the_h1_context_joins_onto_mt5_fetched_m5_bars(dm):
    with terminal():
        m5 = dm._fetch_bars_from_mt5_chunked("EURUSD#", "M5", 200)
        h1 = dm._fetch_bars_from_mt5_chunked("EURUSD#", "H1", 120)
    out = add_contextual_features(m5, mta_df=h1, mta_cfg=MtaCfg(enabled=True, timeframe="H1"))   # raised MergeError
    assert len(out) == len(m5) and "mta_rsi_14" in out.columns


def test_the_h1_context_joins_onto_cached_m5_bars_too(dm):
    """The live loop mixes cached (CSV) and fetched frames; this pins that the mixed case works as well."""
    with terminal():
        m5 = dm._fetch_bars_from_mt5_chunked("EURUSD#", "M5", 200)
        h1 = dm._fetch_bars_from_mt5_chunked("EURUSD#", "H1", 120)
    dm.append_new_bars("EURUSD#", h1, timeframe="H1")
    cached_h1 = dm.load_local_history("EURUSD#", "H1")
    out = add_contextual_features(m5, mta_df=cached_h1, mta_cfg=MtaCfg(enabled=True, timeframe="H1"))
    assert len(out) == len(m5) and isinstance(out.index, pd.DatetimeIndex)
