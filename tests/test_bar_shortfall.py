"""K13: the terminal only returns bars within its "Max. bars in chart" history, so a bootstrap can silently end up with far
fewer bars than `history_bars`. The fetch says so when it comes back short, and a bootstrap that ends below what it needs
(scaled to the timeframe: the H1 context needs far fewer bars than the M5 window) warns once, with the fix."""
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
from loguru import logger

from src.data_manager import DataManager

STEP = {"M5": 300, "H1": 3600}


@pytest.fixture
def log_lines():
    lines = []
    sink = logger.add(lambda m: lines.append(str(m)), format="{level} {message}", level="DEBUG")
    yield lines
    logger.remove(sink)


@pytest.fixture
def dm(tmp_path):
    d = object.__new__(DataManager)
    d.cfg, d.raw_data_dir = NS(timeframe="M5"), str(tmp_path)
    return d


def bars(n, step):
    return [{"time": 1_700_000_000 + i * step, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "tick_volume": 1}
            for i in range(n)]


def terminal(cap, error=None):
    """A terminal whose chart history holds at most `cap` bars (None: the call fails)."""
    mt5 = sys.modules["MetaTrader5"]

    def copy_rates_from_pos(symbol, tf, start, count):
        if cap is None:
            return None
        step = 3600 if tf == mt5.TIMEFRAME_H1 else 300
        return bars(min(count, cap), step)

    return patch.multiple(mt5, copy_rates_from_pos=copy_rates_from_pos, last_error=lambda: error or (1, "ok"), create=True)


def warnings_of(lines, text="below"):
    return [line for line in lines if line.startswith("WARNING") and text in line]


# ---- the fetch ---------------------------------------------------------------------------------------------------------

def test_a_short_fetch_says_how_many_bars_came_back_and_how_many_were_asked_for(dm, log_lines):
    with terminal(cap=5):
        dm._fetch_bars_from_mt5_chunked("EURUSD#", "M5", 200)
    assert any("returned 5" in line and "200" in line and "Max bars in chart" in line for line in log_lines)


def test_a_full_fetch_logs_nothing_about_a_shortfall(dm, log_lines):
    with terminal(cap=10_000):
        df = dm._fetch_bars_from_mt5_chunked("EURUSD#", "M5", 200)
    assert len(df) == 199                       # the forming bar is dropped: this must not look like a shortfall
    assert not any("asked for" in line for line in log_lines)


def test_a_failed_fetch_logs_the_terminal_error(dm, log_lines):
    with terminal(cap=None, error=(-2, "Terminal: Invalid params")):
        assert dm._fetch_bars_from_mt5_chunked("EURUSD#", "M5", 200).empty
    assert any("Invalid params" in line for line in log_lines)


# ---- the bootstrap -----------------------------------------------------------------------------------------------------

def test_a_bootstrap_that_ends_short_warns_once_with_have_and_need(dm, log_lines):
    with terminal(cap=5):
        dm.bootstrap_history("EURUSD#", initial_bars=50)
    (line,) = warnings_of(log_lines)
    assert "4 bars" in line and "50" in line and "Max bars in chart" in line


def test_a_bootstrap_that_reaches_its_need_does_not_warn(dm, log_lines):
    with terminal(cap=10_000):
        dm.bootstrap_history("EURUSD#", initial_bars=50)
    assert warnings_of(log_lines) == []


def test_the_context_timeframe_is_judged_against_the_bars_that_span_the_same_window(dm, log_lines):
    # 45,000 M5 bars span 3,750 H1 bars
    with terminal(cap=100):
        dm.bootstrap_history("EURUSD#", initial_bars=45_000, timeframe="H1")
    (line,) = warnings_of(log_lines)
    assert "3750" in line


def test_a_capped_terminal_does_not_make_the_context_timeframe_warn_when_it_has_enough(dm, log_lines):
    with terminal(cap=5_000):                   # the real cap on this machine: 5,000 H1 bars cover 45,000 M5 bars
        dm.bootstrap_history("EURUSD#", initial_bars=45_000, timeframe="H1")
    assert warnings_of(log_lines) == []
