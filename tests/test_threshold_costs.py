"""C7: the threshold search scores a signal by what its trades would earn NET of costs, in return space (cost in pips x pip
size / entry price, one round trip, spread plus slippage), counting each trade once (greedy non-overlapping, like one
position per symbol), and annualised with the trades per year, not the bars per year. The best and promising thresholds it
produces are only stored in the model metadata; nothing reads them to trade."""
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest
from loguru import logger

from src import ensemble
from src.costs import round_trip_pips
from src.ensemble import calculate_sharpe_ratio, infer_pip_size, net_trade_returns

COSTS = dict(spread_pips=1.0, slippage_pips=0.1, adaptive_slippage=False, adaptive_slippage_multiplier=1.0)


def series(start, step, n=12):
    return pd.Series([start + step * i for i in range(n)], index=pd.RangeIndex(n))


def ones(n=12):
    return pd.Series([1] * n, index=pd.RangeIndex(n))


@pytest.fixture
def log_lines():
    lines = []
    sink = logger.add(lambda m: lines.append(str(m)), format="{level} {message}", level="DEBUG")
    yield lines
    logger.remove(sink)


# ---- the shared round-trip cost ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("adaptive, mult, expected", [(False, 2.0, 1.1), (True, 2.0, 1.2), (True, 1.0, 1.1)])
def test_round_trip_pips_is_one_spread_plus_slippage_scaled_only_when_adaptive(adaptive, mult, expected):
    assert round_trip_pips(1.0, 0.1, adaptive, mult) == pytest.approx(expected)


# ---- the cost is in return space -------------------------------------------------------------------------------------------------

def test_a_five_digit_pair_pays_its_cost_as_a_fraction_of_the_entry_price():
    prices = series(1.2000, 0.0005)
    r = net_trade_returns(ones(), prices, 2, "long", **COSTS)
    # gross (p2 - p0) / p0 = 0.0010 / 1.2000; cost 1.1 pips x 0.0001 / 1.2000
    assert r.iloc[0] == pytest.approx((0.0010 - 0.00011) / 1.2000)


def test_a_jpy_pair_uses_its_own_pip_size():
    prices = series(150.00, 0.05)
    r = net_trade_returns(ones(), prices, 2, "long", **COSTS)
    assert r.iloc[0] == pytest.approx((0.10 - 1.1 * 0.01) / 150.00)


def test_an_explicit_pip_size_wins_over_the_inferred_one():
    r = net_trade_returns(ones(), series(150.00, 0.05), 2, "long", pip_size=0.001, **COSTS)
    assert r.iloc[0] == pytest.approx((0.10 - 1.1 * 0.001) / 150.00)


def test_the_old_pip_value_key_is_accepted_as_a_pip_size():
    r = net_trade_returns(ones(), series(150.00, 0.05), 2, "long", pip_value=0.001, **COSTS)
    assert r.iloc[0] == pytest.approx((0.10 - 1.1 * 0.001) / 150.00)


def test_slippage_is_charged_and_the_adaptive_multiplier_applies():
    plain = net_trade_returns(ones(), series(1.1000, 0.0005), 2, "long", **COSTS).iloc[0]
    adaptive = net_trade_returns(ones(), series(1.1000, 0.0005), 2, "long", **dict(COSTS, adaptive_slippage=True, adaptive_slippage_multiplier=2.0)).iloc[0]
    assert plain - adaptive == pytest.approx(0.1 * 0.0001 / 1.1000)       # slippage doubled: 0.1 pip more


def test_a_short_mirrors_a_long():
    prices = series(1.2000, -0.0005)
    r = net_trade_returns(ones(), prices, 2, "short", **COSTS)
    assert r.iloc[0] == pytest.approx((0.0010 - 0.00011) / 1.2000)


# ---- each trade is counted once ----------------------------------------------------------------------------------------------------

def test_overlapping_signals_count_once_per_holding_period():
    r = net_trade_returns(ones(12), series(1.1000, 0.0005, 12), 3, "long", **COSTS)
    assert len(r) == 3                       # bars 0, 3, 6 (bars 9 to 11 have no known exit); the old code scored 9 overlapping trades


def test_a_gap_between_signals_is_respected():       # guard: passes on the old code too
    signal = pd.Series([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], index=pd.RangeIndex(12))
    r = net_trade_returns(signal, series(1.1000, 0.0005, 12), 3, "long", **COSTS)
    assert list(r.index) == [0, 5]


# ---- the Sharpe ratio -------------------------------------------------------------------------------------------------------------------

def test_the_sharpe_ratio_is_annualised_with_the_trades_per_year_not_the_bars_per_year():
    cfg = NS(prediction_horizon=2, timeframe_minutes=lambda: 5)
    prices = pd.Series(1.1 + np.cumsum([0.0004, -0.0002, 0.0006, 0.0001, -0.0003, 0.0005, 0.0002, -0.0001, 0.0004, 0.0003, -0.0002, 0.0001]), index=pd.RangeIndex(12))
    r = net_trade_returns(ones(), prices, 2, "long", **COSTS)
    expected = r.mean() / r.std() * np.sqrt(252 * (24 * 60 / 5) / 2)
    assert calculate_sharpe_ratio(ones(), ones(), prices, cfg, "long", **COSTS) == pytest.approx(expected)


def test_no_signals_scores_zero():
    cfg = NS(prediction_horizon=2, timeframe_minutes=lambda: 5)
    nothing = pd.Series([0] * 12, index=pd.RangeIndex(12))
    assert calculate_sharpe_ratio(nothing, nothing, series(1.1, 0.0005), cfg, "long", **COSTS) == 0.0


# ---- the pip size guess --------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("level, pip", [(1.10, 0.0001), (0.66, 0.0001), (150.0, 0.01), (95.0, 0.01)])
def test_the_pip_size_is_inferred_from_the_price_level(level, pip):
    assert infer_pip_size(series(level, 0.0)) == pip


def test_prices_that_do_not_look_like_forex_get_one_warning(log_lines, monkeypatch):
    monkeypatch.setattr(ensemble, "_warned_non_forex_pips", False)
    assert infer_pip_size(series(2300.0, 0.5)) == 0.0001
    assert infer_pip_size(series(2300.0, 0.5)) == 0.0001
    warnings = [line for line in log_lines if line.startswith("WARNING") and "pip" in line]
    assert len(warnings) == 1
