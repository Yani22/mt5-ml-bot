"""C9: `custom_pnl` (the threshold metric when `ensemble.threshold_metric` is missing) scores the trades a signal would take
over the forward horizon, net of the shared round-trip costs, with the model's side. It used to score bar i's signal on the
move from i-1 to i (already known at i), treated a 0 as a short, ignored the short model's side and charged slippage twice."""
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from src import ensemble
from src.ensemble import custom_pnl, net_trade_returns

COSTS = dict(spread_pips=1.0, slippage_pips=0.1, adaptive_slippage=False, adaptive_slippage_multiplier=1.0)
HORIZON = 3


def idx(values):
    return pd.Series(values, index=pd.RangeIndex(len(values)), dtype=float)


def score(preds, prices, model_type="long", horizon=HORIZON, **costs):
    return custom_pnl(idx(np.zeros(len(prices))), idx(preds), idx(prices), NS(prediction_horizon=horizon),
                      model_type=model_type, **(costs or COSTS))


def test_a_move_already_known_at_the_signal_bar_earns_nothing():
    prices = [1.1000] * 5 + [1.1100] + [1.1100] * 6          # big jump into bar 5, flat afterwards
    preds = [0] * 5 + [1] + [0] * 6                           # signal on the bar of the jump
    assert score(preds, prices) == pytest.approx(-1.1 * 0.0001 / 1.1100, rel=1e-6)   # only the round-trip cost


def test_no_signal_is_no_trade_and_scores_zero():
    assert score([0] * 12, [1.1 + 0.001 * i for i in range(12)]) == 0.0


def test_the_short_models_signal_profits_when_price_falls():
    prices = [1.2 - 0.001 * i for i in range(12)]
    preds = [1] + [0] * 11
    assert score(preds, prices, model_type="short") > 0
    assert score(preds, prices, model_type="long") < 0


def test_the_cost_dict_reaches_the_score():
    prices = [1.1 + 0.0005 * i for i in range(12)]
    preds = [1] + [0] * 11
    cheap = score(preds, prices, spread_pips=0.0, slippage_pips=0.0)
    dear = score(preds, prices, spread_pips=3.0, slippage_pips=0.0)
    assert cheap - dear == pytest.approx(3.0 * 0.0001 / prices[0], rel=1e-6)


def test_it_is_the_sum_of_the_shared_net_trade_returns():
    prices = idx([1.1 + 0.0007 * (i % 5) for i in range(30)])
    preds = idx([1, 0, 0, 1, 1, 0, 0, 0, 1, 0] * 3)
    expected = net_trade_returns(preds, prices, HORIZON, "long", **COSTS).sum()
    assert score(preds.tolist(), prices.tolist()) == pytest.approx(expected)


def test_overlapping_signals_count_one_trade_per_holding_period():
    prices = [1.1 + 0.001 * i for i in range(12)]
    one = score([1] + [0] * 11, prices)
    many = score([1] * 12, prices)
    assert many == pytest.approx(one + score([0] * 3 + [1] + [0] * 8, prices) + score([0] * 6 + [1] + [0] * 5, prices), rel=0.05)


def test_the_threshold_search_passes_the_model_side_and_costs(monkeypatch):
    seen = {}

    def fake(y_true, preds, prices, cfg, model_type="long", **costs):
        seen.update(model_type=model_type, costs=costs, horizon=cfg.prediction_horizon)
        return 0.0

    monkeypatch.setattr(ensemble, "custom_pnl", fake)
    ens = object.__new__(ensemble.Ensemble)
    ens.cfg = NS(prediction_horizon=HORIZON)
    ens.threshold_metric = "custom_pnl"
    ens.threshold_grid = [0.5]
    ens.trading_costs = dict(COSTS)
    ens._optimize_threshold(idx([0, 1, 0, 1]), idx([0.2, 0.8, 0.3, 0.9]), idx([1.1, 1.2, 1.1, 1.2]), model_type="short")
    assert seen["model_type"] == "short" and seen["costs"]["spread_pips"] == 1.0 and seen["horizon"] == HORIZON
