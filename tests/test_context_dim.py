"""The contextual bandit's input size is the length of the vector get_params builds, not a feature-flag formula (D)."""
import datetime
import itertools

import numpy as np
import pytest
from loguru import logger

from src.bandit_warmstart import _merge_bandit_states
from src.config import CONTEXT_VECTOR_DIM, Cfg
from src.linear_thompson import LinearThompson
from src.risk_controller import RiskController, SymbolRiskState

SYM = "USDJPY#"
NOW = datetime.datetime(2026, 1, 5, 10, 0, tzinfo=datetime.timezone.utc)
CONTEXT = {"vol": 0.1, "price": 150.0, "bar_time": NOW, "equity": 1000.0, "peak_equity": 1000.0,
           "ensemble_auc": 0.6, "adx": 25.0, "macd_diff": 0.01, "volatility_10": 0.001, "dist_from_ema_200": 0.002}


def make_cfg(mta=True, inter_market=False, price_action=True):
    cfg = Cfg()
    cfg.symbols = [SYM]
    cfg.thompson_sampling.contextual_enabled = True
    cfg.thompson_sampling.bandit_reset_enabled = False
    cfg.context_features.mta.enabled = mta
    cfg.context_features.inter_market.enabled = inter_market
    cfg.context_features.price_action.enabled = price_action
    cfg.__post_init__()
    return cfg


@pytest.mark.parametrize("mta,inter_market,price_action", list(itertools.product([True, False], repeat=3)))
def test_context_dim_does_not_depend_on_the_feature_flags(mta, inter_market, price_action):
    cfg = make_cfg(mta, inter_market, price_action)
    assert cfg.thompson_sampling.context_dim == CONTEXT_VECTOR_DIM


@pytest.mark.parametrize("mta,inter_market,price_action", list(itertools.product([True, False], repeat=3)))
def test_the_bandit_takes_every_input_get_params_builds(mta, inter_market, price_action):
    rc = RiskController(make_cfg(mta, inter_market, price_action))
    x = rc._context_vector(SYM, CONTEXT)
    assert len(x) == CONTEXT_VECTOR_DIM
    assert rc.symbol_states[SYM].contextual_bandit.dim == len(x)
    assert len(rc.get_params(SYM, CONTEXT)["context_vector"]) == len(x)


def test_the_last_input_is_the_distance_from_the_200_ema_and_is_kept():
    rc = RiskController(make_cfg())
    params = rc.get_params(SYM, CONTEXT)
    assert params["context_vector"][-1] == pytest.approx(CONTEXT["dist_from_ema_200"] * 100.0)


# ---- a saved state with another input size is not loaded into a bandit that cannot learn from it ----------------

def old_state(cfg, dim):
    state = SymbolRiskState(cfg, [1.0, 1.5], [0.55, 0.6], [0.55, 0.6]).get_state()
    bandit = LinearThompson(num_arms=2, dim=dim)
    bandit.update(0, np.ones(dim), 1.0)
    state["contextual_bandit"] = bandit.get_state()
    return state


def test_a_saved_bandit_of_another_size_is_replaced_by_a_fresh_one_with_a_warning():
    cfg = make_cfg()
    messages = []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        loaded = SymbolRiskState.from_state(cfg, old_state(cfg, CONTEXT_VECTOR_DIM - 1))
    finally:
        logger.remove(sink)
    bandit = loaded.contextual_bandit
    assert bandit.dim == CONTEXT_VECTOR_DIM
    assert float(np.abs(np.array(bandit.b)).sum()) == 0.0
    assert any("contextual bandit" in m.lower() for m in messages)


def test_a_saved_bandit_of_the_right_size_is_kept():
    cfg = make_cfg()
    loaded = SymbolRiskState.from_state(cfg, old_state(cfg, CONTEXT_VECTOR_DIM))
    assert loaded.contextual_bandit.dim == CONTEXT_VECTOR_DIM
    assert float(np.abs(np.array(loaded.contextual_bandit.b)).sum()) > 0.0


def test_warm_start_does_not_merge_contextual_bandits_of_different_sizes():
    cfg = make_cfg()
    live = old_state(cfg, CONTEXT_VECTOR_DIM - 1)
    backtest = old_state(cfg, CONTEXT_VECTOR_DIM)
    merged = _merge_bandit_states(live, backtest, 1.0)
    assert merged["contextual_bandit"]["dim"] == CONTEXT_VECTOR_DIM - 1
    assert np.array(merged["contextual_bandit"]["A"]).shape[-1] == CONTEXT_VECTOR_DIM - 1


def test_warm_start_still_merges_contextual_bandits_of_the_same_size():
    cfg = make_cfg()
    live = old_state(cfg, CONTEXT_VECTOR_DIM)
    backtest = old_state(cfg, CONTEXT_VECTOR_DIM)
    merged = _merge_bandit_states(live, backtest, 1.0)
    b_live = np.array(live["contextual_bandit"]["b"])
    assert np.array(merged["contextual_bandit"]["b"]).sum() == pytest.approx(2 * b_live.sum())
