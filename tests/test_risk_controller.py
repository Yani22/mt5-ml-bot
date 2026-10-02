"""Bandit credit: a closed trade may only teach the bandits what it actually tested (T1, T2)."""
import datetime

import numpy as np
import pytest

from src.config import Cfg
from src.risk_controller import RiskController
from src.trade_types import ClosedTrade

SYM = "EURUSD#"
NOW = datetime.datetime(2026, 1, 5, 10, 0, tzinfo=datetime.timezone.utc)


def make_rc(contextual=True, symbols=(SYM,)):
    cfg = Cfg()
    cfg.symbols = list(symbols)
    cfg.thompson_sampling.contextual_enabled = contextual
    cfg.thompson_sampling.bandit_reset_enabled = False
    cfg.thompson_sampling.min_visits_for_exploration = 5
    cfg.thompson_sampling.exploration_risk_mult = 0.5
    return RiskController(cfg)


def closed(direction="long", pnl=10.0, risk_amount=10.0, atr_idx=1, long_idx=2, short_idx=1, symbol=SYM, dim=None, **kw):
    ctx = None if dim is None else [0.1] * dim
    return ClosedTrade(ticket=1, symbol=symbol, direction=direction, lots=0.01, entry_price=1.1, exit_price=1.101,
                       entry_time=NOW, exit_time=NOW, pnl=pnl, risk_fraction=0.01, atr=0.001, atr_idx=atr_idx,
                       min_prob_long_idx=long_idx, min_prob_short_idx=short_idx, entry_auc=0.6, entry_equity=1000.0,
                       exit_equity=1010.0, context_vector=ctx, risk_amount=risk_amount, **kw)


def ctx_dim(rc):
    return rc.symbol_states[SYM].contextual_bandit.dim


# ---- T1: only the side that traded is credited ----------------------------------------------------------------

def test_a_long_trade_updates_only_the_long_threshold_bandit():
    rc = make_rc(contextual=False)
    rc.update(closed("long"))
    st = rc.symbol_states[SYM]
    assert st.min_prob_bandit_long.counts[2] == 1
    assert st.min_prob_bandit_short.counts.sum() == 0


def test_a_short_trade_updates_only_the_short_threshold_bandit():
    rc = make_rc(contextual=False)
    rc.update(closed("short"))
    st = rc.symbol_states[SYM]
    assert st.min_prob_bandit_short.counts[1] == 1
    assert st.min_prob_bandit_long.counts.sum() == 0


def test_an_unknown_direction_updates_neither_threshold_bandit():
    rc = make_rc(contextual=False)
    rc.update(closed(None))
    st = rc.symbol_states[SYM]
    assert st.min_prob_bandit_long.counts.sum() == 0 and st.min_prob_bandit_short.counts.sum() == 0


def test_the_threshold_adaptation_counter_goes_up_once_per_trade():
    rc = make_rc(contextual=False)
    rc.update(closed("long"))
    assert rc.symbol_states[SYM].min_prob_updates_since_last_adaptation == 1


# ---- T2: the contextual bandit's arm choice is counted --------------------------------------------------------

def test_contextual_closes_count_a_visit_on_the_arm_used():
    rc = make_rc(contextual=True)
    rc.update(closed(atr_idx=2, dim=ctx_dim(rc)))
    assert rc.symbol_states[SYM].atr_bandit.counts[2] == 1


def test_an_arm_stops_being_exploratory_after_enough_contextual_visits(monkeypatch):
    rc = make_rc(contextual=True)
    st = rc.symbol_states[SYM]
    monkeypatch.setattr(st.contextual_bandit, "sample_arm", lambda x: 2)
    ctx = {"ensemble_auc": 0.6}
    assert rc.get_params(SYM, ctx)["is_exploratory"] is True
    for _ in range(5):
        rc.update(closed(atr_idx=2, dim=ctx_dim(rc)))
    params = rc.get_params(SYM, ctx)
    assert params["is_exploratory"] is False and params["exploration_risk_mult"] == 1.0


# ---- T5: the reward is profit in units of the money risked (R), not money --------------------------------------

def test_the_reward_is_pnl_divided_by_the_money_risked():
    rc = make_rc(contextual=False)
    rc.update(closed("long", pnl=-12.0, risk_amount=12.0, long_idx=2))
    st = rc.symbol_states[SYM]
    assert st.min_prob_bandit_long.sum_rewards[2] == pytest.approx(-1.0)
    assert st.atr_bandit.sum_rewards[1] == pytest.approx(-1.0)


def test_the_reward_does_not_depend_on_account_size():
    small, big = make_rc(contextual=False), make_rc(contextual=False)
    small.update(closed("long", pnl=0.6, risk_amount=0.5))
    big.update(closed("long", pnl=600.0, risk_amount=500.0))
    assert small.symbol_states[SYM].atr_bandit.sum_rewards[1] == pytest.approx(
        big.symbol_states[SYM].atr_bandit.sum_rewards[1])


@pytest.mark.parametrize("risk_amount", [None, 0.0, -1.0])
def test_a_trade_without_a_usable_risk_amount_teaches_no_bandit(risk_amount):
    rc = make_rc(contextual=False)
    rc.update(closed("long", pnl=5.0, risk_amount=risk_amount))
    st = rc.symbol_states[SYM]
    assert st.atr_bandit.counts.sum() == 0 and st.min_prob_bandit_long.counts.sum() == 0
    assert len(st.recent_returns) == 0


def test_a_trade_without_a_risk_amount_still_updates_equity_and_loss_streak():
    rc = make_rc(contextual=False)
    rc.update(closed("long", pnl=-5.0, risk_amount=None))
    st = rc.symbol_states[SYM]
    assert st.consecutive_losses == 1 and st.current_equity == 1010.0


@pytest.mark.parametrize("risk_amount", [10.0, None])
def test_win_streaks_still_update_with_or_without_a_risk_amount(risk_amount):
    rc = make_rc(contextual=False)
    rc.cfg.asymmetric_compounding.enabled = True
    rc.update(closed("long", pnl=5.0, risk_amount=risk_amount))
    assert rc.symbol_states[SYM].win_streak == 1
