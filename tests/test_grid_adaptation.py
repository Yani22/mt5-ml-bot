"""Adaptive grids (off by default) had three defects and the reset/warm-start code two more: the contextual branch indexed lists of
arrays like a matrix (TypeError out of every closed trade), the short threshold grid never adapted when the long one did (one shared
counter, reset in between), the "best arm" could be one never pulled, the reset used the global grids instead of the symbol's, and the
warm-start merge paired arms by index across different grids and counted the prior of each `A` matrix twice."""
import json

import numpy as np
import pytest

from src.bandit_warmstart import merge_warmstart
from src.risk_controller import RiskController, ThompsonBandit
from test_risk_controller import SYM, closed, make_rc


def adaptive_rc(contextual):
    rc = make_rc(contextual=contextual)
    ts = rc.cfg.thompson_sampling
    ts.adaptive_grids_enabled = True
    ts.adaptation_interval_updates = 1
    ts.decay = 1.0
    # grids of the minimum size (5): a smaller grid is never refined
    rc.cfg.symbol_overrides = {SYM: {"atr_grid": [1.0, 1.5, 2.0, 2.5, 3.0], "min_prob_grid_long": [0.51, 0.53, 0.55, 0.57, 0.60],
                                     "min_prob_grid_short": [0.51, 0.53, 0.55, 0.57, 0.60]}}
    return RiskController(rc.cfg)


def test_the_contextual_branch_no_longer_crashes_and_the_closed_trade_is_counted():
    rc = adaptive_rc(contextual=True)
    st = rc.symbol_states[SYM]
    dim = st.contextual_bandit.dim
    rc.update(closed("long", pnl=10.0, atr_idx=2, dim=dim))          # raised TypeError out of update() before
    assert st.atr_bandit.counts.sum() == 1            # the trade was learned (the grid may have been refined around it)


def test_the_short_grid_adapts_when_the_long_grid_does():
    rc = adaptive_rc(contextual=False)
    st = rc.symbol_states[SYM]
    st.min_prob_bandit_long.update(2, -0.5)
    st.min_prob_bandit_short.update(1, -0.5)
    st.min_prob_updates_since_last_adaptation = 5      # both sides are judged on the same call
    long_before, short_before = list(st.min_prob_grid_long_values), list(st.min_prob_grid_short_values)
    rc._check_and_trigger_adaptation(SYM)
    assert st.min_prob_grid_long_values != long_before and st.min_prob_grid_short_values != short_before


def test_the_best_arm_is_a_visited_one():
    b = ThompsonBandit(5, 0.0, 1.0)
    b.update(3, -0.4)                                   # the only arm ever pulled lost; unvisited arms must not look like 0.0 > -0.4
    assert RiskController._best_visited_arm(b) == 3
    assert RiskController._best_visited_arm(ThompsonBandit(5, 0.0, 1.0)) is None


def test_no_adaptation_when_nothing_was_pulled():
    rc = adaptive_rc(contextual=False)
    st = rc.symbol_states[SYM]
    st.atr_updates_since_last_adaptation = 5
    grid = list(st.atr_grid_values)
    rc._check_and_trigger_adaptation(SYM)
    assert st.atr_grid_values == grid and st.atr_updates_since_last_adaptation == 0


def test_a_reset_rebuilds_the_symbols_own_grids():
    rc = make_rc(contextual=True)
    own = [0.7, 1.1, 1.6]
    rc.cfg.symbol_overrides = {SYM: {"atr_grid": own}}
    rc = RiskController(rc.cfg)
    from datetime import datetime, timezone
    rc._reset_bandit_state(SYM, datetime(2026, 1, 5, tzinfo=timezone.utc))
    st = rc.symbol_states[SYM]
    assert st.atr_grid_values == own and st.atr_bandit.num_arms == 3 and st.contextual_bandit.num_arms == 3


# ---- warm-start ----
def state_file(path, grid, trades=0, contextual=False):
    rc = make_rc(contextual=contextual)
    rc.cfg.thompson_sampling.decay = 1.0
    rc.cfg.symbol_overrides = {SYM: {"atr_grid": grid}}
    rc = RiskController(rc.cfg)
    for _ in range(trades):
        rc.update(closed("long", atr_idx=1, dim=rc.symbol_states[SYM].contextual_bandit.dim if contextual else None))
    rc.save_state({}, path=str(path))
    return rc


def test_a_backtest_with_other_grids_is_not_merged(tmp_path):
    live, back = tmp_path / "live.json", tmp_path / "ts_risk_controller_state_backtest_x.json"
    state_file(live, [1.0, 1.5, 2.0, 2.5], trades=1)
    state_file(back, [0.5, 1.0, 1.5, 2.0], trades=4)
    merge_warmstart(str(back), str(live), warmstart_weight=1.0)
    counts = json.loads(live.read_text())["symbol_states"][SYM]["atr_bandit"]["counts"]
    assert counts[1] == 1                                # the backtest's arm 1 is another multiple: not added


def test_the_prior_of_a_merged_A_matrix_is_counted_once(tmp_path):
    live, back = tmp_path / "live.json", tmp_path / "ts_risk_controller_state_backtest_x.json"
    grid = [1.0, 1.5, 2.0, 2.5]
    state_file(live, grid, trades=0, contextual=True)
    state_file(back, grid, trades=3, contextual=True)
    merge_warmstart(str(back), str(live), warmstart_weight=1.0)
    arm1 = np.array(json.loads(live.read_text())["symbol_states"][SYM]["contextual_bandit"]["A"][1])
    back_arm1 = np.array(json.loads(back.read_text())["symbol_states"][SYM]["contextual_bandit"]["A"][1])
    lam = 1.0
    assert np.allclose(arm1, back_arm1)                  # lambda*I (live) + (back - lambda*I), not 2*lambda*I + data
    assert np.allclose(np.diag(arm1)[0], back_arm1[0, 0]) and lam == pytest.approx(1.0)


# ---- the contextual bandit's statistics through a grid change ----
def test_a_transferred_contextual_arm_is_lambda_i_plus_the_old_data_with_a_consistent_inverse():
    from src.linear_thompson import LinearThompson
    old = LinearThompson(num_arms=3, dim=2, lambda_prior=1.0)
    x = np.array([1.0, 2.0])
    old.update(0, x, 0.5)
    old.update(1, x, -0.25)
    new = LinearThompson(num_arms=2, dim=2, lambda_prior=1.0)
    RiskController._transfer_contextual_bandit_state(old, [1.0, 1.5, 2.0], new, [1.0, 2.0])
    # old arms 0 and 1 (values 1.0 and 1.5) are both closest to new arm 0 only when 1.5 rounds that way; check the sums, not the mapping
    total_data = (old.A[0] - np.eye(2)) + (old.A[1] - np.eye(2)) + (old.A[2] - np.eye(2))
    assert np.allclose(sum(a - np.eye(2) for a in new.A), total_data)
    assert np.allclose(sum(new.b), old.b[0] + old.b[1] + old.b[2])
    for arm in range(2):
        assert np.allclose(new.invA[arm], np.linalg.inv(new.A[arm]))       # the sampler reads invA, not A
