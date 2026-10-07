"""`thompson_sampling.decay` ("forgetting factor for non-stationarity") was never passed to the bandits, so they never forgot. Now each
update shrinks every arm's statistics by `decay`. The visit count that decides whether an arm is still "exploratory" is kept apart
and never decays: otherwise a rarely chosen arm would fall back under `min_visits_for_exploration` and trade at half risk again."""
import pytest

from src.risk_controller import ThompsonBandit
from test_risk_controller import SYM, closed, make_rc


def test_update_with_decay_shrinks_the_old_statistics():
    b = ThompsonBandit(3, 0.0, 1.0)
    b.update(0, 1.0, decay=0.5)
    b.update(0, 1.0, decay=0.5)
    assert b.counts[0] == pytest.approx(1.5) and b.sum_rewards[0] == pytest.approx(1.5)
    b.update(1, 2.0, decay=0.5)                      # every arm decays, not only the one that was pulled
    assert b.counts[0] == pytest.approx(0.75) and b.counts[1] == pytest.approx(1.0)


def test_visits_never_decay():
    b = ThompsonBandit(2, 0.0, 1.0)
    for _ in range(4):
        b.update(0, 1.0, decay=0.5)
    assert b.visits[0] == 4 and b.counts[0] < 2.0


def test_without_decay_nothing_changes():
    b = ThompsonBandit(2, 0.0, 1.0)
    b.update(0, 1.0)
    b.update(0, 3.0)
    assert b.counts[0] == 2 and b.sum_rewards[0] == 4 and b.visits[0] == 2


def test_visits_survive_a_save_and_an_old_state_without_them_still_loads():
    b = ThompsonBandit(2, 0.0, 1.0)
    for _ in range(3):
        b.update(1, 1.0, decay=0.9)
    again = ThompsonBandit.from_state(b.get_state())
    assert again.visits[1] == 3
    old_state = {k: v for k, v in b.get_state().items() if k != "visits"}
    assert list(ThompsonBandit.from_state(old_state).visits) == list(b.counts)


def test_the_controller_applies_the_configured_decay_to_every_bandit():
    rc = make_rc(contextual=False)
    rc.cfg.thompson_sampling.decay = 0.5
    rc.update(closed("long", atr_idx=1, long_idx=2))
    rc.update(closed("long", atr_idx=1, long_idx=2))
    st = rc.symbol_states[SYM]
    assert st.atr_bandit.counts[1] == pytest.approx(1.5)
    assert st.min_prob_bandit_long.counts[2] == pytest.approx(1.5)


def test_an_arm_pulled_often_stays_known_even_when_its_decayed_count_is_small():
    rc = make_rc(contextual=False)
    rc.cfg.thompson_sampling.decay = 0.5
    for _ in range(6):                                # min_visits_for_exploration is 5
        rc.update(closed("long", atr_idx=1))
    st = rc.symbol_states[SYM]
    assert st.atr_bandit.counts[1] < 5 <= st.atr_bandit.visits[1]
    # force the sampler onto arm 1 and read the exploration flag
    st.atr_bandit.sample = lambda: 1
    rc.cfg.thompson_sampling.contextual_enabled = False
    params = rc.get_params(SYM, {"vol": 0.001, "price": 1.1, "bar_time": None, "equity": 1000.0, "peak_equity": 1000.0,
                                 "ensemble_auc": 0.6})
    assert params["is_exploratory"] is False and params["exploration_risk_mult"] == 1.0
