"""The low-AUC reset trigger needs a real AUC. A context with no `ensemble_auc` (or a NaN) used to read as 0.5, below
`reset_on_low_ensemble_auc`, and wiped the symbol's bandits every cooldown."""
import pandas as pd

from src.config import Cfg
from src.risk_controller import RiskController

SYM = "EURUSD#"
T0 = pd.Timestamp("2026-01-05 10:00", tz="UTC")


def make_rc():
    cfg = Cfg()
    cfg.symbols = [SYM]
    cfg.thompson_sampling.bandit_reset_enabled = True
    cfg.thompson_sampling.reset_on_low_ensemble_auc = 0.52
    cfg.thompson_sampling.reset_on_consecutive_losses = 10 ** 6
    cfg.thompson_sampling.reset_on_drawdown_percent = 2.0
    return RiskController(cfg)


def reset_time(ctx):
    rc = make_rc()
    rc.get_params(SYM, {"vol": 0.001, "price": 1.0, "bar_time": T0, **ctx})
    return rc.symbol_states[SYM].last_reset_time


def test_a_context_without_an_auc_does_not_reset():
    assert reset_time({}) is None


def test_a_nan_auc_does_not_reset():
    assert reset_time({"ensemble_auc": float("nan")}) is None


def test_a_real_low_auc_still_resets():
    assert pd.Timestamp(reset_time({"ensemble_auc": 0.50})) == T0


def test_a_good_auc_does_not_reset():
    assert reset_time({"ensemble_auc": 0.56}) is None


# ---- the low-AUC trigger counts once per model, not once per cooldown ------------------------------------------------

H = pd.Timedelta(hours=1)


def resets_over(steps):
    """Run get_params at (hours after T0, AUC) steps; return the bar times at which a reset happened."""
    rc = make_rc()
    rc.cfg.thompson_sampling.reset_cooldown_hours = 24.0
    seen = []
    for hours, auc in steps:
        before = rc.symbol_states[SYM].last_reset_time
        rc.get_params(SYM, {"vol": 0.001, "price": 1.0, "bar_time": T0 + hours * H, "ensemble_auc": auc})
        after = rc.symbol_states[SYM].last_reset_time
        if after != before:
            seen.append(hours)
    return seen


def test_the_same_low_auc_resets_once_not_every_cooldown():
    assert resets_over([(0, 0.50), (25, 0.50), (50, 0.50), (75, 0.50)]) == [0]


def test_a_new_low_auc_resets_again():
    assert resets_over([(0, 0.50), (25, 0.50), (50, 0.49)]) == [0, 50]


def test_a_new_low_auc_that_arrives_in_the_cooldown_resets_when_it_ends():
    assert resets_over([(0, 0.50), (10, 0.49), (20, 0.49), (25, 0.49)]) == [0, 25]


def test_a_good_auc_never_resets_and_a_low_one_after_it_does():
    assert resets_over([(0, 0.56), (25, 0.56), (50, 0.50)]) == [50]


def test_the_checked_auc_survives_a_save_and_load(tmp_path):
    rc = make_rc()
    rc.cfg.thompson_sampling.state_file = str(tmp_path / "ts.json")
    rc.get_params(SYM, {"vol": 0.001, "price": 1.0, "bar_time": T0, "ensemble_auc": 0.5})
    rc.save_state()
    fresh = make_rc()
    fresh.cfg.thompson_sampling.state_file = str(tmp_path / "ts.json")
    fresh.load_state()
    assert fresh.symbol_states[SYM].last_auc_checked == 0.5
    fresh.get_params(SYM, {"vol": 0.001, "price": 1.0, "bar_time": T0 + 30 * H, "ensemble_auc": 0.5})
    assert pd.Timestamp(fresh.symbol_states[SYM].last_reset_time) == T0   # the restart did not reset it again


def test_a_state_saved_before_the_field_existed_counts_the_first_check_as_a_new_model(tmp_path):
    rc = make_rc()
    state = rc.symbol_states[SYM].get_state()
    state.pop("last_auc_checked")
    from src.risk_controller import SymbolRiskState
    assert SymbolRiskState.from_state(rc.cfg, state).last_auc_checked is None
