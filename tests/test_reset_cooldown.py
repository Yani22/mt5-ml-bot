"""The bandit-reset cooldown runs on the decision bar's time (`context["bar_time"]`), never on the wall clock, so a backtest
(bars from years ago) and the live bot give the same cooldown for the same bars. A missing or unusable bar time skips the
reset with one warning per symbol; there is no wall-clock fallback."""
import datetime

import pandas as pd
import pytest
from loguru import logger

from src.config import Cfg
from src.risk_controller import RiskController

SYM = "EURUSD#"
T0 = pd.Timestamp("2026-01-05 10:00", tz="UTC")
LOW_AUC = 0.1  # below reset_on_low_ensemble_auc: a reset trigger on every call


def make_rc():
    cfg = Cfg()
    cfg.symbols = [SYM]
    cfg.thompson_sampling.bandit_reset_enabled = True
    cfg.thompson_sampling.reset_cooldown_hours = 24.0
    return RiskController(cfg)


def check(rc, bar_time):
    ctx = {} if bar_time is None else {"bar_time": bar_time}
    rc._check_and_trigger_reset(SYM, ctx, ensemble_auc=LOW_AUC)
    return rc.symbol_states[SYM].last_reset_time


@pytest.fixture
def warnings():
    seen = []
    handler = logger.add(lambda m: seen.append(str(m)), level="WARNING")
    yield seen
    logger.remove(handler)


def test_the_reset_is_stamped_with_the_bar_time_not_the_wall_clock():
    rc = make_rc()
    stamped = check(rc, T0)
    assert pd.Timestamp(stamped) == T0


def test_the_cooldown_holds_after_23_bar_hours_and_ends_after_25():
    rc = make_rc()
    check(rc, T0)
    assert pd.Timestamp(check(rc, T0 + pd.Timedelta(hours=23))) == T0  # still cooling down: no new reset
    assert pd.Timestamp(check(rc, T0 + pd.Timedelta(hours=25))) == T0 + pd.Timedelta(hours=25)


def test_a_missing_bar_time_resets_nothing_and_warns_once_per_symbol(warnings):
    rc = make_rc()
    assert check(rc, None) is None
    assert check(rc, None) is None
    assert check(rc, pd.NaT) is None
    assert len([w for w in warnings if "bar time" in w]) == 1


def test_a_naive_stored_reset_time_compares_with_an_aware_bar_time():
    rc = make_rc()
    rc.symbol_states[SYM].last_reset_time = datetime.datetime(2026, 1, 5, 10, 0)  # naive: an old state file
    assert check(rc, T0 + pd.Timedelta(hours=23)) == datetime.datetime(2026, 1, 5, 10, 0)
    assert pd.Timestamp(check(rc, T0 + pd.Timedelta(hours=25))) == T0 + pd.Timedelta(hours=25)


def test_a_naive_bar_time_is_read_as_utc_against_an_aware_stored_time():
    rc = make_rc()
    check(rc, T0)
    naive_23h = (T0 + pd.Timedelta(hours=23)).tz_localize(None)
    assert pd.Timestamp(check(rc, naive_23h)) == T0
