"""`retrain_time_utc` names UTC, but the scheduler compared it with `mt5c.now_utc()`, which is the broker's SERVER clock labelled UTC (about
UTC+3 on XM) whenever a tick answers and real UTC only when none does. The scheduled time then shifted by hours with the source. It now
uses real UTC (an injectable `now_fn`, the system clock by default); the terminal's clock is no longer asked."""
import datetime
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

from src import retraining
from src.retraining import _check_and_trigger_retraining

UTC = datetime.timezone.utc
REAL = datetime.datetime(2026, 1, 5, 20, 56, tzinfo=UTC)          # real UTC
SERVER = datetime.datetime(2026, 1, 5, 23, 56, tzinfo=UTC)        # the same instant on a UTC+3 server clock, labelled UTC


def trigger(monkeypatch, at, retrain_time="23:55", **kw):
    process_cls = MagicMock()
    monkeypatch.setattr(retraining, "Process", process_cls)
    cfg = NS(get_symbol_value=lambda sym, key, default=None: retrain_time)
    mt5c = NS(now_utc=lambda: SERVER)                              # the terminal says 23:56
    dates = {"EURUSD#": None}
    _check_and_trigger_retraining(cfg, "EURUSD#", {"EURUSD#": NS()}, True, MagicMock(), {}, {}, {"EURUSD#": False}, dates, MagicMock(), mt5c,
                                  now_fn=lambda: at, **kw)
    return process_cls, dates


def test_a_server_clock_past_the_time_does_not_trigger_while_real_utc_is_before_it(monkeypatch):
    process_cls, dates = trigger(monkeypatch, REAL)
    process_cls.assert_not_called()
    assert dates["EURUSD#"] is None


def test_real_utc_past_the_time_triggers(monkeypatch):
    process_cls, dates = trigger(monkeypatch, datetime.datetime(2026, 1, 5, 23, 56, tzinfo=UTC))
    process_cls.assert_called_once()
    assert dates["EURUSD#"] == datetime.date(2026, 1, 5)


def test_the_default_clock_is_the_system_clock_not_the_terminal(monkeypatch):
    process_cls = MagicMock()
    monkeypatch.setattr(retraining, "Process", process_cls)
    cfg = NS(get_symbol_value=lambda sym, key, default=None: "23:59")
    terminal = MagicMock()
    terminal.now_utc.return_value = datetime.datetime(2026, 1, 5, 23, 59, 30, tzinfo=UTC)   # would trigger if it were asked
    real_now = datetime.datetime.now(UTC)
    if real_now.hour == 23 and real_now.minute >= 59:               # the one minute a day this check cannot tell apart
        return
    _check_and_trigger_retraining(cfg, "EURUSD#", {"EURUSD#": NS()}, True, MagicMock(), {}, {}, {"EURUSD#": False},
                                  {"EURUSD#": None}, MagicMock(), terminal)
    process_cls.assert_not_called()
    terminal.now_utc.assert_not_called()
