"""K32: the contextual bandit's hour inputs come from the decision bar's own time (`context["bar_time"]`), never from the
wall clock, so a backtest (every bar the same wall-clock hour) and the live bot give the same input for the same bar.
A missing bar time gives (0, 0), which is not on the hour circle, plus one warning per symbol: never the wall clock."""
import datetime
import inspect

import numpy as np
import pandas as pd
import pytest
from loguru import logger

import backtester
from src import symbol_processor
from tests.test_rule_scale_relative import EUR, JPY, _ctx_rc, _x_for

BASE = {"vol": 0.0005, "price": 1.08}


def hour_inputs(rc, sym, monkeypatch, **ctx):
    x = _x_for(rc, sym, {**BASE, **ctx}, monkeypatch)
    return x[3], x[4]


@pytest.mark.parametrize("bar_time, hour", [
    (pd.Timestamp("2026-01-05 03:17"), 3 + 17 / 60),
    (pd.Timestamp("2026-01-05 15:42", tz="UTC"), 15 + 42 / 60),
    (datetime.datetime(2026, 1, 5, 21, 5), 21 + 5 / 60),
])
def test_the_hour_inputs_follow_the_bar_time(monkeypatch, bar_time, hour):
    rc = _ctx_rc()
    s, c = hour_inputs(rc, EUR, monkeypatch, bar_time=bar_time)
    assert s == pytest.approx(np.sin(2 * np.pi * hour / 24)) and c == pytest.approx(np.cos(2 * np.pi * hour / 24))


def test_the_same_bar_time_gives_the_same_input_on_any_symbol_and_run(monkeypatch):
    rc = _ctx_rc()
    t = pd.Timestamp("2026-03-02 09:30")
    assert hour_inputs(rc, EUR, monkeypatch, bar_time=t) == hour_inputs(rc, JPY, monkeypatch, bar_time=t)


def test_a_nat_bar_time_is_treated_as_missing(monkeypatch):
    rc = _ctx_rc()
    assert hour_inputs(rc, EUR, monkeypatch, bar_time=pd.NaT) == (0.0, 0.0)


def test_without_a_bar_time_the_inputs_are_zero_and_it_warns_once_per_symbol(monkeypatch):
    rc = _ctx_rc()
    lines = []
    sink = logger.add(lambda m: lines.append(str(m)), format="{level} {message}", level="WARNING")
    try:
        for _ in range(3):
            assert hour_inputs(rc, JPY, monkeypatch) == (0.0, 0.0)
        assert hour_inputs(rc, EUR, monkeypatch) == (0.0, 0.0)
    finally:
        logger.remove(sink)
    assert len([x for x in lines if JPY in x and "bar time" in x]) == 1
    assert len([x for x in lines if EUR in x and "bar time" in x]) == 1


def test_both_context_builders_carry_the_bar_time():
    assert '"bar_time"' in inspect.getsource(symbol_processor.SymbolProcessor._make_trade_decision)
    assert '"bar_time"' in inspect.getsource(backtester)
