"""The open-position cache feeds the portfolio cap. Its `risk` must be the money at the stop that was placed, at the lots
that were sent: not `equity * the global risk_per_trade`, which ignores per-symbol overrides, the streak multipliers and
the lot flooring."""
import math

import pytest

from tests import test_execution_dry_run as dry
from tests import test_order_retry as live

PRICE, SL, TP, PIP_SIZE, EQUITY = 1.1001, 1.0995, 1.1013, 1e-5, 1000.0


@pytest.fixture(autouse=True)
def _tmp_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(live.execution_module.time, "sleep", lambda s: None)


def expected(lots, pip_value):
    return lots * abs(PRICE - SL) / PIP_SIZE * pip_value


def open_dry(lots, pip_value):
    ex, rm = dry.make(dry_run=True)
    ex.trade("EURUSD#", "long", lots, PRICE, SL, TP, EQUITY, PIP_SIZE, pip_value, dry.NOW, atr=0.0006, auc_score=0.6)
    ((_, entry),) = rm.open_positions_cache.items()
    return entry


def open_live(lots, pip_value):
    client = live.FakeClient([live.result(live.DONE, volume=lots)], deals=live.FILLED)
    ex, rm, _ = live.make(client)
    ex.trade(live.SYMBOL, "long", lots, PRICE, SL, TP, EQUITY, PIP_SIZE, pip_value, live.NOW, atr=0.0006, auc_score=0.6)
    return rm.open_positions_cache[555]


@pytest.mark.parametrize("open_", [open_dry, open_live])
@pytest.mark.parametrize("lots", [0.02, 0.07])        # the old nominal is 7.5 (dynamic risk at AUC 0.6) whatever the lots
def test_the_cache_risk_is_the_money_at_the_placed_stop(open_, lots):
    entry = open_(lots, 1.0)
    assert entry["risk"] == pytest.approx(expected(lots, 1.0))
    assert entry["risk"] == pytest.approx(entry["risk_amount"])


@pytest.mark.parametrize("open_", [open_dry, open_live])
def test_without_a_pip_value_the_cache_risk_stays_a_finite_number(open_):
    entry = open_(0.02, 0.0)                           # risk_amount is None here: fall back to the nominal risk
    assert isinstance(entry["risk"], float) and math.isfinite(entry["risk"]) and entry["risk"] > 0
