"""K8: a position the bot adopts at reconcile (it opened it, but the cache entry was lost) carries the money at its broker stop
as `risk`, so the portfolio cap counts it (it used to count 0). With no stop (sl 0) the nominal risk is used with a warning:
never the full price as a distance. The entry gets no `risk_amount`: the bandits must not be credited with a trade whose arms
nobody knows (atr_idx -1)."""
import pytest
from loguru import logger

from tests.test_execution_dry_run import FakeClient, OURS, _pos, make


@pytest.fixture
def warnings():
    seen = []
    handler = logger.add(lambda m: seen.append(str(m)), level="WARNING")
    yield seen
    logger.remove(handler)


def adopt(sl):
    pos = _pos(902, OURS)
    pos.sl = sl
    ex, rm = make(dry_run=False, client=FakeClient(positions=[pos]))
    ex.reconcile_open_positions_with_mt5()
    return rm, rm.open_positions_cache[902], pos


def test_an_adopted_position_carries_the_money_at_its_broker_stop():
    rm, entry, pos = adopt(sl=149.0)
    expected = pos.volume * abs(pos.price_open - pos.sl) / rm.get_pip_size(pos.symbol) * rm.get_pip_value(pos.symbol)
    assert expected > 0
    assert entry["risk"] == pytest.approx(expected)
    assert rm.total_open_risk() == pytest.approx(expected)


def test_an_adopted_position_without_a_stop_counts_the_nominal_risk_with_a_warning(warnings):
    rm, entry, _ = adopt(sl=0.0)
    nominal = 1000.0 * rm._get_dynamic_value(rm.risk_cfg.dynamic_risk, 0.5, rm.risk_cfg.risk_per_trade)  # monitor equity 1000
    assert entry["risk"] == pytest.approx(nominal) and 0 < entry["risk"] < 1000.0
    assert any("No money-at-stop" in w for w in warnings)


def test_an_adopted_entry_gets_no_risk_amount_so_the_bandits_are_not_credited():
    _, entry, _ = adopt(sl=149.0)
    assert "risk_amount" not in entry and "sl_atr_mult" not in entry
