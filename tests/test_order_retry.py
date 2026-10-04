"""B10: an order may only be re-sent when the broker guarantees it was not executed (requote, price changed, off quotes, too
many requests). Anything else (no result, an exception, timeout, a rejection) is sent once; when a position may exist it is
looked up and tracked, never sent again. A partial fill counts as filled."""
import datetime
import threading
from types import SimpleNamespace as NS

import pytest

import src.execution as execution_module
from src.config import Cfg
from src.execution import Execution
from src.risk import RiskManager

OURS = Cfg().magic_number
NOW = datetime.datetime(2026, 1, 5, 10, 0, tzinfo=datetime.timezone.utc)
SYMBOL = "EURUSD#"
DONE, PARTIAL, REQUOTE, TIMEOUT = 10009, 10010, 10004, 10012


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(execution_module.time, "sleep", lambda s: None)


class FakeClient:
    ORDER_TYPE_BUY, ORDER_TYPE_SELL, TRADE_ACTION_DEAL, ORDER_TIME_GTC = 0, 1, 1, 0
    ORDER_FILLING_IOC, TRADE_RETCODE_DONE = 1, 10009

    def __init__(self, results, positions=((),), deals=None):
        self.results, self.position_views, self.deals = list(results), list(positions), deals or {}
        self.sent, self.position_calls = [], 0

    def symbol_info_tick(self, symbol):
        return NS(ask=1.1001, bid=1.1000)

    def symbol_info(self, symbol):
        return NS(point=1e-5, trade_contract_size=100000, digits=5, trade_stops_level=0,
                  volume_min=0.01, volume_step=0.01, volume_max=100)

    def order_send(self, request):
        self.sent.append(dict(request))
        result = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if isinstance(result, Exception):
            raise result
        return result

    def positions_get(self, *a, **k):
        view = self.position_views[min(self.position_calls, len(self.position_views) - 1)]
        self.position_calls += 1
        return view

    def history_deals_get(self, *a, **k):
        return self.deals.get(k.get("ticket"), [])


class FakeNotifier:
    def __init__(self):
        self.messages = []

    def send_message(self, text, level="INFO"):
        self.messages.append(text)


def result(retcode, deal=900, volume=0.02):
    return NS(retcode=retcode, deal=deal, order=1, volume=volume)


def pos(ticket, volume=0.02, magic=OURS, type_=0):
    return NS(ticket=ticket, volume=volume, magic=magic, type=type_, symbol=SYMBOL)


def make(client):
    rm = RiskManager(Cfg(), client, threading.Lock())
    notifier = FakeNotifier()
    ex = Execution({}, {}, rm, client, NS(), dry_run=False, notifier=notifier, monitor=NS(current_equity=1000.0))
    return ex, rm, notifier


def open_long(ex, lots=0.02):
    return ex.trade(SYMBOL, "long", lots, 1.1001, 1.0995, 1.1013, 1000.0, 1e-5, 1.0, NOW, atr=0.0006, auc_score=0.6)


FILLED = {900: [NS(position_id=555)]}


# ---- a result that does not guarantee "not executed" is never sent twice ------------------------------------------

@pytest.mark.parametrize("outcome", [None, RuntimeError("terminal"), result(TIMEOUT), result(10031), result(10011),
                                     result(10006), result(10016), result(10019), result(10030), result(10018)])
def test_no_resend_unless_the_broker_guarantees_it_was_not_executed(outcome):
    client = FakeClient([outcome])
    ex, rm, _ = make(client)
    assert open_long(ex).ok is False
    assert len(client.sent) == 1
    assert rm.open_positions_cache == {}


# ---- transient rejections that guarantee "not executed" may be retried ------------------------------------------

@pytest.mark.parametrize("code", [REQUOTE, 10020, 10021, 10024])
def test_a_transient_rejection_is_retried_with_the_same_request(code):
    client = FakeClient([result(code), result(DONE)], deals=FILLED)
    ex, rm, _ = make(client)
    assert open_long(ex).ok is True
    assert len(client.sent) == 2 and client.sent[0] == client.sent[1]
    assert 555 in rm.open_positions_cache


def test_retries_are_capped_at_the_configured_number():
    client = FakeClient([result(REQUOTE)])
    ex, _, _ = make(client)
    assert open_long(ex).ok is False
    assert len(client.sent) == Cfg().trading_costs.defaults.retry_order_send


# ---- a partial fill is a fill -----------------------------------------------------------------------------------

def test_a_partial_fill_is_tracked_with_the_filled_volume_and_not_sent_again():
    client = FakeClient([result(PARTIAL, volume=0.01)], deals=FILLED)
    ex, rm, _ = make(client)
    assert open_long(ex, lots=0.02).ok is True
    assert len(client.sent) == 1
    assert rm.open_positions_cache[555]["lots"] == 0.01


# ---- an unknown result: look for the position, never resend -------------------------------------------------------

@pytest.mark.parametrize("outcome", [RuntimeError("timeout"), None, result(TIMEOUT)])
def test_an_unknown_result_with_a_new_own_position_is_tracked(outcome):
    client = FakeClient([outcome], positions=[(), (pos(777),)])
    ex, rm, _ = make(client)
    assert open_long(ex).ok is True
    assert len(client.sent) == 1
    entry = rm.open_positions_cache[777]
    assert entry["lots"] == 0.02 and entry["sl_atr_mult"] is not None and entry["risk_amount"] > 0


def test_an_existing_position_is_not_mistaken_for_the_new_one():
    client = FakeClient([RuntimeError("timeout")], positions=[(pos(500),), (pos(500),)])
    ex, rm, _ = make(client)
    assert open_long(ex).ok is False
    assert len(client.sent) == 1 and rm.open_positions_cache == {}


def test_a_new_position_from_another_system_is_not_ours():
    client = FakeClient([RuntimeError("timeout")], positions=[(), (pos(777, magic=OURS + 1),)])
    ex, rm, _ = make(client)
    assert open_long(ex).ok is False
    assert len(client.sent) == 1 and rm.open_positions_cache == {}


def test_a_position_on_the_other_side_is_not_ours():
    client = FakeClient([RuntimeError("timeout")], positions=[(), (pos(777, type_=1),)])
    ex, rm, _ = make(client)
    assert open_long(ex).ok is False


def test_on_a_netting_account_a_grown_position_counts_as_the_new_fill():
    client = FakeClient([RuntimeError("timeout")], positions=[(pos(500, volume=0.01),), (pos(500, volume=0.03),)])
    ex, rm, _ = make(client)
    assert open_long(ex).ok is True
    assert rm.open_positions_cache[500]["lots"] == pytest.approx(0.02)


def test_a_position_that_cannot_be_listed_is_not_guessed_at():
    client = FakeClient([RuntimeError("timeout")], positions=[None])
    ex, rm, _ = make(client)
    assert open_long(ex).ok is False
    assert len(client.sent) == 1


def test_an_unknown_outcome_is_reported_as_unknown():
    client = FakeClient([RuntimeError("timeout")])
    ex, _, notifier = make(client)
    open_long(ex)
    assert any("UNKNOWN" in m for m in notifier.messages)


# ---- the live cache entry must not need the feature frame (K9) ------------------------------------------------------

def test_a_live_fill_is_cached_even_without_a_feature_frame():
    client = FakeClient([result(DONE)], deals=FILLED)
    ex, rm, _ = make(client)
    assert open_long(ex).ok is True
    assert rm.open_positions_cache[555]["adx"] == 0.0


def test_a_position_reconcile_adopted_in_the_meantime_is_still_recovered_with_full_details():
    """reconcile runs in the main loop while a timed-out send blocks this thread; it adopts the new position with risk 0."""
    client = FakeClient([RuntimeError("timeout")], positions=[(), (pos(777),)])
    ex, rm, _ = make(client)
    rm.open_positions_cache[777] = {"ticket": 777, "symbol": SYMBOL, "risk": 0.0, "atr": 0.0, "sl": 0.0}
    assert open_long(ex).ok is True
    entry = rm.open_positions_cache[777]
    assert entry["atr"] == 0.0006 and entry["sl_atr_mult"] is not None


def test_an_order_placed_result_is_reported_as_unknown():
    client = FakeClient([result(10008)])
    ex, _, notifier = make(client)
    assert open_long(ex).ok is False
    assert any("UNKNOWN" in m for m in notifier.messages)
