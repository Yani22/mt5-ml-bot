"""Reconcile must not mistake "the broker could not be asked" or "the closing deal has not posted yet" for a closed trade, must
learn from the fill (not the requested price), and must stop managing a position that is waiting to be closed."""
import datetime
import threading
from types import SimpleNamespace as NS

import MetaTrader5 as mt5
import pytest

import src.execution as execution_module
from src.config import Cfg
from src.execution import Execution
from src.risk import RiskManager

OURS = Cfg().magic_number
NOW = datetime.datetime(2026, 1, 5, 10, 0, tzinfo=datetime.timezone.utc)
SYMBOL = "EURUSD#"
OPENING = mt5.DEAL_ENTRY_OUT + 1       # under the conftest stub DEAL_ENTRY_OUT is 0, which is real MT5's DEAL_ENTRY_IN
OPENED, CLOSED = 1767607200, 1767610800


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)


class FakeClient:
    ORDER_TYPE_BUY, ORDER_TYPE_SELL, TRADE_ACTION_DEAL, ORDER_TIME_GTC = 0, 1, 1, 0
    ORDER_FILLING_IOC, TRADE_RETCODE_DONE = 1, 10009

    def __init__(self, positions=(), deals=None, equity=None):
        self.positions, self.deals, self.equity = positions, deals or {}, equity

    def symbol_info_tick(self, symbol):
        return NS(ask=1.1001, bid=1.1000)

    def symbol_info(self, symbol):
        return NS(point=1e-5, trade_contract_size=100000, digits=5, trade_stops_level=0,
                  volume_min=0.01, volume_step=0.01, volume_max=100)

    def positions_get(self, *a, **k):
        return self.positions

    def history_deals_get(self, *a, **k):
        return self.deals.get(k.get("position"), [])

    def account_info(self):
        return None if self.equity is None else NS(equity=self.equity)


class FakeNotifier:
    def __init__(self):
        self.messages = []

    def send_message(self, text, level="INFO"):
        self.messages.append(text)


ENTRY_DEAL = NS(entry=OPENING, profit=0.0, commission=-0.5, swap=0.0, fee=0.0, time=OPENED, price=1.1001, magic=OURS)
EXIT_DEAL = NS(entry=mt5.DEAL_ENTRY_OUT, profit=3.5, commission=0.0, swap=0.0, fee=0.0, time=CLOSED, price=1.1013, magic=OURS)
CACHE_ENTRY = {"symbol": SYMBOL, "direction": "long", "lots": 0.02, "entry_price": 1.1001, "entry_time": NOW,
               "risk": 20.0, "risk_amount": 20.0, "atr": 0.0006, "atr_idx": 2, "sl": 1.0995, "tp": 1.1013}


def make(client):
    rm = RiskManager(Cfg(), client, threading.Lock())
    notifier = FakeNotifier()
    ex = Execution({}, {}, rm, client, NS(), dry_run=False, notifier=notifier, monitor=NS(current_equity=1000.0))
    rm.open_positions_cache[777] = dict(CACHE_ENTRY)
    return ex, rm, notifier


def test_a_failed_positions_read_leaves_every_tracked_position_alone():
    ex, rm, _ = make(FakeClient(positions=None, deals={777: [ENTRY_DEAL]}))
    assert ex.reconcile_open_positions_with_mt5() == []
    assert 777 in rm.open_positions_cache and "close_first_seen" not in rm.open_positions_cache[777]


def test_a_position_whose_closing_deal_has_not_posted_is_kept_and_waits():
    ex, rm, _ = make(FakeClient(positions=[], deals={777: [ENTRY_DEAL]}))
    assert ex.reconcile_open_positions_with_mt5() == []
    assert 777 in rm.open_positions_cache and isinstance(rm.open_positions_cache[777]["close_first_seen"], float)


def test_it_closes_when_the_closing_deal_posts_later():
    client = FakeClient(positions=[], deals={777: [ENTRY_DEAL]})
    ex, rm, _ = make(client)
    assert ex.reconcile_open_positions_with_mt5() == []
    client.deals[777] = [ENTRY_DEAL, EXIT_DEAL]
    (closed,) = ex.reconcile_open_positions_with_mt5()
    assert closed.pnl == pytest.approx(3.0) and closed.exit_price == 1.1013 and 777 not in rm.open_positions_cache


def test_after_the_wait_the_entry_is_dropped_with_an_alert_and_no_closed_trade(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(execution_module.time, "time", lambda: clock[0])
    ex, rm, notifier = make(FakeClient(positions=[], deals={777: [ENTRY_DEAL]}))
    assert ex.reconcile_open_positions_with_mt5() == []
    clock[0] += execution_module.CLOSE_DEAL_WAIT_SECONDS + 1
    assert ex.reconcile_open_positions_with_mt5() == []
    assert 777 not in rm.open_positions_cache
    assert any("777" in m for m in notifier.messages)


@pytest.mark.parametrize("name", ["DEAL_ENTRY_OUT_BY", "DEAL_ENTRY_INOUT"])
def test_a_close_by_or_a_reversal_counts_as_closing(monkeypatch, name):
    monkeypatch.setattr(mt5, name, 7, raising=False)
    deal = NS(entry=7, profit=2.0, commission=0.0, swap=0.0, fee=0.0, time=CLOSED, price=1.1010, magic=OURS)
    ex, rm, _ = make(FakeClient(positions=[], deals={777: [ENTRY_DEAL, deal]}))
    (closed,) = ex.reconcile_open_positions_with_mt5()
    assert closed.pnl == pytest.approx(1.5)


def test_exit_equity_is_the_fresh_account_equity():
    ex, rm, _ = make(FakeClient(positions=[], deals={777: [ENTRY_DEAL, EXIT_DEAL]}, equity=1234.5))
    (closed,) = ex.reconcile_open_positions_with_mt5()
    assert closed.exit_equity == 1234.5


def test_exit_equity_falls_back_to_the_monitor_when_the_account_cannot_be_read():
    ex, rm, _ = make(FakeClient(positions=[], deals={777: [ENTRY_DEAL, EXIT_DEAL]}, equity=None))
    (closed,) = ex.reconcile_open_positions_with_mt5()
    assert closed.exit_equity == 1000.0


def test_a_position_waiting_to_be_closed_is_not_managed():
    from src.config import Cfg as _Cfg
    cfg = _Cfg()
    cfg.data_source = "mt5"
    sent = []
    client = NS(symbol_info_tick=lambda s: NS(bid=1.1100, ask=1.1101), symbol_info=lambda s: NS(digits=5),
                order_send=lambda r: sent.append(r) or NS(retcode=mt5.TRADE_RETCODE_DONE, comment=""))
    rm = RiskManager(cfg, client, threading.Lock())
    rm.open_positions_cache[42] = {"ticket": 42, "symbol": SYMBOL, "direction": "long", "entry_price": 1.1000, "atr": 0.0010,
                                   "sl": 1.0990, "tp": 1.1050, "sl_atr_mult": 1.0, "risk": 1.0, "close_first_seen": 1.0}
    rm.manage_open_positions(SYMBOL, 0.0010)
    assert sent == []


# ---- the entry price is the fill, not the request ----
class TradeClient(FakeClient):
    def __init__(self, fill_price):
        super().__init__()
        self.fill_price = fill_price
        self.sent = []

    def order_send(self, request):
        self.sent.append(request)
        return NS(retcode=10009, deal=900, order=1, volume=request["volume"], price=request["price"])

    def history_deals_get(self, *a, **k):
        if k.get("ticket") == 900:
            return [NS(position_id=555, price=self.fill_price)]
        return []


def test_the_cache_keeps_the_fill_price_and_measures_the_stop_from_it():
    client = TradeClient(fill_price=1.1003)          # 2 pips of slippage against a long requested at 1.1001
    rm = RiskManager(Cfg(), client, threading.Lock())
    ex = Execution({}, {}, rm, client, NS(), dry_run=False, monitor=NS(current_equity=1000.0))
    ex.trade(SYMBOL, "long", 0.02, 1.1001, 1.0995, 1.1013, 1000.0, 1e-5, 1.0, NOW, atr=0.0006, auc_score=0.6)
    entry = rm.open_positions_cache[555]
    assert entry["entry_price"] == pytest.approx(1.1003)
    assert entry["sl_atr_mult"] == pytest.approx((1.1003 - 1.0995) / 0.0006)
    assert entry["risk_amount"] == pytest.approx(0.02 * (1.1003 - 1.0995) / 1e-5 * 1.0)


def test_a_missing_fill_price_keeps_the_requested_one():
    client = TradeClient(fill_price=None)
    rm = RiskManager(Cfg(), client, threading.Lock())
    ex = Execution({}, {}, rm, client, NS(), dry_run=False, monitor=NS(current_equity=1000.0))
    ex.trade(SYMBOL, "long", 0.02, 1.1001, 1.0995, 1.1013, 1000.0, 1e-5, 1.0, NOW, atr=0.0006, auc_score=0.6)
    assert rm.open_positions_cache[555]["entry_price"] == pytest.approx(1.1001)
