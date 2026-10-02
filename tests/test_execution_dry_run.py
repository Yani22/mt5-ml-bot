"""Dry-run bookkeeping: simulated positions must be complete, must be able to close, and must never poison a live run."""
import datetime
import threading
from types import SimpleNamespace as NS

import MetaTrader5 as mt5  # stubbed by tests/conftest.py on Linux
import pytest

from src.config import Cfg
from src.execution import Execution
from src.risk import RiskManager

NOW = datetime.datetime(2026, 1, 5, 10, 0, tzinfo=datetime.timezone.utc)


@pytest.fixture(autouse=True)
def _tmp_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # Execution creates results/ relative to the working directory


class FakeClient:
    ORDER_TYPE_BUY, ORDER_TYPE_SELL, TRADE_ACTION_DEAL, ORDER_TIME_GTC = 0, 1, 1, 0
    ORDER_FILLING_IOC, TRADE_RETCODE_DONE = 1, 10009

    def __init__(self, positions=(), deals=None):
        self.positions, self.deals = list(positions), deals or {}

    def symbol_info_tick(self, symbol):
        return NS(ask=1.1001, bid=1.1000)

    def symbol_info(self, symbol):
        return NS(point=1e-5, trade_contract_size=100000, digits=5, trade_stops_level=0,
                  volume_min=0.01, volume_step=0.01, volume_max=100)

    def positions_get(self, *a, **k):
        return self.positions

    def history_deals_get(self, *a, **k):
        return self.deals.get(k.get("position"), [])


class FakeData:
    def __init__(self, prices):
        self.prices = prices

    def get_latest_bar_close(self, symbol):
        return self.prices.get(symbol)


def make(dry_run, client=None, prices=None):
    client = client or FakeClient()
    rm = RiskManager(Cfg(), client, threading.Lock())
    ex = Execution({}, {}, rm, client, FakeData(prices or {}), dry_run=dry_run, monitor=NS(current_equity=1000.0))
    return ex, rm


def open_long(ex, symbol="EURUSD#"):
    ex.trade(symbol, "long", 0.02, 1.1001, 1.0995, 1.1013, 1000.0, 1e-5, 1.0, NOW, atr=0.0006, auc_score=0.6)


def test_dry_run_position_records_what_closing_and_the_symbol_guard_need():
    ex, rm = make(dry_run=True)
    open_long(ex)
    ((ticket, pos),) = rm.open_positions_cache.items()
    assert pos["ticket"] == ticket and pos["dry_run"] is True
    assert (pos["symbol"], pos["direction"], pos["entry_price"], pos["lots"]) == ("EURUSD#", "long", 1.1001, 0.02)


def test_symbol_guard_blocks_a_second_dry_run_position_on_the_same_symbol():
    ex, rm = make(dry_run=True)
    open_long(ex)
    # total_open_risk=0, so only the one-position-per-symbol guard can block this
    assert rm.position_size(1000.0, 0.0006, 0.6, 0.0, "EURUSD#") == (0.0, 0.0)


@pytest.mark.parametrize("price, exit_price, pnl", [(1.1020, 1.1013, 2.4), (1.0990, 1.0995, -1.2)])
def test_dry_run_position_closes_at_target_or_stop_through_reconcile(price, exit_price, pnl):
    ex, rm = make(dry_run=True, prices={"EURUSD#": price})
    open_long(ex)
    closed = ex.reconcile_open_positions_with_mt5()
    assert len(closed) == 1 and closed[0].symbol == "EURUSD#"
    assert closed[0].exit_price == exit_price and closed[0].pnl == pytest.approx(pnl)
    assert rm.open_positions_cache == {}


def test_dry_run_position_stays_open_between_stop_and_target():
    ex, rm = make(dry_run=True, prices={"EURUSD#": 1.1005})
    open_long(ex)
    assert ex.reconcile_open_positions_with_mt5() == []
    assert len(rm.open_positions_cache) == 1


def test_old_dry_run_entries_without_a_symbol_are_dropped_not_kept_forever():
    ex, rm = make(dry_run=True)
    rm.open_positions_cache[111] = {"risk": 42.5, "sl": 1.0, "tp": 2.0}  # shape saved by earlier versions
    assert ex.reconcile_open_positions_with_mt5() == []
    assert rm.open_positions_cache == {}


def test_live_reconcile_drops_leftover_dry_run_entries_and_still_reconciles_real_trades():
    deal = NS(entry=mt5.DEAL_ENTRY_OUT, profit=3.5, time=1767607200, price=1.1013)
    ex, rm = make(dry_run=False, client=FakeClient(positions=[], deals={777: [deal]}))
    rm.open_positions_cache[555] = {"risk": 42.5, "sl": 1.0, "tp": 2.0}  # old dry-run entry: no symbol
    rm.open_positions_cache[666] = {"symbol": "EURUSD#", "direction": "long", "lots": 0.02, "entry_price": 1.1001,
                                    "entry_time": NOW, "dry_run": True}  # simulated, never existed at the broker
    rm.open_positions_cache[777] = {"symbol": "EURUSD#", "direction": "long", "lots": 0.02, "entry_price": 1.1001,
                                    "entry_time": NOW}  # real trade that has closed at the broker
    closed = ex.reconcile_open_positions_with_mt5()
    assert [t.ticket for t in closed] == [777] and closed[0].pnl == 3.5
    assert rm.open_positions_cache == {}
