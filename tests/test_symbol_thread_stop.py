"""K7: a symbol thread can be told to stop, so a reconnect does not leave the old threads looping next to the new ones.
Threads here are daemons and are joined with a timeout, so a regression leaves a stray thread instead of hanging the suite."""
import threading
import time
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from src.mt5_client import MT5Client
from src.symbol_processor import SymbolProcessor, stop_symbol_threads
from test_trade_gate import decide, make_rm, make_sp


class IdleClient:
    """Never sees a new bar (the wait times out)."""

    def wait_for_new_bar(self, symbol, timeframe, stop_event=None):
        return False


class BarClient:
    def wait_for_new_bar(self, symbol, timeframe, stop_event=None):
        return True


def make_loop_processor(client, fetch=None):
    sp = object.__new__(SymbolProcessor)
    sp.symbol, sp.mt5_client, sp.mt5_timeframe = "EURUSD#", client, 5
    sp.cfg = NS(timeframe_minutes=lambda: 5)               # a retry sleep would be 2.5 to 5 minutes
    sp.risk_controller = NS(increment_bar_counter=lambda symbol: None)
    sp.stop_event = threading.Event()
    if fetch is not None:
        sp._fetch_and_prepare_data = fetch
    return sp


def run_in_thread(sp):
    t = threading.Thread(target=sp.run_loop, daemon=True)
    t.start()
    time.sleep(0.1)
    return t


def test_run_loop_exits_promptly_when_stopped_while_waiting_for_a_bar():
    sp = make_loop_processor(IdleClient())
    t = run_in_thread(sp)
    sp.stop()
    t.join(timeout=2)
    assert not t.is_alive()


def test_run_loop_exits_promptly_when_stopped_during_the_sleep_after_an_error():
    def boom():
        raise RuntimeError("feed broke")

    sp = make_loop_processor(BarClient(), fetch=boom)
    t = run_in_thread(sp)
    sp.stop()
    t.join(timeout=2)
    assert not t.is_alive()


def test_a_stopped_processor_sends_no_order_even_when_the_signal_fires():
    sp = make_sp(make_rm())
    sp.stop_event.set()
    decide(sp)
    assert sp.execution.calls == []


# ---- wait_for_new_bar ----------------------------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _five_minute_bars():
    with patch("src.mt5_client.timeframe_to_seconds", return_value=300):   # the conftest MT5 stub has no real constants
        yield


def connected_client():
    client = MT5Client(login=1, password="x", server="s", path="p")
    client._connected = True
    return client


def test_wait_for_new_bar_returns_promptly_once_the_stop_event_is_set():
    client, stop = connected_client(), threading.Event()
    with patch("src.mt5_client.mt5.copy_rates_from_pos", return_value=[[1000, 0, 0, 0, 0]]):
        threading.Timer(0.2, stop.set).start()
        started = time.monotonic()
        assert client.wait_for_new_bar("EURUSD#", 5, stop_event=stop) is False
    assert time.monotonic() - started < 5


def test_wait_for_new_bar_without_an_event_still_returns_true_on_a_new_bar():
    client = connected_client()
    bars = iter([[[1000, 0, 0, 0, 0]], [[1300, 0, 0, 0, 0]]])
    with patch("src.mt5_client.mt5.copy_rates_from_pos", side_effect=lambda *a, **k: next(bars)):
        assert client.wait_for_new_bar("EURUSD#", 5) is True


# ---- stopping all symbol threads --------------------------------------------------------------------------------------

def test_stop_symbol_threads_stops_every_processor_and_reports_the_ones_still_running():
    release = threading.Event()
    stoppable_stop, stubborn_stop = threading.Event(), threading.Event()

    def stoppable():
        stoppable_stop.wait(10)

    def stubborn():
        release.wait(10)

    threads = [
        {"symbol": "EURUSD#", "thread": threading.Thread(target=stoppable, daemon=True), "processor": NS(stop=stoppable_stop.set)},
        {"symbol": "GBPUSD#", "thread": threading.Thread(target=stubborn, daemon=True), "processor": NS(stop=stubborn_stop.set)},
    ]
    for entry in threads:
        entry["thread"].start()
    try:
        assert stop_symbol_threads(threads, timeout=0.5) == ["GBPUSD#"]
        assert stoppable_stop.is_set() and stubborn_stop.is_set()
    finally:
        release.set()


def test_stop_symbol_threads_with_no_threads_is_a_no_op():
    assert stop_symbol_threads([], timeout=0.1) == []
