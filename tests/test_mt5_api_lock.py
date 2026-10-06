"""B14 (the part fix 26 left open): the MetaTrader5 package holds one connection per process, but the symbol threads, the
5 s reconcile loop and the bar fetches call it at the same time, and `last_error()` is process-wide, so another thread's call
can overwrite the error of a failed one. Every `mt5.*` call goes through one process-wide lock (`src/mt5_lock.py`): connect and
teardown hold it with the state lock, the wrappers hold it per call, and `copy_rates_from_pos` and its `last_error()` read share
one hold. A hung call therefore stalls the other threads until it returns."""
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import MetaTrader5 as real_mt5
import pytest

from src import mt5_client as m
from src.data_manager import DataManager
from src.mt5_client import MT5Client, teardown_connection

ROOT = Path(__file__).resolve().parent.parent
WAIT = 0.3          # how long "the other thread did not get in" is observed
JOIN = 3.0          # a thread that should finish does so well inside this


@pytest.fixture
def mt5():
    api = MagicMock()
    api.initialize.return_value = True
    api.login.return_value = True
    api.account_info.return_value = MagicMock(login=12345)
    api.terminal_info.return_value = MagicMock()
    with patch.object(m, "mt5", api), patch.dict(sys.modules, {"MetaTrader5": api}):
        yield api


def connected(**kw):
    args = dict(login=12345, password="pw", server="srv", path="path", max_retries=1, retry_delay=0.0)
    args.update(kw)
    c = MT5Client(**args)
    assert c.connect()
    return c


def start(fn, *args):
    t = threading.Thread(target=fn, args=args, daemon=True)
    t.start()
    return t


def block_in(api_fn):
    """Make `api_fn` stop inside the call until released. Returns (entered, release)."""
    entered, release = threading.Event(), threading.Event()

    def blocked(*a, **k):
        entered.set()
        release.wait(JOIN)
        return MagicMock()
    api_fn.side_effect = blocked
    return entered, release


def test_a_second_wrapper_call_waits_for_the_first(mt5):
    c = connected()
    entered, release = block_in(mt5.account_info)
    second = threading.Event()
    mt5.symbol_info.side_effect = lambda s: second.set() or MagicMock()
    a = start(c.account_info)
    assert entered.wait(JOIN)
    b = start(c.symbol_info, "USDJPY#")
    assert not second.wait(WAIT)             # blocked while the first call is inside MT5
    release.set()
    a.join(JOIN), b.join(JOIN)
    assert second.is_set()


def test_a_wrapper_call_waits_for_a_bar_fetch_in_the_data_manager(mt5):
    c = connected()
    entered, release = block_in(mt5.copy_rates_from_pos)
    second = threading.Event()
    mt5.symbol_info.side_effect = lambda s: second.set() or MagicMock()
    dm = object.__new__(DataManager)
    a = start(dm._fetch_bars_from_mt5_chunked, "USDJPY#", "M5", 100)
    assert entered.wait(JOIN)
    b = start(c.symbol_info, "USDJPY#")
    assert not second.wait(WAIT)
    release.set()
    a.join(JOIN), b.join(JOIN)
    assert second.is_set()


def test_another_threads_call_cannot_land_between_a_failed_fetch_and_its_error_read(mt5):
    c = connected()
    order = []
    mt5.symbol_info.side_effect = lambda s: order.append("symbol_info") or MagicMock()
    mt5.last_error.side_effect = lambda: order.append("last_error") or (1, "Success")

    def copy(*a, **k):
        start(c.symbol_info, "USDJPY#")      # another thread arrives while the fetch is inside MT5
        time.sleep(WAIT)
        order.append("copy")
        return None
    mt5.copy_rates_from_pos.side_effect = copy
    dm = object.__new__(DataManager)
    dm._fetch_bars_from_mt5_chunked("USDJPY#", "M5", 100)
    time.sleep(WAIT)
    assert order == ["copy", "last_error", "symbol_info"]


def test_teardown_waits_for_a_call_that_is_inside_mt5(mt5):
    c = connected()
    entered, release = block_in(mt5.order_send)
    a = start(c.order_send, {})
    assert entered.wait(JOIN)
    b = start(teardown_connection)
    time.sleep(WAIT)
    assert not mt5.shutdown.called           # the connection is not closed under a running call
    release.set()
    a.join(JOIN), b.join(JOIN)
    assert mt5.shutdown.called


def test_a_connect_alongside_a_running_call_finishes_without_a_deadlock(mt5):
    c = connected()
    entered, release = block_in(mt5.symbol_info)
    a = start(c.symbol_info, "USDJPY#")
    assert entered.wait(JOIN)
    other = []
    b = start(lambda: other.append(MT5Client(12345, "pw", "srv", "path", max_retries=1, retry_delay=0.0).connect()))
    time.sleep(WAIT)
    release.set()
    a.join(JOIN), b.join(JOIN)
    assert not a.is_alive() and not b.is_alive() and other == [True]


def test_another_thread_can_call_while_wait_for_new_bar_waits(mt5):
    c = connected()
    mt5.copy_rates_from_pos.return_value = [[1000, 1, 1, 1, 1]]     # the bar never changes: the wait goes on
    stop = threading.Event()
    waiting = []
    a = start(lambda: waiting.append(c.wait_for_new_bar("USDJPY#", real_mt5.TIMEFRAME_M5, stop_event=stop)))
    time.sleep(WAIT)                                                # the first poll is done, the wait is sleeping
    done = threading.Event()
    start(lambda: (c.symbol_info("USDJPY#"), done.set()))
    assert done.wait(0.5)
    assert a.is_alive()                                            # still waiting: the thread did not die early
    stop.set()
    a.join(JOIN)
    assert waiting == [False]


def test_the_lock_module_and_the_offline_bar_loader_import_without_the_metatrader5_package():
    """The lock lives in a module with no MetaTrader5 import, so a module that takes it does not gain a hard dependency on the
    package. Run outside pytest, where no stub is installed. (`src.mt5_client` and `src.time_utils` also import without it now: see
    tests/test_offline_import_without_mt5.py.)"""
    code = ("import sys; sys.modules['MetaTrader5'] = None\n"
            "import src.mt5_lock, src.data; print('ok')")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr[-500:]
