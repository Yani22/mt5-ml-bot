"""B14: the MetaTrader5 package holds ONE connection per process, but the bot makes many MT5Client objects (main, one per
symbol, one per DataManager). They must share it: later clients reuse a healthy connection instead of initialising and
logging in again, one client's shutdown must not close the connection the others use, and a reconnect is an explicit
teardown after which every old client reads as disconnected and the next connect initialises afresh."""
from unittest.mock import MagicMock, patch

import pytest

from src import mt5_client as m
from src.mt5_client import MT5Client, teardown_connection


@pytest.fixture
def mt5():
    api = MagicMock()
    api.initialize.return_value = True
    api.login.return_value = True
    api.account_info.return_value = MagicMock(login=12345)
    api.terminal_info.return_value = MagicMock()
    with patch.object(m, "mt5", api):
        yield api


def client(**kw):
    args = dict(login=12345, password="pw", server="srv", path="path", max_retries=1, retry_delay=0.0)
    args.update(kw)
    return MT5Client(**args)


def test_three_clients_initialise_and_log_in_once(mt5):
    clients = [client() for _ in range(3)]
    assert all(c.connect() for c in clients)
    assert mt5.initialize.call_count == 1 and mt5.login.call_count == 1
    assert all(c.is_connected() for c in clients)


def test_a_later_client_reuses_a_healthy_connection_without_touching_it(mt5):
    client().connect()
    mt5.reset_mock()
    assert client(password="other").connect() is True
    assert not (mt5.initialize.called or mt5.login.called or mt5.shutdown.called)


def test_one_clients_shutdown_does_not_close_the_connection_the_others_use(mt5):
    a, b = client(), client()
    a.connect()
    b.connect()
    a.shutdown()
    assert not mt5.shutdown.called
    assert not a.is_connected() and b.is_connected()
    assert b.account_info() is not None


def test_teardown_closes_the_connection_once_and_every_old_client_reads_disconnected(mt5):
    clients = [client() for _ in range(3)]
    for c in clients:
        c.connect()
    teardown_connection()
    assert mt5.shutdown.call_count == 1
    assert not any(c.is_connected() for c in clients)
    assert all(c.account_info() is None and c.symbol_info("X") is None for c in clients)


def test_after_a_teardown_the_next_connect_initialises_again(mt5):
    client().connect()
    teardown_connection()
    fresh = client()
    assert fresh.connect() is True
    assert mt5.initialize.call_count == 2 and fresh.is_connected()


def test_an_unhealthy_connection_is_not_shared_and_is_initialised_again(mt5):
    client().connect()
    mt5.terminal_info.return_value = None            # the terminal went away
    again = client()
    assert again.connect() is True
    assert mt5.initialize.call_count == 2
    assert not mt5.shutdown.called                   # a false 'unhealthy' reading must not cut off working threads


def test_a_client_that_was_never_connected_can_shut_down_without_error(mt5):
    client().shutdown()
    assert not mt5.shutdown.called


def test_the_retry_sleep_does_not_hold_the_shared_lock(mt5):
    mt5.initialize.return_value = False
    mt5.last_error.return_value = (1, "fail")
    held_during_sleep = []
    with patch.object(m.time, "sleep", side_effect=lambda s: held_during_sleep.append(m._state_lock.locked())):
        assert client(max_retries=2, retry_delay=0.01).connect() is False
    assert held_during_sleep == [False, False]
