import pytest
from unittest.mock import MagicMock, patch
from src.mt5_client import MT5Client
import datetime

@pytest.fixture
def mock_mt5():
    with patch('MetaTrader5.initialize') as mock_initialize, \
         patch('MetaTrader5.login') as mock_login, \
         patch('MetaTrader5.account_info') as mock_account_info, \
         patch('MetaTrader5.shutdown') as mock_shutdown, \
         patch('MetaTrader5.symbols_get') as mock_symbols_get, \
         patch('MetaTrader5.symbol_info_tick') as mock_symbol_info_tick:
        
        mt5_mock = MagicMock()
        mt5_mock.initialize = mock_initialize
        mt5_mock.login = mock_login
        mt5_mock.account_info = mock_account_info
        mt5_mock.shutdown = mock_shutdown
        mt5_mock.symbols_get = mock_symbols_get
        mt5_mock.symbol_info_tick = mock_symbol_info_tick

        # Configure mocks
        mock_initialize.return_value = True
        mock_login.return_value = True
        mock_account_info.return_value = MagicMock(login=12345)
        
        # Mock symbols_get to return a list of symbols
        mock_symbol = MagicMock()
        mock_symbol.name = "EURUSDm#"
        mock_symbols_get.return_value = [mock_symbol]

        # Mock symbol_info_tick to return a tick with a time
        mock_tick = MagicMock()
        mock_tick.time = 1678886400  # A specific timestamp
        mock_symbol_info_tick.return_value = mock_tick

        yield mt5_mock

def test_mt5_client_initialization(mock_mt5):
    client = MT5Client(login=12345, password="password", server="server", path="path")
    assert client.login == 12345
    assert client.password == "password"
    assert client.server == "server"
    assert client.path == "path"

def test_mt5_client_connection_success(mock_mt5):
    client = MT5Client(login=12345, password="password", server="server", path="path")
    client.connect()
    assert client.is_connected
    # Verify that initialize and login were called
    from MetaTrader5 import initialize, login
    initialize.assert_called_once_with(path="path")
    login.assert_called_once_with(login=12345, password="password", server="server")

def test_mt5_client_now_utc(mock_mt5):
    client = MT5Client(login=12345, password="password", server="server", path="path")
    client.connect()

    # Test when connected
    now_utc_time = client.now_utc()
    expected_time = datetime.datetime.fromtimestamp(1678886400, tz=datetime.timezone.utc)
    assert now_utc_time == expected_time
    mock_mt5.symbols_get.assert_called_once()
    mock_mt5.symbol_info_tick.assert_called_once_with("EURUSDm#")

    # Test when not connected (should fall back to system time)
    client.shutdown()
    mock_mt5.symbols_get.reset_mock()
    mock_mt5.symbol_info_tick.reset_mock()
    
    # Mock datetime.datetime.now for consistent testing of fallback
    original_datetime = datetime.datetime
    with patch('datetime.datetime') as mock_dt:
        mock_dt.now.return_value = original_datetime(2023, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
        mock_dt.side_effect = lambda *args, **kw: original_datetime(*args, **kw)
        mock_dt.timezone = datetime.timezone

        now_utc_time_disconnected = client.now_utc()
        assert now_utc_time_disconnected == original_datetime(2023, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
        mock_mt5.symbols_get.assert_not_called()
        mock_mt5.symbol_info_tick.assert_not_called()


def test_mt5_client_account_info(mock_mt5):
    client = MT5Client(login=12345, password="password", server="server", path="path")

    # Test when not connected
    assert client.account_info() is None
    mock_mt5.account_info.assert_not_called()

    # Connect the client
    client.connect()
    mock_mt5.account_info.reset_mock() # Reset mock after connect() calls it internally

    # Test when connected
    mock_account_info = MagicMock()
    mock_account_info.login = 12345
    mock_account_info.balance = 1000.0
    mock_mt5.account_info.return_value = mock_account_info
    account_info = client.account_info()
    assert account_info == mock_account_info
    mock_mt5.account_info.assert_called_once()

    # Test after shutdown
    client.shutdown()
    mock_mt5.account_info.reset_mock()
    assert client.account_info() is None
    mock_mt5.account_info.assert_not_called()

def test_mt5_client_shutdown(mock_mt5):
    client = MT5Client(login=12345, password="password", server="server", path="path")
    client.connect()
    assert client.is_connected()
    client.shutdown()
    assert not client.is_connected()
    from MetaTrader5 import shutdown
    shutdown.assert_called_once()

def test_mt5_client_connection_failure(mock_mt5):
    mock_mt5.initialize.return_value = False
    client = MT5Client(login=12345, password="password", server="server", path="path")
    client.connect()
    assert not client.is_connected()
    from MetaTrader5 import initialize
    assert initialize.call_count == 3
