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
         patch('MetaTrader5.symbol_info_tick') as mock_symbol_info_tick, \
         patch('MetaTrader5.terminal_info') as mock_terminal_info:
        
        mt5_mock = MagicMock()
        mt5_mock.initialize = mock_initialize
        mt5_mock.login = mock_login
        mt5_mock.account_info = mock_account_info
        mt5_mock.shutdown = mock_shutdown
        mt5_mock.symbols_get = mock_symbols_get
        mt5_mock.symbol_info_tick = mock_symbol_info_tick
        mt5_mock.terminal_info = mock_terminal_info

        # Configure mocks
        mock_initialize.return_value = True
        mock_login.return_value = True
        mock_account_info.return_value = MagicMock(login=12345)
        
        # Mock symbols_get to return a list of symbols
        mock_eurusdm_symbol = MagicMock()
        mock_eurusdm_symbol.name = "EURUSDm#"
        mock_gbpusdm_symbol = MagicMock()
        mock_gbpusdm_symbol.name = "GBPUSDm#"
        mock_symbols_get.return_value = [mock_eurusdm_symbol, mock_gbpusdm_symbol]

        # Mock symbol_info_tick to return a tick with a time
        def symbol_info_tick_side_effect(symbol_name):
            if symbol_name == "EURUSDm#":
                mock_tick = MagicMock()
                mock_tick.time = 1678886400  # A specific timestamp for EURUSDm
                return mock_tick
            elif symbol_name == "GBPUSDm#":
                mock_tick = MagicMock()
                mock_tick.time = 1678886500  # A specific timestamp for GBPUSDm
                return mock_tick
            return None # Default for other symbols
        
        mock_symbol_info_tick.side_effect = symbol_info_tick_side_effect

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

def test_mt5_client_now_utc_from_symbol_tick(mock_mt5):
    client = MT5Client(login=12345, password="password", server="server", path="path")
    client.connect()

    # EURUSDm# fails, but GBPUSDm# succeeds
    def symbol_info_tick_side_effect_fallback(symbol_name):
        if symbol_name == "EURUSDm#":
            return None # Simulate failure for EURUSDm#
        elif symbol_name == "GBPUSDm#":
            return MagicMock(time=1678886500) # Simulate success for GBPUSDm#
        return None
    mock_mt5.symbol_info_tick.side_effect = symbol_info_tick_side_effect_fallback
    
    mock_eurusdm_symbol = MagicMock()
    mock_eurusdm_symbol.name = "EURUSDm#"
    mock_gbpusdm_symbol = MagicMock()
    mock_gbpusdm_symbol.name = "GBPUSDm#"
    mock_mt5.symbols_get.return_value = [mock_eurusdm_symbol, mock_gbpusdm_symbol]

    now_utc_time = client.now_utc()
    expected_time = datetime.datetime.fromtimestamp(1678886500, tz=datetime.timezone.utc)
    assert now_utc_time == expected_time
    # Verify calls
    assert mock_mt5.symbol_info_tick.call_count == 2 # Once for EURUSDm#, once for GBPUSDm#
    mock_mt5.symbol_info_tick.assert_any_call("EURUSDm#")
    mock_mt5.symbol_info_tick.assert_any_call("GBPUSDm#")
    mock_mt5.symbols_get.assert_called_once()

def test_get_timezone_offset(mock_mt5):
    client = MT5Client(login=12345, password="password", server="server", path="path")
    client.connect()

    # Mock server time (UTC) and system time (UTC)
    server_time = datetime.datetime(2023, 1, 1, 14, 0, 0, tzinfo=datetime.timezone.utc)
    system_time = datetime.datetime(2023, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)

    with patch.object(client, 'now_utc', return_value=server_time), \
         patch('datetime.datetime') as mock_dt:
        mock_dt.now.return_value = system_time
        mock_dt.side_effect = lambda *args, **kw: datetime.datetime(*args, **kw)
        mock_dt.timezone = datetime.timezone

        offset = client.get_timezone_offset()
        assert offset == 2.0

def test_mt5_client_now_utc_fallback_to_system_time(mock_mt5):
    client = MT5Client(login=12345, password="password", server="server", path="path")
    client.connect()

    # Test when connected but no symbols provide a valid tick
    mock_mt5.symbol_info_tick.side_effect = lambda symbol_name: None # No valid ticks
    mock_mt5.symbols_get.return_value = [MagicMock(name="SYMBOL1"), MagicMock(name="SYMBOL2")]

    # Mock datetime.datetime.now for consistent testing of fallback
    original_datetime = datetime.datetime
    with patch('datetime.datetime') as mock_dt:
        mock_dt.now.return_value = original_datetime(2023, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
        mock_dt.side_effect = lambda *args, **kw: original_datetime(*args, **kw)
        mock_dt.timezone = datetime.timezone

        now_utc_time = client.now_utc()
        assert now_utc_time == original_datetime(2023, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
        mock_mt5.symbol_info_tick.assert_any_call("EURUSDm#") # Initial attempt
        mock_mt5.symbols_get.assert_called_once() # Fallback to iterating symbols
        assert mock_mt5.symbol_info_tick.call_count > 1 # Called for EURUSDm# and then for other symbols

    # Test when not connected (should fall back to system time directly)
    client.shutdown()
    mock_mt5.symbols_get.reset_mock()
    mock_mt5.symbol_info_tick.reset_mock()
    
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
