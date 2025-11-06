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
        
        # Configure mocks
        mock_initialize.return_value = True
        mock_login.return_value = True
        mock_account_info.return_value = MagicMock(login=12345)
        
        # Mock symbols_get to return a list of symbols
        mock_symbol = MagicMock()
        mock_symbol.name = "EURUSDm"
        mock_symbols_get.return_value = [mock_symbol]

        # Mock symbol_info_tick to return a tick with a time
        mock_tick = MagicMock()
        mock_tick.time = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
        mock_symbol_info_tick.return_value = mock_tick

        yield

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
    now = client.now_utc()
    assert isinstance(now, datetime.datetime)
    assert now.tzinfo == datetime.timezone.utc
