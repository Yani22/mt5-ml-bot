import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# MetaTrader5 has no Linux wheel; stub it so tests can import src.mt5_client.
try:
    import MetaTrader5  # noqa: F401
except ImportError:
    import types
    _stub = types.ModuleType("MetaTrader5")
    for _name in ("initialize", "login", "account_info", "shutdown", "symbols_get",
                  "symbol_info_tick", "terminal_info", "symbol_info", "last_error",
                  "copy_rates_from_pos", "copy_rates_from", "copy_rates_range"):
        setattr(_stub, _name, lambda *a, **k: None)
    sys.modules["MetaTrader5"] = _stub
    for _i, _name in enumerate((
        "DEAL_ENTRY_OUT", "ORDER_FILLING_IOC", "ORDER_TIME_GTC", "ORDER_TYPE_BUY",
        "ORDER_TYPE_SELL", "POSITION_TYPE_BUY", "TIMEFRAME_D1", "TIMEFRAME_H1",
        "TIMEFRAME_H4", "TIMEFRAME_M1", "TIMEFRAME_M15", "TIMEFRAME_M30",
        "TIMEFRAME_M5", "TIMEFRAME_MN1", "TIMEFRAME_W1", "TRADE_ACTION_DEAL",
        "TRADE_ACTION_SLTP", "TRADE_RETCODE_DONE")):
        setattr(_stub, _name, _i)
    # Read from the XM terminal's Python package (not the sequential stand-ins above): deposits, withdrawals, credit, bonus
    for _name, _value in ("DEAL_TYPE_BALANCE", 2), ("DEAL_TYPE_CREDIT", 3), ("DEAL_TYPE_CORRECTION", 5), ("DEAL_TYPE_BONUS", 6):
        setattr(_stub, _name, _value)


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_shared_mt5_connection():
    """src.mt5_client keeps process-wide connection state; do not let it leak between tests."""
    from src import mt5_client
    reset = getattr(mt5_client, "_reset_shared_state", None)
    if reset:
        reset()
    yield
    if reset:
        reset()
