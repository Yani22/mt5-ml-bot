import MetaTrader5 as mt5
import os
from dotenv import load_dotenv
from loguru import logger

# Import the project's own MT5Client
from src.mt5_client import MT5Client

# Load .env at the global scope, just like main.py
load_dotenv()

def get_symbol_info():
    """
    Connects to MT5 using the project's MT5Client and fetches symbol information.
    """
    # Instantiate the client exactly as in main.py
    mt5c = MT5Client(
        login=os.getenv("MT5_LOGIN"),
        password=os.getenv("MT5_PASSWORD"),
        server=os.getenv("MT5_SERVER"),
        path=os.getenv("MT5_PATH"),
    )

    # Attempt to connect
    if not mt5c.connect():
        logger.error("Failed to connect to MT5 using the project's client. Please check credentials and MT5 terminal.")
        return

    symbol = "AUDUSDm#"
    
    # Get symbol info
    info = mt5.symbol_info(symbol)
    if info is None:
        logger.error(f"Symbol {symbol} not found. Please check the symbol name.")
        mt5c.shutdown()
        return

    print(f"\n--- Properties for symbol: {symbol} ---")
    print(f"Description: {info.description}")
    print(f"Contract Size: {info.trade_contract_size}")
    print(f"Margin Currency: {info.currency_margin}")
    print(f"Profit Currency: {info.currency_profit}")
    print(f"Minimum Volume (lots): {info.volume_min}")
    print(f"Maximum Volume (lots): {info.volume_max}")
    print(f"Volume Step (lot increment): {info.volume_step}")
    print(f"------------------------------------")

    # Shut down the connection
    mt5c.shutdown()

if __name__ == "__main__":
    get_symbol_info()