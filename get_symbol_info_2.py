import MetaTrader5 as mt5
import sys
import os

def get_info(symbol: str):
    """Connects to MT5 and prints contract size and pip size for a given symbol."""
    
    # Ensure MT5 is initialized
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        print("Please ensure your MT5 terminal is running.")
        return

    print(f"MT5 Initialized. Version: {mt5.version()}")

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Failed to find symbol '{symbol}'. It may not be enabled in your Market Watch.")
        print("Error code:", mt5.last_error())
        mt5.shutdown()
        return

    # For most non-forex instruments like GOLD, pip_size is the same as point size.
    pip_size = symbol_info.point

    print("\n" + "="*40)
    print(f"Symbol Information for: {symbol}")
    print("="*40)
    print(f"  -> Contract Size: {symbol_info.trade_contract_size}")
    print(f"  -> Pip Size:      {pip_size}")
    print("="*40)
    
    print("\nRECOMMENDED YAML CONFIGURATION:")
    print("---------------------------------")
    print("symbol_overrides:")
    print(f'  "{symbol}":')
    print(f"    contract_size: {symbol_info.trade_contract_size}")
    print(f"    pip_size: {pip_size}")
    print("---------------------------------")
    print("\nCopy the snippet above into your config.yaml file.")
    
    mt5.shutdown()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_symbol_info.py <SYMBOL>")
        print("Example: python get_symbol_info.py \"GOLDm#\"")
    else:
        get_info(sys.argv[1])

# run $python get_symbol_info_2.py "EURUSDm#"