import MetaTrader5 as mt5

def timeframe_to_mt5_timeframe(timeframe_str: str):
    """Converts a timeframe string (e.g., 'M5') to a MetaTrader 5 timeframe constant."""
    timeframe_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    return timeframe_map.get(timeframe_str, mt5.TIMEFRAME_M1)

def timeframe_to_seconds(timeframe: int | str) -> int:
    """Converts a MetaTrader 5 timeframe constant (e.g., mt5.TIMEFRAME_M5) or string ('M5') to seconds."""
    if isinstance(timeframe, int):
        # Handle MT5 integer constants
        if timeframe == mt5.TIMEFRAME_M1: return 60
        if timeframe == mt5.TIMEFRAME_M5: return 300
        if timeframe == mt5.TIMEFRAME_M15: return 900
        if timeframe == mt5.TIMEFRAME_M30: return 1800
        if timeframe == mt5.TIMEFRAME_H1: return 3600
        if timeframe == mt5.TIMEFRAME_H4: return 14400
        if timeframe == mt5.TIMEFRAME_D1: return 86400
        if timeframe == mt5.TIMEFRAME_W1: return 604800
        if timeframe == mt5.TIMEFRAME_MN1: return 2592000 # Approximate for 30 days
        raise ValueError(f"Unsupported MT5 integer timeframe: {timeframe}")
    elif isinstance(timeframe, str):
        if timeframe.startswith('M'):
            minutes = int(timeframe[1:])
            return minutes * 60
        elif timeframe.startswith('H'):
            hours = int(timeframe[1:])
            return hours * 60 * 60
        elif timeframe.startswith('D'):
            return 24 * 60 * 60 # D1
        elif timeframe.startswith('W'):
            return 7 * 24 * 60 * 60 # W1
        elif timeframe.startswith('MN'):
            return 30 * 24 * 60 * 60 # MN1 (approx)
        raise ValueError(f"Unsupported string timeframe: {timeframe}")
    else:
        raise TypeError(f"timeframe must be int or str, got {type(timeframe)}")