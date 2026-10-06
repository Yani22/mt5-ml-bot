"""Fill and stop rules of the backtester. Bars are bid prices; a long buys at the ask (bid + spread) and sells at the bid, a
short sells at the bid and buys back at the ask. Stops move by the rule `RiskManager.manage_open_positions` applies live."""
import math


def bar_spread(raw, fallback):
    """A bar's spread in price units: the file's value, or `fallback` when it is 0, negative, NaN or unreadable."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return fallback
    return value if value > 0 and not math.isnan(value) else fallback


def entry_price(direction, open_px, spread):
    return open_px + spread if direction == "long" else open_px


def exit_hit(direction, sl, tp, o, h, l, spread):
    """(price, reason) when the bar stops or targets the position, else None. Both touched: the stop (a bar has no intrabar order).
    A bar that opens beyond the stop fills at the open."""
    if direction == "long":
        if l <= sl:
            return min(sl, o), "Stop Loss"
        if h >= tp:
            return tp, "Take Profit"
        return None
    if h + spread >= sl:
        return max(sl, o + spread), "Stop Loss"
    if l + spread <= tp:
        return tp, "Take Profit"
    return None


def next_stop(direction, entry, sl, atr_entry, placed_mult, close, spread, atr_now, breakeven, trailing_mult):
    """The stop after the closed bar: breakeven at +1R (1R = placed multiple x entry ATR), then a trail of `trailing_mult` x ATR
    once +1R is reached or the stop already sits at entry. Only ever tightens. A long exits at the bid (close), a short at the ask."""
    is_long = direction == "long"
    exit_ref = close if is_long else close + spread
    move = (exit_ref - entry) if is_long else (entry - exit_ref)
    reached = bool(placed_mult) and move >= placed_mult * atr_entry - 1e-9
    at_entry = (sl >= entry) if is_long else (0 < sl <= entry)
    new_sl = sl
    if breakeven and reached and not at_entry:
        new_sl = entry
    if trailing_mult > 0 and (reached or at_entry):
        dist = atr_now * trailing_mult
        new_sl = max(new_sl, exit_ref - dist) if is_long else min(new_sl, exit_ref + dist)
    tightened = new_sl > sl + 1e-9 if is_long else 0 < new_sl < sl - 1e-9
    return new_sl if tightened else sl
