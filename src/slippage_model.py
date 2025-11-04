# src/slippage_model.py
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Cfg, TradingCostsDefaultsCfg

def calculate_dynamic_slippage(
    cfg: "Cfg",
    symbol: str,
    current_atr: float,
    current_spread_pips: float,
    order_lots: float
) -> float:
    """
    Calculates dynamic slippage in pips based on market conditions and order size.

    Args:
        cfg: The main configuration object.
        symbol: The trading symbol (e.g., "EURUSDm#").
        current_atr: The current Average True Range for the symbol.
        current_spread_pips: The current bid-ask spread in pips for the symbol.
        order_lots: The volume of the order in lots.

    Returns:
        The calculated dynamic slippage in pips.
    """
    tc_defaults: "TradingCostsDefaultsCfg" = cfg.trading_costs.defaults

    if not tc_defaults.dynamic_slippage_enabled:
        return tc_defaults.slippage_pips # Return static slippage if dynamic is disabled

    # Get multipliers from config
    atr_mult = tc_defaults.slippage_atr_multiplier
    spread_mult = tc_defaults.slippage_spread_multiplier
    lot_mult = tc_defaults.slippage_lot_multiplier

    # Calculate slippage components
    slippage_from_atr = atr_mult * current_atr
    slippage_from_spread = spread_mult * current_spread_pips
    slippage_from_lots = lot_mult * math.log1p(order_lots) # Using log1p to handle small lots gracefully

    # Combine components. A simple sum for now, can be weighted or more complex.
    dynamic_slippage_pips = slippage_from_atr + slippage_from_spread + slippage_from_lots

    # Ensure slippage is always positive
    return max(0.0, dynamic_slippage_pips)
