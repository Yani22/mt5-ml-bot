def round_trip_pips(spread_pips: float, slippage_pips: float, adaptive_slippage: bool = False,
                    adaptive_slippage_multiplier: float = 1.0) -> float:
    """Pips one round trip costs before commission: one spread (the bars are bid-only and entry and exit both use the bar
    close, so a long or a short pays exactly one) plus slippage, scaled by the adaptive multiplier when that is enabled.
    Shared by the backtester and the threshold search so the two cannot drift apart."""
    slippage = float(slippage_pips) * (float(adaptive_slippage_multiplier) if adaptive_slippage else 1.0)
    return float(spread_pips) + slippage
