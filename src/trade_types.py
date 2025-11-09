from dataclasses import dataclass
import datetime
from typing import Optional, List

@dataclass
class ClosedTrade:
    """A comprehensive closed trade object for all post-trade processing."""
    ticket: int
    symbol: str
    direction: str
    lots: float
    entry_price: float
    exit_price: float
    entry_time: datetime.datetime
    exit_time: datetime.datetime
    pnl: float
    risk_fraction: float
    atr: float
    atr_idx: int
    min_prob_long_idx: int
    min_prob_short_idx: int
    entry_auc: float
    entry_equity: Optional[float] = None
    exit_equity: Optional[float] = None
    adx: float = 0.0
    macd_diff: float = 0.0
    volatility_10: float = 0.0
    dist_from_ema_200: float = 0.0
    inter_market_feature: float = 0.0
    mta_feature: float = 0.0
    context_vector: Optional[List[float]] = None # NEW: Store context vector for contextual bandits

    def __repr__(self):
        return f"<ClosedTrade ticket={self.ticket}, pnl={self.pnl:.2f}>"
