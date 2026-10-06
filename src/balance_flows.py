"""Deposits, withdrawals, credit, bonus and corrections move equity without a trade. A peak equity that does not move with
them shows a withdrawal as a drawdown (and a deposit would hide a real one), so the peak is rescaled by the net money paid in
or out, read from the broker's balance deals (B7). The block is a FRACTION of the peak, so the peak is scaled in proportion
(the drawdown fraction is unchanged), not moved by the dollar amount."""
from __future__ import annotations

import datetime
import threading
from typing import Iterable, Optional

from loguru import logger  # type: ignore

try:
    import MetaTrader5 as mt5  # type: ignore
except ImportError:  # pragma: no cover
    mt5 = None

# Deal times are on the server clock and a late deal can post after a look, so each look re-reads this much history and the
# deal ticket decides what is new.
LOOKBACK_MARGIN = datetime.timedelta(days=1)


def _flow_types() -> set:
    """The deal types that are money in or out of the account, not trades."""
    if mt5 is None:
        return set()
    return {getattr(mt5, name) for name in ("DEAL_TYPE_BALANCE", "DEAL_TYPE_CREDIT", "DEAL_TYPE_CORRECTION", "DEAL_TYPE_BONUS")
            if hasattr(mt5, name)}


def shift_peak(peak: float, equity_after: float, net: float) -> float:
    """The peak after `net` money was paid in (+) or out (-), given the equity read right after it. Keeps the drawdown
    fraction `1 - equity / peak` as it was before the flow. When the equity before the flow was not positive there is no
    fraction to keep, and the peak is the equity."""
    equity_before = equity_after - net
    if equity_before <= 0 or equity_after <= 0:
        return max(equity_after, 0.0)
    return peak * equity_after / equity_before


class BalanceFlowCursor:
    """Remembers which balance deals have been counted. The first look only records what is already in the history (the equity
    read next to it already includes those deals); later looks return the net of deals not seen before."""

    def __init__(self, since: Optional[datetime.datetime] = None, tickets: Iterable[int] = ()):
        self.since = since
        self.tickets = set(tickets)
        self._lock = threading.Lock()  # symbol threads share one RiskManager: a deal must be counted once

    def poll(self, mt5_client, now: datetime.datetime) -> float:
        """Net money paid in (+) or out (-) since the last look. A failed read returns 0.0 and the next look re-reads the
        window, so nothing is lost and the peak is never lowered on an error."""
        with self._lock:
            return self._poll(mt5_client, now)

    def _poll(self, mt5_client, now: datetime.datetime) -> float:
        first_look = self.since is None
        start = now - LOOKBACK_MARGIN if first_look else self.since - LOOKBACK_MARGIN
        try:
            deals = mt5_client.history_deals_get(start, now + LOOKBACK_MARGIN)
        except Exception as e:
            logger.warning(f"Balance deals could not be read ({e}); the equity peak is left as it is.")
            return 0.0
        types = _flow_types()
        net = 0.0
        for d in deals or []:
            ticket = getattr(d, "ticket", None)
            if ticket is None or getattr(d, "type", None) not in types or ticket in self.tickets:
                continue
            self.tickets.add(ticket)
            if not first_look:
                net += float(getattr(d, "profit", 0.0) or 0.0)
        if deals is not None or first_look:
            self.since = now
        return net

    def to_state(self) -> dict:
        return {"since": self.since.isoformat() if self.since else None, "tickets": sorted(self.tickets)}

    @classmethod
    def from_state(cls, state: Optional[dict]) -> "BalanceFlowCursor":
        """A state saved before this existed (None) starts from now: old deposits and withdrawals are never replayed."""
        try:
            since = datetime.datetime.fromisoformat(state["since"]) if state and state.get("since") else None
            return cls(since, state.get("tickets", []) if state else [])
        except Exception:
            return cls()
