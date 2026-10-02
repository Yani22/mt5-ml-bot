"""Refuses to place real orders unless live trading was switched on explicitly."""
import os
from typing import Mapping, Optional

FLAG = "ALLOW_LIVE_TRADING"


class LiveTradingNotAllowed(RuntimeError):
    pass


def require_live_permission(dry_run: bool, env: Optional[Mapping[str, str]] = None) -> None:
    """Dry-run needs no permission. A live run needs ALLOW_LIVE_TRADING=1 in the environment (or .env)."""
    if dry_run:
        return
    env = os.environ if env is None else env
    if env.get(FLAG) != "1":
        raise LiveTradingNotAllowed(
            f"Live trading is disabled: this bot places real orders. Set {FLAG}=1 in the environment only after "
            "reading the README, otherwise run with dry_run=True."
        )
