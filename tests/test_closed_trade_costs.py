"""B11: the profit of a closed trade is net of commission, swap and fee, summed over all the position's deals (the entry deal
usually carries the commission). deal.profit alone is the gross price move."""
import datetime
from types import SimpleNamespace as NS

import MetaTrader5 as mt5
import pytest

from test_execution_dry_run import FakeClient, make, OURS, NOW  # noqa: F401
from test_execution_dry_run import _tmp_cwd  # noqa: F401

OPENING = mt5.DEAL_ENTRY_OUT + 1        # under the conftest stub DEAL_ENTRY_OUT is 0, which is real MT5's DEAL_ENTRY_IN
OPENED, CLOSED = 1767607200, 1767610800
CACHE_ENTRY = {"symbol": "EURUSD#", "direction": "long", "lots": 0.02, "entry_price": 1.1001, "entry_time": NOW,
               "risk": 20.0, "risk_amount": 20.0}


def entry_deal(commission=0.0):
    return NS(entry=OPENING, profit=0.0, commission=commission, swap=0.0, fee=0.0, time=OPENED, price=1.1001, magic=OURS)


def exit_deal(profit, commission=0.0, swap=0.0, fee=0.0):
    return NS(entry=mt5.DEAL_ENTRY_OUT, profit=profit, commission=commission, swap=swap, fee=fee, time=CLOSED,
              price=1.1013, magic=OURS)


def reconcile(*deals):
    ex, rm = make(dry_run=False, client=FakeClient(positions=[], deals={777: list(deals)}))
    rm.open_positions_cache[777] = dict(CACHE_ENTRY)
    (closed,) = ex.reconcile_open_positions_with_mt5()
    return closed


def test_pnl_is_profit_plus_commission_swap_and_fee_over_all_deals():
    closed = reconcile(entry_deal(commission=-0.5), exit_deal(3.5, commission=-0.5, swap=-0.2, fee=-0.1))
    assert closed.pnl == pytest.approx(2.2)


def test_a_breakeven_exit_that_paid_the_entry_commission_is_a_loss():
    closed = reconcile(entry_deal(commission=-0.5), exit_deal(0.0))
    assert closed.pnl == pytest.approx(-0.5)


def test_exit_price_and_time_come_from_the_closing_deal():
    closed = reconcile(entry_deal(commission=-0.5), exit_deal(3.5))
    assert closed.exit_price == 1.1013
    assert closed.exit_time == datetime.datetime.fromtimestamp(CLOSED, tz=datetime.timezone.utc)


def test_deals_without_cost_fields_still_give_the_profit():
    plain = NS(entry=mt5.DEAL_ENTRY_OUT, profit=3.5, time=CLOSED, price=1.1013, magic=OURS)
    assert reconcile(plain).pnl == 3.5


def test_a_loss_with_costs_is_counted_in_full():
    closed = reconcile(entry_deal(commission=-0.5), exit_deal(-4.0, commission=-0.5, swap=-0.3))
    assert closed.pnl == pytest.approx(-5.3)


def test_the_loss_watchdog_counts_a_closing_deal_that_only_lost_costs():
    """A breakeven exit (profit 0) that paid commission and swap on the closing deal is a loss, not a skipped zero."""
    import threading
    from src.config import Cfg
    from src.risk import RiskManager

    cfg = Cfg()
    cfg.data_source = "mt5"
    deals = [exit_deal(0.0, commission=-0.4, swap=-0.1), exit_deal(0.0, commission=-0.4, swap=-0.1)]
    deals[0].time, deals[1].time = 1, 2
    client = NS(history_deals_get=lambda since, until: deals)
    rm = RiskManager(cfg, client, threading.Lock())
    assert rm._count_consecutive_losses(NOW) == 2
