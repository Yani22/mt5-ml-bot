"""The backtester's portfolio cap and its bandit reward use the money at the stop that was PLACED (`initial_sl`, as live does
since fixes 8 and 54), not `entry_equity * risk_fraction` (nominal) and not the stop as breakeven or trailing moved it."""
import inspect

import pytest

import backtester
from tests.test_backtest_costs import USDJPY, at, make_bt, open_long


def test_the_open_risk_is_the_money_at_the_placed_stop_not_the_nominal_risk():
    bt = make_bt()
    open_long(bt, 1.1000, 1.0990, 1.1020, lots=0.5)        # 10 pips x 0.5 lot = $50; nominal 10,000 x 1% = $100
    assert bt._total_open_risk() == pytest.approx(50.0)


def test_a_stop_moved_to_breakeven_keeps_its_open_risk():
    bt = make_bt()
    pos = open_long(bt, 1.1000, 1.0990, 1.1020, lots=0.5)
    pos.sl = pos.entry_price                                 # breakeven: the money at the CURRENT stop is 0
    assert bt._total_open_risk() == pytest.approx(50.0)


def test_each_position_is_valued_in_its_own_symbol_and_closed_ones_are_left_out():
    bt = make_bt(USDJPY)
    open_long(bt, 150.000, 149.850, 150.150, sym="USDJPY#", lots=1.0)   # 150 points x $0.6667 = $100
    closed = open_long(bt, 150.000, 149.850, 150.150, sym="USDJPY#", lots=1.0)
    closed.status = "closed"
    assert bt._total_open_risk() == pytest.approx(100.0)


def test_the_closed_trade_reward_uses_the_placed_stop_after_breakeven():
    bt = make_bt()
    pos = open_long(bt, 1.1000, 1.0990, 1.1020)
    pos.sl = pos.entry_price
    bt._update_positions("EURUSD#", at(1.1020))
    (trade,) = bt.risk_controller.trades
    assert trade.risk_amount == pytest.approx(100.0)         # not 0: R stays pnl / money at the placed stop
    assert trade.sl_atr_mult == pytest.approx(1.0)           # 10 pips against an ATR of 10 pips


def test_the_fill_sums_the_open_risk_through_the_helper():
    # the order is sized when it fills at the next open (`_open_pending`), no longer in the decision loop
    src = inspect.getsource(backtester.HybridBacktester._open_pending)
    assert "self._total_open_risk()" in src and "p.entry_equity * p.risk_fraction" not in src
