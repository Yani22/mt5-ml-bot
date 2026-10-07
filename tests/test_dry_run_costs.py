"""A dry-run long is entered at the ASK and closed on the bid close, so its spread is already in the entry price; a short is entered at
the bid and bought back at the ask, so it pays the spread once at the exit. The dry-run used to subtract the spread from both."""
import pytest

from test_execution_dry_run import FakeClient, make, NOW


def run(direction, entry, sl, tp, close):
    ex, rm = make(dry_run=True, client=FakeClient(), prices={"EURUSD#": close})
    ex.trade("EURUSD#", direction, 0.02, entry, sl, tp, 1000.0, 1e-5, 1.0, NOW, atr=0.0006, auc_score=0.6)
    (closed,) = ex.reconcile_open_positions_with_mt5()
    return closed


def test_a_long_pays_no_second_spread():
    closed = run("long", entry=1.1001, sl=1.0995, tp=1.1013, close=1.1013)     # 12 pips of move on 0.02 lots = 2.4
    assert closed.pnl == pytest.approx(2.4)


def test_a_short_pays_the_spread_once_at_the_exit():
    closed = run("short", entry=1.1000, sl=1.1006, tp=1.0988, close=1.0988)    # 12 pips of move, minus one pip of spread (0.2)
    assert closed.pnl == pytest.approx(2.2)


def test_the_dead_persistence_helpers_and_the_live_branch_are_gone():
    from src.execution import Execution
    for name in ("_save_open_positions_state", "_load_open_positions_state"):
        assert not hasattr(Execution, name)
    ex, rm = make(dry_run=False, client=FakeClient())
    assert not hasattr(ex, "state_file")
    assert ex.check_closed_trades({}, NOW) == []
