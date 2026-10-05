"""K5: when both models pass their threshold and AUC gate on the same bar the signals conflict (the labels are mutually
exclusive: forward return above +x versus below -x), so the bar is skipped instead of always taking the long."""
import pytest

from src.decision import choose_direction
from test_trade_gate import FakeEnsemble, decide, make_rm, make_sp


def choose(prob_long=0.9, prob_short=0.1, auc_long=0.6, auc_short=0.6, min_long=0.55, min_short=0.55, min_auc=0.55):
    return choose_direction(prob_long, prob_short, min_long, min_short, auc_long, auc_short, min_auc)


def test_only_long_passing_gives_long_with_its_auc():
    assert choose(auc_long=0.61) == ("long", 0.61, False)


def test_only_short_passing_gives_short_with_its_auc():
    assert choose(prob_long=0.1, prob_short=0.9, auc_short=0.62) == ("short", 0.62, False)


def test_neither_passing_gives_no_direction():
    assert choose(prob_long=0.1, prob_short=0.1) == (None, 0.5, False)


def test_both_passing_is_a_conflict_and_gives_no_direction():
    assert choose(prob_long=0.9, prob_short=0.8) == (None, 0.5, True)


def test_a_side_that_fails_its_auc_gate_is_not_part_of_a_conflict():
    assert choose(prob_long=0.9, prob_short=0.8, auc_long=0.50) == ("short", 0.6, False)
    assert choose(prob_long=0.9, prob_short=0.8, auc_short=0.50) == ("long", 0.6, False)


def test_the_thresholds_are_inclusive():
    assert choose(prob_long=0.55, prob_short=0.1)[0] == "long"


# ---- wired into the live decision --------------------------------------------------------------------------------

def test_the_live_decision_sends_no_order_on_a_conflict():
    sp = make_sp(make_rm())
    sp.ens_long, sp.ens_short = FakeEnsemble(prob=0.9), FakeEnsemble(prob=0.9)
    decide(sp)
    assert sp.execution.calls == []


def test_the_live_decision_trades_short_when_only_short_passes():
    sp = make_sp(make_rm())
    sp.ens_long, sp.ens_short = FakeEnsemble(prob=0.1), FakeEnsemble(prob=0.9)
    decide(sp)
    assert [c["direction"] for c in sp.execution.calls] == ["short"]


def test_the_live_decision_trades_short_when_long_fails_its_auc_gate():
    sp = make_sp(make_rm())
    sp.ens_long, sp.ens_short = FakeEnsemble(prob=0.9, auc=0.50), FakeEnsemble(prob=0.9)
    decide(sp)
    assert [c["direction"] for c in sp.execution.calls] == ["short"]


def test_the_live_decision_still_trades_long_when_only_long_passes():
    sp = make_sp(make_rm())
    decide(sp)       # make_sp: long signals, short does not
    assert [c["direction"] for c in sp.execution.calls] == ["long"]
