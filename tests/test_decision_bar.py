"""B12: the data frame passed to the decision already ends at the bar that just closed (the forming bar is dropped when
the bars are fetched), so the decision must read its LAST row, not the one before it."""
import pandas as pd
import pytest

from test_trade_gate import make_rm, make_sp


class FeatureEnsemble:
    """Signals (probability 0.9) only when the feature `f` of the row it is given is 1."""
    ensemble_cv_auc_ = 0.6

    def predict_proba(self, X):
        return pd.Series([0.9 if float(X["f"].iloc[0]) == 1.0 else 0.1], index=X.index)


def decide(sp, f, atr):
    idx = pd.date_range("2026-01-05", periods=3, freq="5min")
    X = pd.DataFrame({"f": f, "atr_14": atr}, index=idx)
    data = pd.DataFrame({"close": 1.1000}, index=idx)
    sp._make_trade_decision(data, X)


def make():
    sp = make_sp(make_rm())
    sp.ens_long = sp.ens_short = FeatureEnsemble()
    return sp


def test_a_signal_on_the_bar_that_just_closed_trades():
    sp = make()
    decide(sp, f=[0.0, 0.0, 1.0], atr=0.0010)
    assert len(sp.execution.calls) == 1


def test_a_signal_on_the_bar_before_the_last_one_does_not_trade():
    sp = make()
    decide(sp, f=[0.0, 1.0, 0.0], atr=0.0010)
    assert sp.execution.calls == []


def test_the_stop_is_sized_on_the_atr_of_the_bar_that_just_closed():
    sp = make()
    decide(sp, f=[1.0, 1.0, 1.0], atr=[0.0030, 0.0020, 0.0010])
    (call,) = sp.execution.calls
    assert abs(call["price"] - call["sl"]) == pytest.approx(2.0 * 0.0010)     # FakeRiskController: 2.0 x ATR
    assert call["atr"] == pytest.approx(0.0010)
