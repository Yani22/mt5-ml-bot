import numpy as np
import pandas as pd
import pytest

from src.backtest_models import WalkForwardModels


class FakeEns:
    def __init__(self, rows):
        self.rows, self.ensemble_cv_auc_ = rows, 0.6

    def predict_proba(self, X):
        return pd.Series(0.5 + 0.0 * X["f"].to_numpy(), index=X.index)


def frame(n=100):
    idx = pd.date_range("2026-01-05", periods=n, freq="5min", tz="UTC")
    X = pd.DataFrame({"f": np.arange(n, dtype=float)}, index=idx)
    return X, pd.Series(1, index=idx), pd.Series(0, index=idx), pd.Series(1.0, index=idx)


def schedule(fits, train_bars=40, every=20, horizon=12, n=100, X=None):
    X0, yl, ys, px = frame(n)

    def fit_fn(Xt, yt, pt, side, prev):
        fits.append((side, Xt["f"].iloc[0], Xt["f"].iloc[-1]))
        return FakeEns(len(Xt))

    return WalkForwardModels(X0 if X is None else X, yl, ys, px, train_bars, every, horizon, fit_fn)


def test_nothing_is_decided_before_the_first_window_and_the_first_fit_is_lazy():
    fits = []
    wf = schedule(fits)
    assert wf.start == 40 and fits == []
    with pytest.raises(ValueError):
        wf.probs(39)


def test_the_training_rows_end_horizon_rows_before_the_retrain_bar():
    fits = []
    wf = schedule(fits)
    wf.probs(40)
    # row 40 decides; the last row whose label is known is 40 - 12 = 28; the first is 40 - 40 = 0
    assert ("long", 0.0, 28.0) in fits and ("short", 0.0, 28.0) in fits
    assert wf.fits[0] == (40, 0, 29)


def test_a_retrain_happens_every_block_and_only_then():
    fits = []
    wf = schedule(fits)
    for i in range(40, 100):
        wf.probs(i)
    assert [f[0] for f in wf.fits] == [40, 60, 80]
    assert len(fits) == 6


def test_a_block_is_cached_and_the_next_block_fits_long_and_short_once():
    fits = []
    wf = schedule(fits)
    wf.probs(59)
    n_before = len(fits)
    wf.probs(59)
    assert len(fits) == n_before
    wf.probs(60)
    assert len(fits) == n_before + 2


def test_changing_the_future_does_not_change_the_first_fit():
    seen, seen2 = [], []
    X, yl, ys, px = frame()

    def make(store):
        def fit_fn(Xt, yt, pt, side, prev):
            store.append(Xt["f"].to_numpy().copy())
            return FakeEns(len(Xt))
        return fit_fn

    WalkForwardModels(X, yl, ys, px, 40, 20, 12, make(seen)).probs(40)
    X2 = X.copy()
    X2.iloc[29:, 0] = -1.0                      # everything after the last known label row
    WalkForwardModels(X2, yl, ys, px, 40, 20, 12, make(seen2)).probs(40)
    assert all(np.array_equal(p, q) for p, q in zip(seen, seen2)) and len(seen) == 2


def test_a_training_window_too_small_to_fit_is_an_error_not_a_silent_constant_model():
    from src.backtest_models import make_fit_fn
    X, yl, _, px = frame(300)
    with pytest.raises(ValueError, match="train_bars"):
        make_fit_fn(None, "USDJPY#", None)(X, yl, px, "long", None)
