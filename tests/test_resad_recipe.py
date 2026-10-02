"""Look-ahead and fill checks for scripts/resad_recipe.py (synthetic data only)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import resad_recipe as rr  # noqa: E402


def _bars(n=400, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 0.1, n))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + 0.05
    l_ = np.minimum(o, c) - 0.05
    return o, h, l_, c, np.full(n, 0.02), np.full(n, 0.3)


def test_barrier_label_uses_only_window_up_to_cap():
    o, h, l_, c, sp, atr = _bars()
    cap = 20
    yl, ys = rr.barrier_labels(o, h, l_, c, sp, atr, cap)
    n = len(c)
    assert np.isnan(yl[n - cap - 1:]).all() and np.isnan(ys[n - cap - 1:]).all()
    for cut in range(100, 380, 37):
        # labels at t <= cut - cap - 2 only need bars up to t + 1 + cap < cut
        pl, ps = rr.barrier_labels(o[:cut], h[:cut], l_[:cut], c[:cut], sp[:cut], atr[:cut], cap)
        k = cut - cap - 1
        assert np.array_equal(yl[:k], pl[:k]) and np.array_equal(ys[:k], ps[:k])


def test_dir_label_drops_last_rows():
    c = np.arange(50, dtype=float)
    yl, ys = rr.dir_labels(c, 12)
    assert np.isnan(yl[-12:]).all() and (yl[:-12] == 1).all() and (ys[:-12] == 0).all()


def test_same_bar_tie_is_a_loss():
    o, h, l_, sp = np.array([10.0, 10.0]), np.array([10.0, 12.0]), np.array([10.0, 8.0]), np.zeros(2)
    assert rr.first_touch(1, 10.0, 11.0, 9.0, o, h, l_, sp, 1, 5)[2] == "sl"
    assert rr.first_touch(-1, 10.0, 9.0, 11.0, o, h, l_, sp, 1, 5)[2] == "sl"


def test_short_stop_triggers_on_ask():
    # bid high 10.9 never reaches the 11.0 stop, but ask = bid + 0.2 does
    o, h, l_, sp = np.array([10.0, 10.0]), np.array([10.0, 10.9]), np.array([10.0, 9.95]), np.full(2, 0.2)
    exit_px, held, outcome = rr.first_touch(-1, 10.0, 9.0, 11.0, o, h, l_, sp, 1, 5)
    assert outcome == "sl" and exit_px == 11.0


def test_cap_exit_at_open_after_cap():
    o = np.array([10.0, 10.0, 10.1, 10.2, 10.3])
    h, l_, sp = o + 0.01, o - 0.01, np.zeros(5)
    assert rr.first_touch(1, 10.0, 12.0, 8.0, o, h, l_, sp, 1, 3) == (10.3, 3, "cap")


def test_ema_cross_signal_uses_only_past_bars():
    import pandas as pd
    import resad_trend as rt
    o, h, l_, c, sp, atr = _bars(600, seed=3)
    s = pd.Series(c, index=pd.date_range("2020-01-01", periods=600, freq="h"))
    full = rt.cross_signals(s)
    assert (full != 0).sum() > 3
    for cut in range(120, 600, 53):
        assert np.array_equal(full[:cut], rt.cross_signals(s.iloc[:cut]))
