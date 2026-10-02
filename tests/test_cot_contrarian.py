"""Look-ahead and direction checks for scripts/cot_contrarian.py (synthetic data only)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import cot_contrarian as cc  # noqa: E402


def test_percentile_ignores_the_future():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=400).cumsum(), index=pd.date_range("2012-01-03", periods=400, freq="7D"))
    full = cc.percentile(s)
    for cut in range(150, 400, 47):
        part = cc.percentile(s.iloc[:cut])
        assert np.allclose(full.iloc[:cut].to_numpy(), part.to_numpy(), equal_nan=True)


def test_entry_is_after_friday_release():
    tuesday = pd.Timestamp("2024-03-05")
    assert tuesday.weekday() == 1
    e = cc.entry_date(tuesday)
    assert e.weekday() == 0 and e > tuesday + pd.Timedelta(days=3)  # Monday after the Friday publication


def test_contrarian_direction_and_inverted_pairs():
    assert cc.pair_side(0.95, 1) == -1       # crowded long AUD -> sell AUDUSD
    assert cc.pair_side(0.05, 1) == 1        # crowded short AUD -> buy AUDUSD
    assert cc.pair_side(0.95, -1) == 1       # crowded long JPY -> sell JPY = buy USDJPY
    assert cc.pair_side(0.50, 1) == 0
    assert cc.pair_side(0.95, 1, contrarian=False) == 1
