"""Look-ahead check for the experimental walk-forward features (scripts/extra_features.py)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from extra_features import extra_features  # noqa: E402


def _bars(n_m5=6000, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_m5, freq="5min")
    close = 1.1 + np.cumsum(rng.normal(0, 0.0004, n_m5))
    high = close + rng.uniform(0, 0.0003, n_m5)
    low = close - rng.uniform(0, 0.0003, n_m5)
    m5 = pd.DataFrame({"open": close, "high": high, "low": low, "close": close,
                       "volume": rng.integers(100, 900, n_m5)}, index=idx)
    h1 = m5.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                "volume": "sum"}).dropna()
    return m5, h1


def test_extra_features_do_not_use_future_bars():
    m5, h1 = _bars()
    full = extra_features(m5, h1)
    bad = set()
    for cut in range(2200, 5800, 97):
        m5_cut = m5.iloc[:cut]
        t_end = m5_cut.index[-1] + pd.Timedelta("5min")
        h1_cut = h1[h1.index + pd.Timedelta("1h") <= t_end]  # only fully closed H1 bars existed at that time
        part = extra_features(m5_cut, h1_cut)
        tail = slice(cut - 3, cut)
        for col in full.columns:
            if not np.allclose(full[col].iloc[tail], part[col].iloc[tail], equal_nan=True):
                bad.add(col)
    assert not bad, f"extra features change when future bars are removed (look-ahead): {sorted(bad)}"
