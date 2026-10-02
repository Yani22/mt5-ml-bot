"""Cross-asset features must not use reference bars that close after the target bar."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from cross_features import cross_features  # noqa: E402


def _series(tmp, name, n=800, seed=0, gaps=False):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    if gaps:  # an index that is closed 8 hours a day
        idx = idx[idx.hour < 16]
    c = 100 + np.cumsum(rng.normal(0, 0.1, len(idx)))
    pd.DataFrame({"close": c}, index=idx).to_csv(tmp / f"{name}_H1.csv")


def test_cross_features_do_not_use_future_reference_bars(tmp_path):
    _series(tmp_path, "AAA", seed=1)
    _series(tmp_path, "IDX", seed=2, gaps=True)
    target = pd.date_range("2024-01-01", periods=800, freq="1h", tz="UTC")  # the real data is tz-aware
    full = cross_features(target, "TGT", str(tmp_path), refs=["AAA", "IDX"])
    assert full.index.equals(target)
    assert (full.iloc[50:].nunique() > 50).all(), "cross features are constant: alignment is broken"
    for cut in range(200, 780, 37):
        # reference history truncated to what existed at the target's bar `cut`
        for name in ("AAA", "IDX"):
            df = pd.read_csv(tmp_path / f"{name}_H1.csv", index_col=0, parse_dates=True)
            df[df.index <= target[cut - 1].tz_localize(None)].to_csv(tmp_path / f"{name}_H1.csv")
        part = cross_features(target[:cut], "TGT", str(tmp_path), refs=["AAA", "IDX"])
        assert np.allclose(full.iloc[cut - 3:cut].to_numpy(), part.iloc[cut - 3:cut].to_numpy(), equal_nan=True), cut
        _series(tmp_path, "AAA", seed=1)
        _series(tmp_path, "IDX", seed=2, gaps=True)
