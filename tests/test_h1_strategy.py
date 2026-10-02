"""Parity checks for the live H1 strategy: it must make the same decisions as the walk-forward it was sized against."""
import os

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier

from src import h1_strategy as hs
from src.h1_features import WARMUP, build_h1

HIST = "data/historical_data/USDJPY_H1.csv"


def test_fold_start_boundaries():
    assert hs.fold_start(6000) == 6000
    assert hs.fold_start(6999) == 6000
    assert hs.fold_start(7000) == 7000
    assert hs.fold_start(16803) == 16000


def test_signal_side_rule():
    # same rule as walkforward_h1.simulate: shifted prob above 0.5 + offset and above the other side
    assert hs.signal_side(0.62, 0.40, 0.5, 0.5) == 1
    assert hs.signal_side(0.40, 0.62, 0.5, 0.5) == -1
    assert hs.signal_side(0.55, 0.45, 0.5, 0.5) == 0  # below 0.56
    assert hs.signal_side(0.60, 0.60, 0.5, 0.5) == 0  # tie -> no trade
    assert hs.signal_side(0.62, 0.50, 0.55, 0.45) == 1  # base rates shift: 0.57 vs 0.55, 0.57 > 0.56
    assert hs.signal_side(0.60, 0.50, 0.55, 0.45) == 0  # 0.55 is not above 0.56


def _reference_fold(h1, X, s):
    """What scripts/walkforward_h1.py does for the fold starting at row s, on the FULL history."""
    fwd = (h1["close"].shift(-hs.HORIZON) - h1["close"]).dropna()
    common = X.index.intersection(fwd.index)
    Xc, fwd = X.loc[common], fwd.loc[common]
    tr = slice(s - hs.TRAIN_ROWS, s - hs.HORIZON)
    yl, ys = (fwd > 0).astype(int), (fwd < 0).astype(int)
    ml = LGBMClassifier(**hs.LGBM_PARAMS).fit(Xc.iloc[tr], yl.iloc[tr])
    ms = LGBMClassifier(**hs.LGBM_PARAMS).fit(Xc.iloc[tr], ys.iloc[tr])
    return ml, ms, float(yl.iloc[tr].mean()), float(ys.iloc[tr].mean())


@pytest.mark.skipif(not os.path.exists(HIST), reason="needs data/historical_data/USDJPY_H1.csv")
def test_live_decision_matches_walkforward_on_truncated_history():
    h1 = pd.read_csv(HIST, index_col=0, parse_dates=True)
    X_full = build_h1(h1)
    n = len(X_full)
    last_fold = hs.fold_start(n - 1)
    rows = [last_fold, last_fold + 1, last_fold + 400, n - 1, last_fold - 1000, last_fold - 1]  # incl. fold switches
    refs = {}
    strat = hs.H1Strategy()
    for r in rows:
        s = hs.fold_start(r)
        if s not in refs:
            refs[s] = _reference_fold(h1, X_full, s)
        ml, ms, bl, bs = refs[s]
        row = X_full.iloc[[r]]
        pl, ps = ml.predict_proba(row)[0, 1], ms.predict_proba(row)[0, 1]
        # live sees ONLY bars up to and including this one (h1 position = feature row + WARMUP)
        d = strat.decide(h1.iloc[: r + WARMUP + 1])
        assert d.bar_time == X_full.index[r]
        assert np.isclose(d.p_long, pl, atol=1e-9) and np.isclose(d.p_short, ps, atol=1e-9), (r, d, pl, ps)
        assert d.side == hs.signal_side(pl, ps, bl, bs)
        assert np.isclose(d.atr, row["atr_14"].iloc[0])
        assert np.isclose(d.stop_distance, hs.STOP_ATR * d.atr)


def test_decide_refuses_short_history():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=1500, freq="1h")
    c = 150 + np.cumsum(rng.normal(0, 0.1, len(idx)))
    h1 = pd.DataFrame({"open": c, "high": c + 0.05, "low": c - 0.05, "close": c, "volume": 100}, index=idx)
    d = hs.H1Strategy().decide(h1)
    assert d.side == 0 and "history" in d.reason
