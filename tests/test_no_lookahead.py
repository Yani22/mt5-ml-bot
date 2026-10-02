"""Look-ahead leakage checks: a feature at bar t must not depend on bars after t."""
import numpy as np
import pandas as pd

from src.config import FeatureCfg, MtaCfg, PriceActionCfg
from src.features import add_contextual_features, build_dynamic_features, build_static_features
from src.labels import generate_long_short_labels


def _ohlc(n=1500, freq="5min", seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    close = 1.1 + np.cumsum(rng.normal(0, 0.0005, n))
    high = close + rng.uniform(0, 0.0004, n)
    low = close - rng.uniform(0, 0.0004, n)
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close,
                         "volume": rng.integers(100, 1000, n)}, index=idx)


def _features(df):
    static = build_static_features(df, "TEST", pa_cfg=PriceActionCfg())
    return build_dynamic_features(df, static, FeatureCfg(), "TEST")


def test_features_do_not_use_future_bars():
    df = _ohlc()
    full = _features(df)
    bad = set()
    # A 1-2 bar leak only shows in the last rows before a cut, so try many cuts.
    for cut in range(400, 1400, 13):
        part = _features(df.iloc[:cut])
        tail = slice(cut - 3, cut)
        bad |= {c for c in full.columns
                if not np.allclose(full[c].iloc[tail], part[c].iloc[tail], equal_nan=True)}
    assert not bad, f"features change when future bars are removed (look-ahead): {sorted(bad)}"


def test_mta_feature_ignores_unclosed_higher_timeframe_bar():
    m5 = _ohlc(n=600)
    # H1 indexed by bar OPEN time (MT5 convention); its close is only known at open + 1h.
    h1 = m5["close"].resample("1h").last().to_frame("close")
    cfg = MtaCfg()
    base = add_contextual_features(m5[["close"]].copy(), mta_df=h1, mta_cfg=cfg)

    t = pd.Timestamp("2024-01-02 10:00")
    h1_changed = h1.copy()
    h1_changed.loc[t, "close"] += 0.05  # this bar is still forming at 10:00-10:55
    changed = add_contextual_features(m5[["close"]].copy(), mta_df=h1_changed, mta_cfg=cfg)

    window = slice(t, t + pd.Timedelta(minutes=55))
    cols = [c for c in base.columns if c.startswith("mta_")]
    assert base.loc[window, cols].equals(changed.loc[window, cols]), \
        "M5 bars inside an H1 hour see that hour's (unfinished) close"


def test_labels_drop_unknown_tail():
    df = _ohlc(n=100)
    h = 12
    y_long, y_short = generate_long_short_labels(df, h, 0.0)
    # The last h rows have no known future; they must be dropped, not labelled 0.
    assert len(y_long) == len(y_short) == len(df) - h
    assert y_long.index[-1] == df.index[-h - 1]
    assert not y_long.isna().any()
