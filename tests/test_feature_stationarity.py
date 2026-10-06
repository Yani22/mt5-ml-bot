"""C5 / C6: the model must not see absolute price levels (macd, momentum, atr, donchian, bollinger bands, emas, the MTA ema),
a cumulative OBV that depends on where the window starts, or values back-filled from the future at the window start. It gets
relative versions; `X` keeps the raw `atr_14` (stops and trailing read it), `macd_diff`, `adx`, `volatility_10` and
`dist_from_ema_200` (the contextual bandit reads them). `min_child_samples` defaults to 200, the value of the M5 walk-forward
harness (`scripts/walkforward.py`), not 5."""
import numpy as np
import pandas as pd
import pytest

from src import features as F
from src.config import FeatureCfg, MtaCfg, PriceActionCfg
from src.features import (add_contextual_features, add_relative_features, build_dynamic_features, build_static_features,
                          model_matrix)
from src.strategy_ml import MLStrategy


def ohlc(n=4000, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    close = 1.1 + np.cumsum(rng.normal(0, 0.0005, n))
    high = close + rng.uniform(0, 0.0004, n)
    low = close - rng.uniform(0, 0.0004, n)
    return pd.DataFrame({"open": close * scale, "high": high * scale, "low": low * scale, "close": close * scale,
                         "volume": rng.integers(100, 1000, n)}, index=idx)


def features(df):
    static = build_static_features(df, "TEST", pa_cfg=PriceActionCfg())
    return add_relative_features(build_dynamic_features(df, static, FeatureCfg(), "TEST"), df)


# ---- the model sees no absolute level -----------------------------------------------------------------------------

def test_the_model_columns_do_not_change_when_the_price_scale_changes():
    """A feature that tracks the price level (macd, atr, bands, emas, donchian ...) changes by 100x here."""
    a, b = model_matrix(features(ohlc())), model_matrix(features(ohlc(scale=100.0)))
    assert list(a.columns) == list(b.columns)
    bad = [c for c in a.columns if not np.allclose(a[c], b[c], rtol=1e-6, atol=1e-9, equal_nan=True)]
    assert not bad, f"these model columns depend on the price level: {bad}"


def test_the_model_columns_do_not_depend_on_where_the_window_starts():
    """OBV is cumulative from the first row of the window; the recursive indicators must converge. Compare the last rows of a
    long series with the same rows of a tail of it."""
    df = ohlc()
    full, tail = model_matrix(features(df)), model_matrix(features(df.iloc[2500:]))
    last = df.index[-400:]
    bad = [c for c in full.columns if not np.allclose(full.loc[last, c], tail.loc[last, c], rtol=1e-4, atol=1e-7, equal_nan=True)]   # atol: the 200-bar ema is still warming up (7e-8)
    assert not bad, f"these model columns depend on the window start: {bad}"


def test_the_raw_columns_the_stops_and_the_bandit_read_are_still_in_x_in_raw_units():
    df = ohlc()
    X = features(df)
    assert np.allclose(X["atr_14"].dropna(), (X["atr_rel"] * df["close"]).dropna())          # price units, for stops and trailing
    assert np.allclose(X["macd_diff"].dropna(), (X["macd_diff_rel"] * df["close"]).dropna())
    for col in ("adx", "volatility_10", "dist_from_ema_200"):
        assert col in X.columns


def test_the_raw_level_columns_are_not_in_the_model_matrix():
    cols = set(model_matrix(features(ohlc())).columns)
    raw = {"macd", "macd_signal", "macd_diff", "momentum_5", "momentum_10", "atr_14", "vol_ma_20", "obv", "donchian_h",
           "donchian_l", "donchian_m", "ema_fast", "ema_slow", "bb_high", "bb_low", "macd_x_adx"}
    assert not (cols & raw)
    assert {"atr_rel", "macd_rel", "obv_chg_20", "bb_pos", "donchian_h_rel", "ema_fast_dist"} <= cols


def test_the_mta_ema_reaches_the_model_only_as_a_distance_from_the_higher_timeframe_close():
    m5 = ohlc(n=600)
    h1 = m5["close"].resample("1h").last().to_frame("close")
    X = add_contextual_features(m5[["close"]].copy(), mta_df=h1, mta_cfg=MtaCfg(), relative=True)
    cols = set(model_matrix(X).columns)
    ema = [c for c in X.columns if c.startswith("mta_ema_") and "dist" not in c]
    dist = [c for c in X.columns if c.startswith("mta_ema_dist_")]
    assert ema and dist and not (cols & set(ema)) and set(dist) <= cols


def test_the_model_learns_from_the_model_columns_only():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2026-01-05", periods=1500, freq="5min")
    X = pd.DataFrame({"f1": rng.normal(size=1500), "atr_14": 0.001, "macd": rng.normal(size=1500)}, index=idx)
    y = pd.Series((rng.normal(size=1500) > 0).astype(int), index=idx)
    member = MLStrategy(model="lgbm", calibrate=False, n_estimators=10, max_depth=3)
    member.fit(X, y)
    assert list(member._pipe.feature_names_in_) == ["f1"]


# ---- no values from the future at the window start ----------------------------------------------------------------

def test_warmup_rows_stay_empty_instead_of_taking_values_from_later_bars():
    X = features(ohlc())
    assert X["volatility_20"].iloc[:19].isna().all()      # was back-filled from bar 20 before


def test_sanitize_drops_leading_rows_it_cannot_fill_without_the_future():
    idx = pd.date_range("2026-01-05", periods=10, freq="5min")
    X = pd.DataFrame({"a": [np.nan] * 3 + [1.0] * 7, "b": 1.0}, index=idx)
    member = MLStrategy(model="lgbm", calibrate=False, n_estimators=2)
    assert list(member._sanitize(X).index) == list(idx[3:])


@pytest.mark.parametrize("module", ["features", "strategy_ml", "ensemble"])
def test_no_back_fill_is_left_in_the_feature_path(module):
    import inspect
    from src import ensemble, strategy_ml
    source = inspect.getsource({"features": F, "strategy_ml": strategy_ml, "ensemble": ensemble}[module])
    assert ".bfill(" not in source


# ---- C6 -----------------------------------------------------------------------------------------------------------

def test_lightgbm_min_child_samples_defaults_to_the_walk_forward_value():
    assert MLStrategy(model="lgbm")._pipe.named_steps["clf"].min_child_samples == 200


def test_a_configured_min_child_samples_still_wins():
    assert MLStrategy(model="lgbm", min_child_samples=7)._pipe.named_steps["clf"].min_child_samples == 7


# ---- the research pipelines keep their matrix ---------------------------------------------------------------------

NEW = {"macd_rel", "macd_signal_rel", "macd_diff_rel", "momentum_5_rel", "momentum_10_rel", "atr_rel", "obv_chg_20",
       "donchian_h_rel", "donchian_l_rel", "donchian_m_rel", "ema_fast_dist", "ema_slow_dist", "bb_pos", "macd_rel_x_adx"}


def test_the_builders_the_h1_bot_and_the_research_scripts_call_add_no_relative_columns():
    df = ohlc()
    static = build_static_features(df, "TEST", pa_cfg=PriceActionCfg())
    X = build_dynamic_features(df, static, FeatureCfg(), "TEST")
    assert not (NEW & set(X.columns))
    from src.h1_features import build_h1
    assert not (NEW & set(build_h1(df).columns))
    m5 = ohlc(n=600)
    h1 = m5["close"].resample("1h").last().to_frame("close")
    plain = add_contextual_features(m5[["close"]].copy(), mta_df=h1, mta_cfg=MtaCfg())
    assert not [c for c in plain.columns if "mta_ema_dist" in c]


def test_the_tuner_and_the_main_pipeline_build_the_relative_columns():
    """tuner.py cannot be imported on Linux, so its source is checked (as in fix 9)."""
    assert "add_relative_features(build_dynamic_features(" in open("tuner.py").read()
    X = F.build_features(ohlc(n=1500), FeatureCfg(), __import__("src.config", fromlist=["Cfg"]).Cfg(), symbol="TEST")
    assert NEW <= set(X.columns)
