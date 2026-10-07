"""The tuner must score the matrix the bot trains on. It used to start from the full matrix built with the default feature config
and rebuild only the dynamic columns, so columns the trial's own settings would not create (the default roc lags `ret_5`, `ret_10`)
stayed in and the tuned `roc_lags` could only add columns."""
import numpy as np
import pandas as pd

import tuner
from src.config import FeatureCfg
from src.features import build_features, model_matrix
from tests.test_feature_stationarity import ohlc


def hourly(df):
    return df["close"].resample("1h").last().to_frame("close")


def test_the_trial_matrix_is_the_matrix_the_bot_builds():
    df = ohlc(n=2600)
    mta = hourly(df)
    for lags in ([1, 2, 3], [2, 5, 10, 15]):
        trial_cfg = FeatureCfg(roc_lags=lags, ema_slow=50)
        got = tuner.trial_matrix(df, trial_cfg, "TEST", mta_df=mta, inter_market_df=None)
        want = build_features(df.copy(), trial_cfg, tuner.cfg, symbol="TEST", mta_df=mta)
        assert list(got.columns) == list(want.columns)
        pd.testing.assert_frame_equal(got, want)
        ret_columns = sorted(c for c in model_matrix(got).columns if c.startswith("ret_") and c[4:].isdigit())
        assert ret_columns == sorted(f"ret_{lag}" for lag in lags)


def test_a_smaller_lag_set_really_removes_columns():
    df = ohlc(n=2600)
    wide = tuner.trial_matrix(df, FeatureCfg(roc_lags=[1, 3, 5, 10]), "TEST", None, None)
    narrow = tuner.trial_matrix(df, FeatureCfg(roc_lags=[1, 2, 3]), "TEST", None, None)
    assert "ret_10" in wide.columns and "ret_10" not in narrow.columns and "ret_5" not in narrow.columns
    assert np.isfinite(narrow["ret_2"].iloc[-1])
