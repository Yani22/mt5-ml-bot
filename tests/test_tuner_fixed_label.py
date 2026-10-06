"""C3: the tuner compared AUCs across label definitions (horizon 3-15 and a threshold per trial), so it picked the label that is
easiest to predict, not one worth trading. It also kept adding trials to an old study of the same name after the features, the
label or the search ranges changed, and its time-series split had no purge gap. Now the label comes from the config, only the
features and the model parameters are tuned, the study name carries a signature of everything a trial's score depends on, and the
split leaves the label horizon as a gap."""
import functools

import numpy as np
import optuna
import pandas as pd
import pytest

import tuner
from src.config import FeatureCfg, PriceActionCfg
from src.features import build_static_features
from tests.test_feature_stationarity import ohlc


@pytest.fixture
def small_tuner(monkeypatch):
    """The real objective on a small synthetic series: one cheap model, a few features ranges."""
    models = [{"name": "lgbm", "defaults": {"n_estimators": 10},
               "tune": {"n_estimators": [5, 10], "learning_rate": [0.05, 0.1, "log"], "max_depth": [3, 4]}}]
    monkeypatch.setitem(tuner.yaml_cfg, "models", models)
    monkeypatch.setitem(tuner.yaml_cfg, "features", {"rsi_period": [14, 14], "ema_slow": [50, 50]})
    monkeypatch.setitem(tuner.yaml_cfg, "cv_samples_per_split", 400)
    monkeypatch.setattr(tuner.cfg, "models", models)
    monkeypatch.setattr(tuner.cfg, "cv_samples_per_split", 400)
    monkeypatch.setattr(tuner.cfg, "prediction_horizon", 12)
    monkeypatch.setattr(tuner.cfg.features, "min_pct_change", 0.0001)
    df = ohlc(n=2600)
    static = build_static_features(df, "TEST", pa_cfg=PriceActionCfg())
    return functools.partial(tuner.objective, df=df, static_features=static, symbol="TEST")


def run_one(objective):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1)
    return study.trials[0]


def test_the_label_is_not_a_tuned_parameter(small_tuner):
    trial = run_one(small_tuner)
    assert trial.value is not None and np.isfinite(trial.value), "the trial failed"
    assert "prediction_horizon" not in trial.params and "min_pct_change" not in trial.params


def test_the_trial_uses_the_configured_label(small_tuner, monkeypatch):
    seen = []
    real = tuner.generate_labels
    monkeypatch.setattr(tuner, "generate_labels", lambda df, horizon, pct: seen.append((horizon, pct)) or real(df, horizon, pct))
    run_one(small_tuner)
    assert seen == [(12, 0.0001)]


def test_the_split_leaves_the_label_horizon_as_a_gap(small_tuner, monkeypatch):
    gaps = []

    class Recording(tuner.TimeSeriesSplit):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            gaps.append(self.gap)
    monkeypatch.setattr(tuner, "TimeSeriesSplit", Recording)
    run_one(small_tuner)
    assert gaps and set(gaps) == {12}


def test_a_validation_block_with_a_warmup_row_does_not_fail_the_trial():
    """A fitted model drops rows it cannot score (warmup NaN), so its output can be shorter than the labels."""
    idx = pd.date_range("2026-01-05", periods=6, freq="5min")
    y = pd.Series([0, 1, 0, 1, 0, 1], index=idx)
    p = pd.Series([0.2, 0.7, 0.3, 0.6, 0.4], index=idx[1:])
    assert 0.0 <= tuner.fold_auc(y, p) <= 1.0


# ---- the study name -----------------------------------------------------------------------------------------------

BASE = dict(label=(12, 0.0001), feature_ranges={"rsi_period": [14, 14]}, models=[{"name": "lgbm", "tune": {"max_depth": [3, 10]}}],
            columns=["a", "b"], cv_samples=300)


def sig(**over):
    return tuner.study_signature(**{**BASE, **over})


def test_the_signature_is_stable_for_the_same_inputs():
    assert sig() == sig()


@pytest.mark.parametrize("change", [
    {"label": (6, 0.0001)}, {"label": (12, 0.0002)},
    {"feature_ranges": {"rsi_period": [10, 20]}},
    {"models": [{"name": "lgbm", "tune": {"max_depth": [3, 12]}}]},
    {"models": [{"name": "lgbm", "tune": {"max_depth": [3, 10]}, "defaults": {"min_child_samples": 50}}]},
    {"columns": ["a", "c"]}, {"cv_samples": 600},
])
def test_the_signature_changes_with_anything_a_trial_score_depends_on(change):
    assert sig(**change) != sig()


def test_the_study_name_carries_the_signature():
    name = tuner.study_name_for("USDJPY#", 45000, sig())
    assert "USDJPY_" in name and sig() in name and "45000" in name
    assert tuner.study_name_for("USDJPY#", 45000, sig(columns=["x"])) != name


def test_the_run_uses_a_signed_study_name():
    """`run_tuning_for_symbol` needs the data pipeline, so its source is checked."""
    import inspect
    source = inspect.getsource(tuner.run_tuning_for_symbol)
    assert "study_name_for(" in source and "feature_model_tuning_{sym" not in source


# ---- the saved file -----------------------------------------------------------------------------------------------

def test_the_saved_params_do_not_carry_the_label():
    flat = {"feature_rsi_period": 14, "feature_ema_slow": 200, "model_lgbm_max_depth": 5, "model_lgbm_learning_rate": 0.05}
    saved = tuner.structure_best_params(flat)
    assert saved == {"features": {"rsi_period": 14, "ema_slow": 200},
                     "models": {"lgbm": {"max_depth": 5, "learning_rate": 0.05}}}
    assert "prediction_horizon" not in saved and "min_pct_change" not in saved
