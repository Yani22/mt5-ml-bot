"""The tuned-params JSON (AUDIT section D) goes through the same key and type checks as config.yaml: `load_optuna_params` used to hand
back whatever the file held, and `FeatureCfg(**features)` or the model code then ignored it or crashed with a TypeError."""
import json
from types import SimpleNamespace as NS

import pytest

import src.utils as utils
from src.config import ConfigError

CFG = NS(features=NS(min_pct_change=0.0002))


@pytest.fixture
def write(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "PARAMS_DIR", str(tmp_path))

    def _write(params):
        (tmp_path / utils.optuna_params_filename("USDJPY#")).write_text(json.dumps(params), encoding="utf-8")
        return tmp_path / utils.optuna_params_filename("USDJPY#")
    return _write


def load():
    return utils.load_optuna_params("USDJPY#", CFG)


def test_a_misspelled_feature_key_stops_the_load_and_names_the_file(write):
    path = write({"models": {}, "features": {"rsi_perod": 14}})
    with pytest.raises(ConfigError, match=r"rsi_perod.*did you mean `rsi_period`|rsi_period.*rsi_perod") as e:
        load()
    assert str(path) in str(e.value)


def test_a_string_for_an_int_feature_stops_the_load(write):
    write({"models": {}, "features": {"rsi_period": "14"}})
    with pytest.raises(ConfigError, match=r"features\.rsi_period"):
        load()


def test_a_float_for_an_int_feature_stops_the_load(write):
    write({"models": {}, "features": {"ema_slow": 200.5}})
    with pytest.raises(ConfigError, match=r"features\.ema_slow"):
        load()


def test_a_roc_lags_list_loads(write):
    write({"models": {}, "features": {"roc_lags": [1, 3, 5]}})
    assert load()["features"]["roc_lags"] == [1, 3, 5]


def test_an_unknown_model_in_the_file_stops_the_load(write):
    write({"models": {"lgbn": {"n_estimators": 100}}, "features": {}})
    with pytest.raises(ConfigError, match="lgbn"):
        load()


def test_a_parameter_the_model_never_reads_stops_the_load(write):
    write({"models": {"lgbm": {"num_leaves": 31}}, "features": {}})
    with pytest.raises(ConfigError, match="num_leaves"):
        load()


def test_a_string_for_a_model_parameter_stops_the_load(write):
    write({"models": {"lgbm": {"n_estimators": "200"}}, "features": {}})
    with pytest.raises(ConfigError, match=r"models\.lgbm\.n_estimators"):
        load()


def test_models_must_be_a_mapping_of_mappings(write):
    write({"models": [{"name": "lgbm"}], "features": {}})
    with pytest.raises(ConfigError, match="models"):
        load()


def test_a_tuned_label_must_be_numbers(write):
    write({"models": {}, "features": {}, "prediction_horizon": "9"})
    with pytest.raises(ConfigError, match="prediction_horizon"):
        load()


def test_an_unknown_top_level_key_stops_the_load(write):
    write({"models": {}, "features": {}, "prediction_horizn": 9})
    with pytest.raises(ConfigError, match=r"prediction_horizn.*did you mean `prediction_horizon`"):
        load()


def test_what_the_tuner_writes_loads_back(write):
    import tuner
    flat = {"feature_rsi_period": 14, "feature_ema_slow": 200, "feature_roc_lags": [2, 5, 10, 15],
            "model_lgbm_n_estimators": 321, "model_lgbm_learning_rate": 0.0123, "model_lgbm_max_depth": 6,
            "model_lgbm_subsample": 0.7, "model_lgbm_colsample_bytree": 0.9}
    saved = utils.save_optuna_params("USDJPY#", tuner.structure_best_params(flat))
    assert saved.endswith("USDJPY_best_params.json")
    loaded = load()
    assert loaded["models"]["lgbm"]["n_estimators"] == 321
    assert loaded["features"]["roc_lags"] == [2, 5, 10, 15]


def test_a_file_with_the_legacy_label_keys_still_loads(write):
    write({"models": {"lgbm": {"n_estimators": 100}}, "features": {"min_pct_change": 0.0001}, "prediction_horizon": 9, "min_pct_change": 0.0001})
    assert load()["prediction_horizon"] == 9


# ---- the real tuner path: ranges from the tracked config -> a trial -> the file -> the loader -----------------------------

def _tuner_file(ranges_models, ranges_features):
    import optuna
    import tuner
    trial = optuna.create_study().ask()
    flat = {}
    for model in ranges_models:
        for k, v in tuner.suggest_params(trial, f"model_{model['name']}", model.get("tune", {})).items():
            flat[f"model_{model['name']}_{k}"] = v
    for k, v in tuner.suggest_params(trial, "feature", ranges_features).items():
        flat[f"feature_{k}"] = v
    return utils.save_optuna_params("USDJPY#", tuner.structure_best_params(flat))


def test_a_trial_drawn_from_the_tracked_ranges_gives_a_file_the_bot_loads(write):
    import yaml
    from pathlib import Path
    raw = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text(encoding="utf-8"))
    _tuner_file(raw["models"], raw["features"])
    assert load()["models"]["lgbm"]["n_estimators"] >= 100


def test_an_int_log_range_gives_a_file_the_bot_refuses(write):
    # why the config check refuses [lo, hi, log] for an int setting: the tuner draws a decimal and the loader rejects it
    _tuner_file([{"name": "lgbm", "tune": {"n_estimators": [100, 500, "log"]}}], {})
    with pytest.raises(ConfigError, match=r"models\.lgbm\.n_estimators"):
        load()
