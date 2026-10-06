"""Config blocks that were free-form dicts (AUDIT section D): `models`, `ensemble` and `logging` are checked for keys, values and types.

The model parameters are the ones `MLStrategy` reads (`model_params.get(...)`); any other key was silently ignored."""
import ast
from pathlib import Path

import pytest

from src.config import Cfg, ConfigError, MODEL_PARAMS

ROOT = Path(__file__).resolve().parent.parent


def load(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return Cfg.from_yaml(str(path))


GOOD_MODELS = "models:\n- name: lgbm\n  defaults:\n    n_estimators: 200\n    learning_rate: 0.05\n  tune:\n    n_estimators: [100, 500]\n    learning_rate: [0.01, 0.1, log]\n"


# ---- models ---------------------------------------------------------------------------------------------------

def test_the_model_list_loads(tmp_path):
    assert [m["name"] for m in load(tmp_path, GOOD_MODELS).models] == ["lgbm"]


def test_models_must_be_a_list(tmp_path):
    with pytest.raises(ConfigError, match="models"):
        load(tmp_path, "models:\n  lgbm: {}\n")


def test_an_unknown_model_name_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"models\[0\]\.name.*lgbn|lgbn.*models\[0\]\.name"):
        load(tmp_path, "models:\n- name: lgbn\n")


def test_a_model_without_a_name_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"models\[0\].*name"):
        load(tmp_path, "models:\n- defaults: {n_estimators: 100}\n")


def test_two_models_with_one_name_stop_the_load(tmp_path):
    with pytest.raises(ConfigError, match="twice|duplicate"):
        load(tmp_path, "models:\n- name: lgbm\n- name: lgbm\n")


def test_an_unknown_key_in_a_model_entry_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"models\[0\].*default"):
        load(tmp_path, "models:\n- name: lgbm\n  default: {n_estimators: 100}\n")


def test_a_parameter_the_model_never_reads_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"unknown key `num_leaves` in `models\[0\]\.defaults`"):
        load(tmp_path, "models:\n- name: lgbm\n  defaults:\n    num_leaves: 31\n")


def test_a_misspelled_parameter_suggests_the_right_one(tmp_path):
    with pytest.raises(ConfigError, match="did you mean `learning_rate`"):
        load(tmp_path, "models:\n- name: lgbm\n  defaults:\n    learning_rat: 0.05\n")


def test_a_parameter_in_tune_is_checked_the_same_way(tmp_path):
    with pytest.raises(ConfigError, match=r"unknown key `num_leaves` in `models\[0\]\.tune`"):
        load(tmp_path, "models:\n- name: lgbm\n  tune:\n    num_leaves: [10, 50]\n")


def test_the_logreg_model_reads_no_parameter(tmp_path):
    with pytest.raises(ConfigError, match=r"unknown key `C` in `models\[0\]\.defaults`"):
        load(tmp_path, "models:\n- name: logreg\n  defaults:\n    C: 1.0\n")


def test_a_string_for_an_int_parameter_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"models\[0\]\.defaults\.n_estimators"):
        load(tmp_path, 'models:\n- name: lgbm\n  defaults:\n    n_estimators: "200"\n')


def test_a_fraction_for_an_int_parameter_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"models\[0\]\.defaults\.max_depth"):
        load(tmp_path, "models:\n- name: lgbm\n  defaults:\n    max_depth: 5.5\n")


def test_an_int_is_accepted_for_a_float_parameter_default(tmp_path):
    load(tmp_path, "models:\n- name: lgbm\n  defaults:\n    subsample: 1\n")


def test_a_tune_range_needs_two_or_three_entries(tmp_path):
    with pytest.raises(ConfigError, match=r"models\[0\]\.tune\.n_estimators"):
        load(tmp_path, "models:\n- name: lgbm\n  tune:\n    n_estimators: [100, 200, 300, 400]\n")


def test_a_three_entry_tune_range_must_end_in_log(tmp_path):
    with pytest.raises(ConfigError, match=r"models\[0\]\.tune\.learning_rate"):
        load(tmp_path, "models:\n- name: lgbm\n  tune:\n    learning_rate: [0.01, 0.1, lin]\n")


def test_a_tune_range_with_a_string_bound_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"models\[0\]\.tune\.n_estimators"):
        load(tmp_path, 'models:\n- name: lgbm\n  tune:\n    n_estimators: [100, "500"]\n')


def test_a_range_that_mixes_int_and_float_bounds_stops_the_load(tmp_path):
    # the tuner picks suggest_int or suggest_float from the type of the bounds
    with pytest.raises(ConfigError, match=r"models\[0\]\.tune\.n_estimators"):
        load(tmp_path, "models:\n- name: lgbm\n  tune:\n    n_estimators: [100, 500.0]\n")


def test_a_float_parameter_with_int_bounds_stops_the_load(tmp_path):
    # [0, 1] would make the tuner pick only 0 or 1 for a subsample
    with pytest.raises(ConfigError, match=r"models\[0\]\.tune\.subsample"):
        load(tmp_path, "models:\n- name: lgbm\n  tune:\n    subsample: [0, 1]\n")


def test_a_range_with_low_above_high_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"models\[0\]\.tune\.max_depth"):
        load(tmp_path, "models:\n- name: lgbm\n  tune:\n    max_depth: [10, 3]\n")


def test_every_parameter_a_model_reads_is_in_the_table():
    """MODEL_PARAMS must list exactly the keys `MLStrategy` takes from its parameters (plus `device`)."""
    tree = ast.parse((ROOT / "src" / "strategy_ml.py").read_text(encoding="utf-8"))
    read = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", None) in ("get", "pop")
                and getattr(node.func.value, "id", None) == "model_params" and isinstance(node.args[0], ast.Constant)):
            read.add(node.args[0].value)
    listed = {name for params in MODEL_PARAMS.values() for name in params} | {"device"}
    assert read == listed


# ---- ensemble ---------------------------------------------------------------------------------------------------

def test_the_ensemble_block_of_the_tracked_style_loads(tmp_path):
    load(tmp_path, "ensemble:\n  method: soft_vote\n  threshold_metric: sharpe_ratio\n  threshold_grid: auto\n"
                   "  min_ensemble_auc: 0.55\n  auto_threshold: true\n")


def test_an_empty_ensemble_block_is_an_empty_dict(tmp_path):
    assert load(tmp_path, "ensemble:\n").ensemble == {}


def test_an_unknown_ensemble_key_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match="ensemble.*threshold_metrik|threshold_metrik.*ensemble"):
        load(tmp_path, "ensemble:\n  threshold_metrik: f1\n")


def test_an_unknown_ensemble_method_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"ensemble\.method"):
        load(tmp_path, "ensemble:\n  method: softvote\n")


def test_an_unknown_threshold_metric_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"ensemble\.threshold_metric"):
        load(tmp_path, "ensemble:\n  threshold_metric: sharpe\n")


def test_a_quoted_false_for_a_flag_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"ensemble\.flat_mode"):
        load(tmp_path, 'ensemble:\n  flat_mode: "false"\n')


def test_a_threshold_grid_is_auto_a_number_or_a_list_of_numbers(tmp_path):
    load(tmp_path, "ensemble:\n  threshold_grid: [0.5, 0.55]\n")
    load(tmp_path, "ensemble:\n  threshold_grid: 0.55\n")
    with pytest.raises(ConfigError, match=r"ensemble\.threshold_grid"):
        load(tmp_path, "ensemble:\n  threshold_grid: automatic\n")
    with pytest.raises(ConfigError, match=r"ensemble\.threshold_grid"):
        load(tmp_path, 'ensemble:\n  threshold_grid: [0.5, "0.55"]\n')


def test_the_weights_must_name_models_and_be_numbers(tmp_path):
    load(tmp_path, GOOD_MODELS + "ensemble:\n  weights:\n    lgbm: 1.0\n")
    with pytest.raises(ConfigError, match=r"ensemble\.weights.*xgb"):
        load(tmp_path, GOOD_MODELS + "ensemble:\n  weights:\n    xgb: 1.0\n")
    with pytest.raises(ConfigError, match=r"ensemble\.weights\.lgbm"):
        load(tmp_path, GOOD_MODELS + 'ensemble:\n  weights:\n    lgbm: "1.0"\n')


# ---- logging ----------------------------------------------------------------------------------------------------

def test_the_logging_block_loads(tmp_path):
    load(tmp_path, "logging:\n  level: INFO\n  to_file: false\n  rotate: 10 MB\n  retention: 7 days\n")


def test_an_unknown_logging_key_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match="logging.*to_files|to_files.*logging"):
        load(tmp_path, "logging:\n  to_files: true\n")


def test_a_quoted_false_for_to_file_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"logging\.to_file"):
        load(tmp_path, 'logging:\n  to_file: "false"\n')


def test_an_unknown_log_level_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"logging\.level"):
        load(tmp_path, "logging:\n  level: LOUD\n")


# ---- guard ------------------------------------------------------------------------------------------------------

def test_the_tracked_config_still_loads():
    Cfg.from_yaml(str(ROOT / "config.yaml"))


# ---- ranges the tuner would turn into a file the bot then refuses --------------------------------------------------

def test_a_log_range_for_an_int_parameter_stops_the_load(tmp_path):
    # tuner.suggest_params draws a decimal for every [lo, hi, log] range, so n_estimators would be saved as 287.3
    with pytest.raises(ConfigError, match=r"models\[0\]\.tune\.n_estimators.*log|log.*models\[0\]\.tune\.n_estimators"):
        load(tmp_path, "models:\n- name: lgbm\n  tune:\n    n_estimators: [100, 500, log]\n")


def test_a_feature_range_with_a_float_bound_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"features\.rsi_period"):
        load(tmp_path, "features:\n  rsi_period: [14, 20.0]\n")


def test_a_feature_range_with_a_string_bound_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"features\.rsi_period"):
        load(tmp_path, 'features:\n  rsi_period: [14, "20"]\n')


def test_a_feature_range_with_four_entries_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"features\.ema_slow"):
        load(tmp_path, "features:\n  ema_slow: [100, 150, 200, 250]\n")


def test_the_tracked_style_feature_ranges_load(tmp_path):
    cfg = load(tmp_path, "features:\n  rsi_period: [14, 14]\n  ema_slow: [200, 200]\n  roc_lags: [1, 3, 5]\n")
    assert (cfg.features.rsi_period, cfg.features.ema_slow, cfg.features.roc_lags) == (14, 200, [1, 3, 5])


def test_a_lower_case_log_level_stops_the_load(tmp_path):
    # the logger (loguru) knows only upper-case level names and raises on `info` at start-up
    with pytest.raises(ConfigError, match=r"logging\.level"):
        load(tmp_path, "logging:\n  level: info\n")
