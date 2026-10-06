"""Config limits (AUDIT section D): a model parameter outside what the library accepts, or a useless value it accepts, and a stacking `meta`
that fails and silently falls back to the average, must stop the load, not show up as a crash hours into a tuning run.

The limits in `MODEL_LIMITS` were probed against the installed lightgbm, xgboost and scikit-learn (Linux and Wine Python had the same versions);
`test_the_limits_match_the_libraries` keeps them tied to what the libraries do."""
import warnings
from pathlib import Path

import numpy as np
import pytest

from src.config import Cfg, ConfigError, MODEL_LIMITS

ROOT = Path(__file__).resolve().parent.parent
NAN, INF = float("nan"), float("inf")


def load(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return Cfg.from_yaml(str(path))


def model_yaml(model, section, key, value):
    return f"models:\n- name: {model}\n  {section}:\n    {key}: {value}\n"


# (model, parameter, a value inside the limit, values outside it)
CASES = [
    ("lgbm", "n_estimators", 1, [0, -5]),
    ("lgbm", "learning_rate", 0.05, [0, -0.1, NAN, INF]),
    ("lgbm", "subsample", 1.0, [0, -0.5, 1.0001, NAN]),
    ("lgbm", "colsample_bytree", 1.0, [0, 1.5, NAN]),
    ("lgbm", "min_child_samples", 0, [-1]),
    ("xgb", "n_estimators", 1, [0]),
    ("xgb", "max_depth", 1, [0, -1]),
    ("xgb", "learning_rate", 0.1, [0, -1, NAN, INF]),
    ("xgb", "subsample", 1.0, [0, 1.0001, NAN]),
    ("xgb", "colsample_bytree", 1.0, [0, 1.0001, NAN]),
    ("rf", "n_estimators", 1, [0]),
    ("rf", "max_depth", 1, [0, -1]),
    ("rf", "min_samples_leaf", 1, [0, -1]),
    ("rf", "min_samples_leaf", 0.5, [1.0, 1.5, 0.0, NAN]),
]


def _label(value):
    return "nan" if value != value else ("inf" if value == INF else str(value))


def _yaml_value(value):
    return ".nan" if value != value else (".inf" if value == INF else str(value))


@pytest.mark.parametrize("model,key,good,bad", CASES, ids=[f"{m}.{k}" for m, k, _, _ in CASES])
def test_a_default_inside_the_limit_loads_and_outside_it_stops_the_load(tmp_path, model, key, good, bad):
    load(tmp_path, model_yaml(model, "defaults", key, _yaml_value(good)))
    for value in bad:
        with pytest.raises(ConfigError, match=rf"models\[0\]\.defaults\.{key}"):
            load(tmp_path, model_yaml(model, "defaults", key, _yaml_value(value)))


def test_a_tune_bound_outside_the_limit_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"models\[0\]\.tune\.subsample"):
        load(tmp_path, "models:\n- name: lgbm\n  tune:\n    subsample: [0.5, 1.5]\n")
    with pytest.raises(ConfigError, match=r"models\[0\]\.tune\.learning_rate"):
        load(tmp_path, "models:\n- name: lgbm\n  tune:\n    learning_rate: [0.0, 0.1]\n")
    with pytest.raises(ConfigError, match=r"models\[0\]\.tune\.n_estimators"):
        load(tmp_path, "models:\n- name: lgbm\n  tune:\n    n_estimators: [0, 100]\n")


def test_a_fixed_tune_value_is_checked_like_a_default(tmp_path):
    with pytest.raises(ConfigError, match=r"models\[0\]\.tune\.subsample"):
        load(tmp_path, "models:\n- name: lgbm\n  tune:\n    subsample: 1.5\n")


def test_nan_and_infinity_stop_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"learning_rate"):
        load(tmp_path, model_yaml("lgbm", "defaults", "learning_rate", ".nan"))
    with pytest.raises(ConfigError, match=r"subsample"):
        load(tmp_path, model_yaml("lgbm", "defaults", "subsample", ".inf"))


def test_the_message_says_the_limit(tmp_path):
    with pytest.raises(ConfigError, match=r"> 0.*got 0"):
        load(tmp_path, model_yaml("lgbm", "defaults", "learning_rate", "0"))


def test_lightgbm_takes_any_max_depth(tmp_path):
    # -1 is the default; the library accepts any whole number (0 and below mean no limit)
    load(tmp_path, model_yaml("lgbm", "defaults", "max_depth", "-1"))
    load(tmp_path, "models:\n- name: lgbm\n  tune:\n    max_depth: [-1, 10]\n")


def test_the_tracked_style_ranges_load(tmp_path):
    load(tmp_path, "models:\n- name: lgbm\n  defaults: {n_estimators: 200, learning_rate: 0.05, max_depth: -1, subsample: 0.8, colsample_bytree: 0.8}\n"
                   "  tune:\n    n_estimators: [100, 500]\n    learning_rate: [0.01, 0.1, log]\n    max_depth: [3, 10]\n"
                   "    subsample: [0.6, 1.0]\n    colsample_bytree: [0.6, 1.0]\n")


def test_the_tracked_config_still_loads():
    Cfg.from_yaml(str(ROOT / "config.yaml"))


def test_the_tuned_params_json_gets_the_limits_too(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import src.utils as utils
    monkeypatch.setattr(utils, "PARAMS_DIR", str(tmp_path))
    (tmp_path / utils.optuna_params_filename("USDJPY#")).write_text(json.dumps({"models": {"lgbm": {"subsample": 1.5}}, "features": {}}))
    with pytest.raises(ConfigError, match=r"models\.lgbm\.subsample"):
        utils.load_optuna_params("USDJPY#", SimpleNamespace(features=SimpleNamespace(min_pct_change=0.0001)))


# ---- the limits against the libraries ---------------------------------------------------------------------------------

# values a library takes but that make a useless model (a constant one, or no trees): the config refuses them anyway
ACCEPTED_BUT_USELESS = {
    ("xgb", "n_estimators", 0), ("xgb", "max_depth", 0), ("xgb", "learning_rate", 0), ("xgb", "learning_rate", INF),
    ("xgb", "subsample", 0), ("xgb", "colsample_bytree", 0), ("xgb", "subsample", NAN), ("xgb", "colsample_bytree", NAN),
    ("xgb", "learning_rate", NAN), ("lgbm", "learning_rate", INF),
}


def _fit(model, **params):
    rng = np.random.RandomState(0)
    X = rng.randn(80, 4)
    y = (X[:, 0] + 0.5 * rng.randn(80) > 0).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model == "lgbm":
            from lightgbm import LGBMClassifier
            est = LGBMClassifier(**{"n_estimators": 3, "verbose": -1, "n_jobs": 1, "random_state": 0, **params})
        elif model == "xgb":
            pytest.importorskip("xgboost")
            from xgboost import XGBClassifier
            est = XGBClassifier(**{"n_estimators": 3, "n_jobs": 1, "random_state": 0, "eval_metric": "logloss", **params})
        else:
            from sklearn.ensemble import RandomForestClassifier
            est = RandomForestClassifier(**{"n_estimators": 3, "n_jobs": 1, "random_state": 0, **params})
        est.fit(X, y)
        return est.predict_proba(X)[:, 1]


@pytest.mark.parametrize("model,key,good,bad", CASES, ids=[f"{m}.{k}" for m, k, _, _ in CASES])
def test_the_limits_match_the_libraries(model, key, good, bad):
    assert np.isfinite(_fit(model, **{key: good})).all()
    for value in bad:
        try:
            ok = np.isfinite(_fit(model, **{key: value})).all()
        except Exception:
            continue  # the library refuses it too
        assert (model, key, value) in ACCEPTED_BUT_USELESS or not ok, f"{model}.{key}={value} fits but the config refuses it, and it is not listed as useless"


def test_every_limit_has_a_case():
    covered = {(m, k) for m, k, _, _ in CASES}
    assert {(m, k) for (m, k) in MODEL_LIMITS if k != "max_depth" or m != "lgbm"} <= covered


# ---- ensemble.meta ------------------------------------------------------------------------------------------------

def test_a_valid_meta_loads(tmp_path):
    load(tmp_path, "ensemble:\n  method: stacking\n  meta:\n    type: logit\n    C: 0.5\n")


def test_a_non_positive_C_stops_the_load(tmp_path):
    # LogisticRegression raises, and Ensemble.fit swallows it and silently averages the members instead
    for bad in ("0", "-1", ".nan", ".inf"):
        with pytest.raises(ConfigError, match=r"ensemble\.meta\.C"):
            load(tmp_path, f"ensemble:\n  meta:\n    C: {bad}\n")


def test_a_string_for_C_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"ensemble\.meta\.C"):
        load(tmp_path, 'ensemble:\n  meta:\n    C: "1.0"\n')


def test_a_meta_type_other_than_logit_stops_the_load(tmp_path):
    # nothing reads `type`: anything else would still run a logistic regression
    with pytest.raises(ConfigError, match=r"ensemble\.meta\.type"):
        load(tmp_path, "ensemble:\n  meta:\n    type: xgb\n")


def test_an_unknown_meta_key_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"ensemble\.meta.*did you mean `C`|`c`.*ensemble\.meta"):
        load(tmp_path, "ensemble:\n  meta:\n    c: 1.0\n")
