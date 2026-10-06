"""Tuned params were a pickle that `load_optuna_params` unpickled with no check, while the model folders get an HMAC signature
check before anything is unpickled. A swapped or downloaded `<symbol>_best_params.pkl` could run code at start-up. The params are
plain data, so they are JSON now and nothing in `optuna_params/` is ever unpickled; a leftover `.pkl` stops start-up with a
message that says what to do."""
import json
import pickle
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

import src.utils as utils
from src.config import ConfigError

CFG = NS(features=NS(min_pct_change=0.0002))
PARAMS = {"models": {"lgbm": {"n_estimators": 123, "learning_rate": 0.05}}, "features": {"roc_lags": [1, 3, 5], "min_pct_change": 0.0001},
          "prediction_horizon": 9, "min_pct_change": 0.0001}


@pytest.fixture
def params_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "PARAMS_DIR", str(tmp_path))
    return tmp_path


class Boom:
    """Unpickling this creates a marker file: proof that a pickle was (or was not) loaded."""
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (Path(self.marker).write_text, ("executed",))


def test_a_saved_dict_loads_back_unchanged(params_dir):
    utils.save_optuna_params("USDJPY#", PARAMS)
    assert utils.load_optuna_params("USDJPY#", CFG) == PARAMS


def test_numpy_values_from_a_study_are_saved_as_plain_numbers(params_dir):
    utils.save_optuna_params("USDJPY#", {"models": {"lgbm": {"n_estimators": np.int64(200), "learning_rate": np.float64(0.1)}}})
    raw = json.loads((params_dir / utils.optuna_params_filename("USDJPY#")).read_text())
    assert raw["models"]["lgbm"] == {"n_estimators": 200, "learning_rate": 0.1}


def test_a_leftover_pickle_is_never_unpickled_and_stops_the_load(params_dir):
    marker = params_dir / "executed.txt"
    with open(params_dir / "USDJPY_best_params.pkl", "wb") as f:
        pickle.dump(Boom(str(marker)), f)
    with pytest.raises(ConfigError, match="USDJPY_best_params.pkl"):
        utils.load_optuna_params("USDJPY#", CFG)
    assert not marker.exists()


def test_with_both_files_the_json_wins_and_the_pickle_is_left_alone(params_dir):
    marker = params_dir / "executed.txt"
    with open(params_dir / "USDJPY_best_params.pkl", "wb") as f:
        pickle.dump(Boom(str(marker)), f)
    utils.save_optuna_params("USDJPY#", PARAMS)
    assert utils.load_optuna_params("USDJPY#", CFG) == PARAMS
    assert not marker.exists()


def test_unreadable_json_stops_the_load_instead_of_falling_back_to_defaults(params_dir):
    (params_dir / utils.optuna_params_filename("USDJPY#")).write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="USDJPY_best_params.json"):
        utils.load_optuna_params("USDJPY#", CFG)


def test_no_file_still_means_the_config_defaults(params_dir):
    assert utils.load_optuna_params("USDJPY#", CFG) is None


def test_json_without_models_still_gives_the_empty_model_params(params_dir):
    (params_dir / utils.optuna_params_filename("USDJPY#")).write_text('{"something": 1}', encoding="utf-8")
    loaded = utils.load_optuna_params("USDJPY#", CFG)
    assert loaded["features"] == {"min_pct_change": 0.0002} and loaded["lgbm"] == {}


def test_utils_no_longer_imports_or_calls_pickle():
    src = Path(utils.__file__).read_text(encoding="utf-8")
    assert "import pickle" not in src and "pickle.load" not in src
