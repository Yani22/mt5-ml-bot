"""The tuner must save tuned params under the exact file name the bot loads (C4)."""
import os

import pytest

import src.utils as utils
from src.config import Cfg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS = {"models": {"lgbm": {"n_estimators": 123}}, "features": {"min_pct_change": 0.0002},
          "prediction_horizon": 9}


def test_filename_drops_the_hash_and_matches_the_documented_pattern():
    assert utils.optuna_params_filename("EURUSDm#") == "EURUSDm_best_params.json"
    assert utils.optuna_params_filename("EURUSD#") == "EURUSD_best_params.json"
    assert utils.optuna_params_filename("USDJPY") == "USDJPY_best_params.json"


@pytest.mark.parametrize("symbol", ["EURUSDm#", "EURUSD#", "USDJPY"])
def test_a_file_saved_under_the_tuner_name_is_loaded_by_the_bot(tmp_path, monkeypatch, symbol):
    monkeypatch.setattr(utils, "PARAMS_DIR", str(tmp_path))
    path = utils.save_optuna_params(symbol, PARAMS)
    assert os.path.basename(path) == utils.optuna_params_filename(symbol)
    loaded = utils.load_optuna_params(symbol, Cfg())
    assert loaded is not None and loaded["models"]["lgbm"]["n_estimators"] == 123


def test_tuner_saves_through_the_shared_filename_helper():
    src = open(os.path.join(ROOT, "tuner.py")).read()
    assert "save_optuna_params(sym" in src, "tuner must write through the shared helper"
    assert "_best_params." not in src, "tuner must not build the params file name by hand"
    assert "pickle" not in src, "tuned params are JSON, never a pickle"
