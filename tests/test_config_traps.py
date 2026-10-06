"""Config traps (AUDIT section D): a YAML setting that does nothing, or a key that nothing reads, must not pass silently."""
import logging
import shutil
import subprocess
from pathlib import Path

import pytest

from src.config import Cfg, ConfigError

ROOT = Path(__file__).resolve().parent.parent


def load(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return Cfg.from_yaml(str(path))


# ---- ensemble.min_ensemble_auc was ignored: the gate read the RiskCfg default ------------------------------------

def test_the_ensemble_block_sets_the_auc_gate(tmp_path):
    cfg = load(tmp_path, "symbols: [USDJPY#]\nensemble:\n  min_ensemble_auc: 0.60\n")
    assert cfg.get_symbol_value("USDJPY#", "min_ensemble_auc", 0.55) == 0.60


def test_the_risk_block_still_sets_the_auc_gate(tmp_path):
    cfg = load(tmp_path, "symbols: [USDJPY#]\nrisk:\n  min_ensemble_auc: 0.58\n")
    assert cfg.get_symbol_value("USDJPY#", "min_ensemble_auc", 0.55) == 0.58


def test_a_symbol_override_still_beats_the_ensemble_block(tmp_path):
    cfg = load(tmp_path, "symbols: [USDJPY#]\nensemble:\n  min_ensemble_auc: 0.60\n"
                         "symbol_overrides:\n  USDJPY#:\n    min_ensemble_auc: 0.0\n")
    assert cfg.get_symbol_value("USDJPY#", "min_ensemble_auc", 0.55) == 0.0


def test_the_same_value_in_both_blocks_is_fine(tmp_path):
    cfg = load(tmp_path, "ensemble:\n  min_ensemble_auc: 0.57\nrisk:\n  min_ensemble_auc: 0.57\n")
    assert cfg.risk.min_ensemble_auc == 0.57


def test_two_different_values_in_the_two_blocks_stop_the_load(tmp_path):
    with pytest.raises(ConfigError, match="min_ensemble_auc"):
        load(tmp_path, "ensemble:\n  min_ensemble_auc: 0.60\nrisk:\n  min_ensemble_auc: 0.55\n")


# ---- ensemble_training.min_samples_for_ensemble was never read (the loader looked at the top level) --------------

def test_the_ensemble_training_block_sets_the_minimum_sample_count(tmp_path):
    assert load(tmp_path, "ensemble_training:\n  min_samples_for_ensemble: 500\n").min_samples_for_ensemble == 500


def test_the_top_level_minimum_sample_count_still_works(tmp_path):
    assert load(tmp_path, "min_samples_for_ensemble: 700\n").min_samples_for_ensemble == 700


def test_an_unknown_key_in_ensemble_training_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match="ensemble_training"):
        load(tmp_path, "ensemble_training:\n  min_sample_for_ensemble: 500\n")


def test_two_different_minimum_sample_counts_stop_the_load(tmp_path):
    with pytest.raises(ConfigError, match="min_samples_for_ensemble"):
        load(tmp_path, "min_samples_for_ensemble: 700\nensemble_training:\n  min_samples_for_ensemble: 500\n")


# ---- unknown top-level keys --------------------------------------------------------------------------------------

def test_an_unknown_top_level_key_stops_the_load_and_names_the_likely_fix(tmp_path):
    with pytest.raises(ConfigError) as err:
        load(tmp_path, "symbls: [USDJPY#]\n")
    assert "symbls" in str(err.value) and "symbols" in str(err.value)


def test_a_known_top_level_key_loads(tmp_path):
    assert load(tmp_path, "symbols: [USDJPY#]\nroc_lags_options: [[1, 2]]\n").symbols == ["USDJPY#"]


def test_the_working_copy_of_config_yaml_loads():
    Cfg.from_yaml(str(ROOT / "config.yaml"))


def test_the_committed_config_yaml_loads(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("no git")
    shown = subprocess.run(["git", "show", "HEAD:config.yaml"], cwd=ROOT, capture_output=True, text=True)
    if shown.returncode != 0:
        pytest.skip("no committed config.yaml")
    load(tmp_path, shown.stdout)


# ---- symbol_overrides keys that match no traded symbol ------------------------------------------------------------

def test_an_override_for_a_symbol_that_is_not_traded_is_warned_about(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="src.config"):
        cfg = load(tmp_path, "symbols: [USDJPY#]\nsymbol_overrides:\n  EURUSDm#:\n    min_prob_long: 0.6\n")
    assert cfg.symbols == ["USDJPY#"]
    assert any("EURUSDm#" in r.getMessage() and "symbol_overrides" in r.getMessage() for r in caplog.records)


def test_overrides_for_traded_symbols_do_not_warn(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="src.config"):
        load(tmp_path, "symbols: [USDJPY#]\nsymbol_overrides:\n  USDJPY#:\n    min_prob_long: 0.6\n")
    assert not [r for r in caplog.records if "symbol_overrides" in r.getMessage()]
