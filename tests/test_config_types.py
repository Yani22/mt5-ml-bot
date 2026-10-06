"""Config value types (AUDIT section D): a setting of the wrong type must stop the load, not be coerced.

`bool("false")` is True and `int(2.7)` is 2, so a quoted or fractional value used to turn a setting on or round it away."""
from pathlib import Path

import pytest

from src.config import Cfg, ConfigError

ROOT = Path(__file__).resolve().parent.parent


def load(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return Cfg.from_yaml(str(path))


# ---- blocks ---------------------------------------------------------------------------------------------------

def test_a_quoted_false_in_a_block_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"watchdog\.enabled"):
        load(tmp_path, 'watchdog:\n  enabled: "false"\n')


def test_a_string_for_a_float_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"risk\.risk_per_trade"):
        load(tmp_path, 'risk:\n  risk_per_trade: "0.01"\n')


def test_true_for_an_int_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"risk\.max_positions"):
        load(tmp_path, "risk:\n  max_positions: true\n")


def test_true_for_a_float_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"risk\.risk_per_trade"):
        load(tmp_path, "risk:\n  risk_per_trade: true\n")


def test_a_fraction_for_an_int_in_a_block_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"risk\.max_positions"):
        load(tmp_path, "risk:\n  max_positions: 2.5\n")


def test_a_string_inside_a_grid_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"thompson_sampling\.atr_grid"):
        load(tmp_path, 'thompson_sampling:\n  atr_grid: [1.0, "1.5"]\n')


def test_a_scalar_where_a_list_is_expected_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"thompson_sampling\.atr_grid"):
        load(tmp_path, "thompson_sampling:\n  atr_grid: 1.5\n")


def test_a_scalar_where_a_mapping_is_expected_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"risk\.dynamic_risk"):
        load(tmp_path, "risk:\n  dynamic_risk: 0.01\n")


def test_the_message_names_the_expected_type_and_the_value(tmp_path):
    with pytest.raises(ConfigError, match=r"float.*'0\.01'|'0\.01'.*float"):
        load(tmp_path, 'risk:\n  risk_per_trade: "0.01"\n')


def test_a_nested_block_is_checked_too(tmp_path):
    with pytest.raises(ConfigError, match=r"trading_costs\.defaults\.spread_pips"):
        load(tmp_path, 'trading_costs:\n  defaults:\n    spread_pips: "1.0"\n')


def test_a_context_feature_block_is_checked_too(tmp_path):
    with pytest.raises(ConfigError, match=r"context_features\.mta\.enabled"):
        load(tmp_path, 'context_features:\n  mta:\n    enabled: "false"\n')


def test_an_exponent_without_a_decimal_point_gets_a_hint(tmp_path):
    with pytest.raises(ConfigError, match=r"1\.0e-4"):
        load(tmp_path, "features:\n  min_pct_change: 1e-4\n")


def test_an_exponent_with_a_decimal_point_loads(tmp_path):
    assert load(tmp_path, "features:\n  min_pct_change: 1.0e-4\n").features.min_pct_change == 1e-4


# ---- top level ------------------------------------------------------------------------------------------------

def test_a_quoted_false_at_the_top_level_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match="use_gpu"):
        load(tmp_path, 'use_gpu: "false"\n')


def test_a_quoted_false_for_startup_logging_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match="startup_logging"):
        load(tmp_path, 'startup_logging: "false"\n')


def test_a_fraction_for_history_bars_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match="history_bars"):
        load(tmp_path, "history_bars: 2.7\n")


def test_a_word_for_an_int_is_a_config_error_not_a_value_error(tmp_path):
    with pytest.raises(ConfigError, match="history_bars"):
        load(tmp_path, "history_bars: lots\n")


def test_a_string_for_the_ensemble_auc_gate_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match="min_ensemble_auc"):
        load(tmp_path, 'ensemble:\n  min_ensemble_auc: "0.6"\n')


def test_a_string_for_the_minimum_sample_count_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match="min_samples_for_ensemble"):
        load(tmp_path, 'ensemble_training:\n  min_samples_for_ensemble: "500"\n')


def test_symbols_must_be_a_list_of_strings(tmp_path):
    with pytest.raises(ConfigError, match="symbols"):
        load(tmp_path, "symbols: USDJPY#\n")


# ---- guards: what is valid still loads -------------------------------------------------------------------------

def test_an_int_is_accepted_for_a_float(tmp_path):
    cfg = load(tmp_path, "watchdog:\n  cooldown_hours: 1\nrisk:\n  risk_per_trade: 1\n")
    assert cfg.watchdog.cooldown_hours == 1
    assert cfg.risk.risk_per_trade == 1


def test_null_is_accepted_for_an_optional_field(tmp_path):
    cfg = load(tmp_path, "watchdog:\n  daily_loss_limit: null\nretraining_window_bars: null\n")
    assert cfg.watchdog.daily_loss_limit is None
    assert cfg.retraining_window_bars is None


def test_an_optional_field_takes_a_value_of_its_type(tmp_path):
    assert load(tmp_path, "retraining_window_bars: 5000\n").retraining_window_bars == 5000


def test_real_booleans_and_ints_load(tmp_path):
    cfg = load(tmp_path, "use_gpu: true\nhistory_bars: 3000\nrisk:\n  breakeven_at_1R: false\n  max_positions: 2\n")
    assert cfg.use_gpu is True
    assert cfg.history_bars == 3000
    assert cfg.risk.breakeven_at_1R is False
    assert cfg.risk.max_positions == 2


def test_a_list_of_ints_is_accepted_for_a_float_grid(tmp_path):
    assert load(tmp_path, "thompson_sampling:\n  atr_grid: [1, 2]\n").thompson_sampling.atr_grid == [1, 2]


def test_a_session_filter_of_quoted_times_loads(tmp_path):
    cfg = load(tmp_path, 'risk:\n  session_filter:\n    start: "00:00"\n    end: "23:59"\n')
    assert cfg.risk.session_filter == {"start": "00:00", "end": "23:59"}


def test_the_tracked_config_still_loads():
    Cfg.from_yaml(str(ROOT / "config.yaml"))
