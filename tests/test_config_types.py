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


# ---- symbol_overrides: each value is checked against the setting it overrides --------------------------------------

def test_a_string_for_a_float_override_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"symbol_overrides\.USDJPY#\.min_ensemble_auc"):
        load(tmp_path, 'symbols: [USDJPY#]\nsymbol_overrides:\n  USDJPY#:\n    min_ensemble_auc: "0.0"\n')


def test_a_string_inside_an_override_grid_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"symbol_overrides\.USDJPY#\.atr_grid"):
        load(tmp_path, 'symbols: [USDJPY#]\nsymbol_overrides:\n  USDJPY#:\n    atr_grid: [1.0, "1.5"]\n')


def test_a_quoted_false_for_a_bool_override_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"symbol_overrides\.USDJPY#\.breakeven_at_1R"):
        load(tmp_path, 'symbols: [USDJPY#]\nsymbol_overrides:\n  USDJPY#:\n    breakeven_at_1R: "false"\n')


def test_a_cost_override_is_checked_too(tmp_path):
    with pytest.raises(ConfigError, match=r"symbol_overrides\.USDJPY#\.spread_pips"):
        load(tmp_path, 'symbols: [USDJPY#]\nsymbol_overrides:\n  USDJPY#:\n    spread_pips: "2.2"\n')


def test_an_override_block_that_is_not_a_mapping_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"symbol_overrides\.USDJPY#"):
        load(tmp_path, "symbols: [USDJPY#]\nsymbol_overrides:\n  USDJPY#: 0.0\n")


def test_valid_overrides_still_load(tmp_path):
    cfg = load(tmp_path, "symbols: [USDJPY#]\nsymbol_overrides:\n  USDJPY#:\n    min_ensemble_auc: 0\n"
                         "    atr_grid: [1, 1.5]\n    max_spread_atr: 0\n    breakeven_at_1R: false\n    spread_pips: 2.2\n"
                         '    retrain_time_utc: "01:02"\n')
    assert cfg.get_symbol_value("USDJPY#", "min_ensemble_auc") == 0
    assert cfg.get_symbol_value("USDJPY#", "retrain_time_utc") == "01:02"


def test_a_list_of_times_is_accepted_for_the_retrain_override(tmp_path):
    cfg = load(tmp_path, 'symbols: [USDJPY#]\nsymbol_overrides:\n  USDJPY#:\n    retrain_time_utc: ["01:02", "13:02"]\n')
    assert cfg.get_symbol_value("USDJPY#", "retrain_time_utc") == ["01:02", "13:02"]


def test_an_override_key_that_has_no_config_block_but_is_read_is_accepted(tmp_path):
    cfg = load(tmp_path, "symbols: [USDJPY#]\nsymbol_overrides:\n  USDJPY#:\n    min_risk_reward_ratio: 1.5\n")
    assert cfg.get_symbol_value("USDJPY#", "min_risk_reward_ratio") == 1.5


def test_a_misspelled_override_key_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"min_ensemble_aux.*did you mean `min_ensemble_auc`"):
        load(tmp_path, "symbols: [USDJPY#]\nsymbol_overrides:\n  USDJPY#:\n    min_ensemble_aux: 0.0\n")


def test_a_risk_setting_that_nothing_reads_per_symbol_is_refused_as_an_override(tmp_path):
    # block_on_drawdown is a RiskCfg field, but it is read from the global block only
    with pytest.raises(ConfigError, match="block_on_drawdown"):
        load(tmp_path, "symbols: [USDJPY#]\nsymbol_overrides:\n  USDJPY#:\n    block_on_drawdown: 0.1\n")


def test_an_unused_setting_is_refused_as_an_override(tmp_path):
    with pytest.raises(ConfigError, match="min_auc_improvement"):
        load(tmp_path, "symbols: [USDJPY#]\nsymbol_overrides:\n  USDJPY#:\n    min_auc_improvement: 0.01\n")


def _keys_read_per_symbol():
    """Every key the code passes to `get_symbol_value`, found by parsing the source (tests and .venv left out)."""
    import ast
    constant, dynamic = set(), []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT).parts
        if rel[0] in (".venv", "tests") or "site-packages" in rel:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "get_symbol_value":
                arg = node.args[1] if len(node.args) > 1 else None
                if isinstance(arg, ast.Constant):
                    constant.add(arg.value)
                else:
                    dynamic.append(f"{'/'.join(rel)}:{node.lineno}")
    return constant, dynamic


def test_every_key_the_code_reads_per_symbol_can_be_overridden():
    from src.config import OVERRIDABLE
    constant, _ = _keys_read_per_symbol()
    assert constant - set(OVERRIDABLE) == set()


def test_every_overridable_key_is_read_somewhere():
    from src.config import OVERRIDABLE
    constant, _ = _keys_read_per_symbol()
    assert set(OVERRIDABLE) - constant == set()


def test_the_only_per_symbol_read_with_a_computed_key_is_the_startup_log():
    _, dynamic = _keys_read_per_symbol()
    assert [d.split(":")[0] for d in dynamic] == ["src/utils.py"]


# ---- the dicts inside the risk block ------------------------------------------------------------------------------

def test_a_quoted_false_inside_dynamic_risk_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"risk\.dynamic_risk\.enabled"):
        load(tmp_path, 'risk:\n  dynamic_risk:\n    enabled: "false"\n')


def test_a_string_inside_dynamic_tp_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"risk\.dynamic_tp\.base_tp_mult"):
        load(tmp_path, 'risk:\n  dynamic_tp:\n    base_tp_mult: "2.0"\n')


def test_an_unquoted_session_time_stops_the_load(tmp_path):
    # YAML 1.1 reads an unquoted 10:30 as the integer 630
    with pytest.raises(ConfigError, match=r"risk\.session_filter\.start"):
        load(tmp_path, 'risk:\n  session_filter:\n    start: 10:30\n    end: "23:59"\n')


def test_a_dynamic_risk_override_is_checked_too(tmp_path):
    with pytest.raises(ConfigError, match=r"symbol_overrides\.USDJPY#\.dynamic_risk\.enabled"):
        load(tmp_path, 'symbols: [USDJPY#]\nsymbol_overrides:\n  USDJPY#:\n    dynamic_risk:\n      enabled: "false"\n')


def test_dynamic_risk_with_real_types_loads(tmp_path):
    cfg = load(tmp_path, "risk:\n  dynamic_risk:\n    enabled: false\n    base_risk: 0.01\n    max_risk: 1\n")
    assert cfg.risk.dynamic_risk["enabled"] is False
