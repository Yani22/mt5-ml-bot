"""A setting that nothing reads (or that the code overrides) must say so at start-up instead of looking like it works."""
from src.config import Cfg
from src.ignored_settings import ignored_setting_warnings


def quiet_cfg():
    cfg = Cfg()
    cfg.risk.dynamic_tp = {**cfg.risk.dynamic_tp, "enabled": False}
    return cfg


def test_a_default_config_with_dynamic_tp_off_has_nothing_to_report():
    assert ignored_setting_warnings(quiet_cfg()) == []


def test_force_retrain_on_startup_is_reported():
    cfg = quiet_cfg()
    cfg.force_retrain_on_startup = True
    (msg,) = ignored_setting_warnings(cfg)
    assert "force_retrain_on_startup" in msg and "retrain_time_utc" in msg


def test_ensemble_weights_and_flat_mode_are_reported():
    cfg = quiet_cfg()
    cfg.ensemble = {"weights": {"lgbm": 1.0}, "flat_mode": True}
    text = " ".join(ignored_setting_warnings(cfg))
    assert "ensemble.weights" in text and "ensemble.flat_mode" in text


def test_an_enabled_dynamic_tp_is_reported_as_never_applied():
    cfg = Cfg()
    (msg,) = ignored_setting_warnings(cfg)
    assert "dynamic_tp" in msg and "atr_multiplier_tp" in msg


def test_settings_nothing_reads_are_reported_only_when_they_differ_from_the_default():
    cfg = quiet_cfg()
    cfg.thompson_sampling.reward_normalization_factor = 500.0
    cfg.asymmetric_compounding.lookback_trades = 9
    cfg.optuna_pruning_interval = 7
    text = " ".join(ignored_setting_warnings(cfg))
    for name in ("reward_normalization_factor", "lookback_trades", "optuna_pruning_interval"):
        assert name in text


def test_the_startup_summary_logs_the_warnings(monkeypatch):
    from src import utils
    seen = []
    monkeypatch.setattr(utils.logger, "warning", lambda m, *a, **k: seen.append(m))
    cfg = quiet_cfg()
    cfg.force_retrain_on_startup = True
    utils.log_startup_summary(cfg)
    assert any("force_retrain_on_startup" in m for m in seen)
