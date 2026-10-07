"""A key missing from config.yaml must give the dataclass default, not a second default hard-coded in `from_yaml` (it said 100 trials
where the field says 150, and `EURUSD` where the field says `EURUSD#`), and the one default that was less safe than the shipped
YAML (`warmstart_weight` 1.0, which merges backtest bandit state into the live file) is now off."""
from src.config import Cfg

SCALARS = ("symbols", "timeframe", "history_bars", "retrain_every_bars", "prediction_horizon", "data_source", "use_gpu",
           "cv_samples_per_split", "optuna_n_trials", "optuna_pruning_interval", "n_jobs", "initial_equity",
           "min_samples_for_ensemble", "force_retrain_on_startup", "retraining_window_bars", "startup_logging", "magic_number")


def test_an_empty_yaml_gives_the_dataclass_defaults(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("{}\n")
    loaded, default = Cfg.from_yaml(str(path)), Cfg()
    for name in SCALARS:
        assert getattr(loaded, name) == getattr(default, name), name
    assert loaded.optuna_n_trials == 150 and loaded.symbols == ["EURUSD#"]


def test_the_warm_start_merge_is_off_unless_asked_for():
    assert Cfg().thompson_sampling.warmstart_weight == 0.0
