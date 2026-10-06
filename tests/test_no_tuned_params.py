"""Without `optuna_params/<symbol>_best_params.json`, `load_optuna_params` returns None. `trainer.py`, the retrain child and the
backtester called `.get` on that None for the tuned horizon and threshold, so the trainer and the backtester crashed and the
daily live retrain failed inside the child's catch-all (logged, nothing trained). They now fall back to the config values."""
from types import SimpleNamespace as NS

import pandas as pd
import pytest

import backtester
import trainer
from src import retraining

SYM = "USDJPY#"


class Stop(Exception):
    pass


def test_the_child_retrains_with_the_config_defaults_when_there_are_no_tuned_params(monkeypatch):
    idx = pd.date_range("2026-01-05", periods=5, freq="5min")
    frame = pd.DataFrame({"close": 1.0}, index=idx)
    seen = {}
    monkeypatch.setattr(retraining, "DataManager", lambda cfg: NS(load_cached=lambda *a, **k: (frame, frame, None)))
    monkeypatch.setattr(retraining, "generate_long_short_labels",
                        lambda data, horizon, pct: seen.update(horizon=horizon, pct=pct) or (pd.Series(1, index=idx), pd.Series(0, index=idx)))
    monkeypatch.setattr(retraining, "load_ensemble", lambda *a, **k: NS(ensemble_cv_auc_=0.6))
    monkeypatch.setattr(retraining, "discard_staged_ensemble", lambda *a: None)
    calls = []
    monkeypatch.setattr(retraining, "safe_retrain_ensemble", lambda *a, **k: calls.append(k))
    cfg = NS(prediction_horizon=12, retraining_window_bars=100, risk=NS(min_auc_improvement=0.005),
             get_symbol_value=lambda sym, key, default=None: default)

    retraining.run_retraining_in_background(cfg, SYM, NS(min_pct_change=0.0002), False, None, {SYM: None})

    assert len(calls) == 2                                           # long and short both retrained
    assert seen == {"horizon": 12, "pct": 0.0002}
    assert all(call["model_params"] is None for call in calls)       # None still means "use the defaults" downstream


def test_the_trainer_uses_the_config_defaults_when_there_are_no_tuned_params(monkeypatch):
    seen = {}
    monkeypatch.setattr(trainer, "load_optuna_params", lambda sym, cfg: None)
    monkeypatch.setattr(trainer, "get_training_data", lambda **k: seen.update(k) or (None, None, None, None))
    cfg = NS(prediction_horizon=12, retraining_window_bars=100, features=NS(min_pct_change=0.0002))

    result = trainer.retrain_symbol(cfg, SYM, dry_run=True, mt5_instance=object())

    assert result["ok"] is False and "Not enough data" in result["reason"]    # got as far as the data step
    assert seen["prediction_horizon"] == 12 and seen["min_pct_change"] == 0.0002


def test_the_backtester_uses_the_config_defaults_when_there_are_no_tuned_params(monkeypatch):
    seen = {}

    def load_backtest_frame(cfg, sym, feature_cfg, horizon, min_pct_change, **k):
        seen.update(prediction_horizon=horizon, min_pct_change=min_pct_change)
        raise Stop

    monkeypatch.setattr(backtester, "load_optuna_params", lambda sym, cfg: None)
    monkeypatch.setattr(backtester, "load_backtest_frame", load_backtest_frame)
    bt = object.__new__(backtester.HybridBacktester)
    bt.cfg = NS(symbols=[SYM], prediction_horizon=12, features=NS(min_pct_change=0.0002), data_source="csv", history_bars=100,
                fetch=NS(raw_data_dir="data/historical_data"),
                context_features=NS(mta=NS(enabled=False), inter_market=NS(enabled=False)))
    try:
        bt.run()
    except Stop:
        pass
    assert seen.get("prediction_horizon") == 12 and seen.get("min_pct_change") == 0.0002
