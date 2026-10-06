"""K12: the retrain child trained with `n_jobs=-1` (every core) next to the live symbol threads. The child now leaves one core free;
every other caller of the models (trainer, backtester, tuner, live predict) keeps the all-cores default."""
from types import SimpleNamespace as NS

import pandas as pd

from src import retraining
from src.config import Cfg
from src.ensemble import Ensemble
from src.strategy_ml import MLStrategy

SYM = "USDJPY#"


def lgbm_jobs(strategy):
    return strategy._pipe.named_steps["clf"].n_jobs


def test_the_default_is_still_every_core():
    assert lgbm_jobs(MLStrategy(model="lgbm")) == -1


def test_a_strategy_takes_its_thread_count():
    assert lgbm_jobs(MLStrategy(model="lgbm", n_jobs=3)) == 3


def test_an_ensemble_hands_its_thread_count_to_every_member():
    cfg = Cfg()
    cfg.models = [{"name": "lgbm", "defaults": {}}, {"name": "rf", "defaults": {}}]
    ens = Ensemble(cfg, n_jobs=2)
    assert {name: m._pipe.named_steps["clf"].n_jobs for name, m in ens.members.items()} == {"lgbm": 2, "rf": 2}


def test_the_child_leaves_one_core_free(monkeypatch):
    monkeypatch.setattr(retraining.os, "cpu_count", lambda: 8)
    assert retraining.retrain_n_jobs() == 7


def test_the_child_keeps_one_thread_on_a_single_core_machine(monkeypatch):
    monkeypatch.setattr(retraining.os, "cpu_count", lambda: 1)
    assert retraining.retrain_n_jobs() == 1
    monkeypatch.setattr(retraining.os, "cpu_count", lambda: None)
    assert retraining.retrain_n_jobs() == 1


def test_the_child_passes_the_cap_to_both_retrains(monkeypatch):
    monkeypatch.setattr(retraining.os, "cpu_count", lambda: 8)
    idx = pd.date_range("2026-01-05", periods=5, freq="5min")
    frame = pd.DataFrame({"close": 1.0}, index=idx)
    monkeypatch.setattr(retraining, "DataManager", lambda cfg: NS(load_cached=lambda *a, **k: (frame, frame, None)))
    monkeypatch.setattr(retraining, "generate_long_short_labels", lambda *a: (pd.Series(1, index=idx), pd.Series(0, index=idx)))
    monkeypatch.setattr(retraining, "load_ensemble", lambda *a, **k: NS(ensemble_cv_auc_=0.6))
    monkeypatch.setattr(retraining, "discard_staged_ensemble", lambda *a: None)
    calls = []
    monkeypatch.setattr(retraining, "safe_retrain_ensemble", lambda *a, **k: calls.append(k))
    cfg = NS(prediction_horizon=12, retraining_window_bars=100, risk=NS(min_auc_improvement=0.005),
             get_symbol_value=lambda sym, key, default=None: default)

    retraining.run_retraining_in_background(cfg, SYM, NS(min_pct_change=0.0002), False, None, {SYM: None})

    assert [call["n_jobs"] for call in calls] == [7, 7]
