"""C2c / C8: a retrain used to replace the model only when its CV AUC beat the old model's STORED AUC by `min_auc_improvement`,
two numbers from different windows, with a daily retrain that leaves about 300 fresh bars (an AUC standard error of about 0.03,
ten times the threshold). A model whose features differ from the new ones also stayed in place and predicted 0.5. Now the
newest model replaces the old one whenever it fitted (a member has feature names) and reports a finite AUC; the
`min_ensemble_auc` gate alone decides whether it trades. The walk-forward backtest therefore refreshes at every block."""
import inspect
import math
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src import retraining, utils
from src.config import Cfg
from src.ensemble import Ensemble

SYM = "EURUSD#"
LIVE = "EURUSD_ensemble_long"


@pytest.fixture
def models(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "MODEL_DIR", str(tmp_path))
    return tmp_path


def fake(auc, names=("f1", "f2")):
    saved = []
    e = NS(ensemble_cv_auc_=auc, feature_names=lambda: None if names is None else list(names),
           save=lambda path: saved.append(path))
    e.saved = saved
    return e


def retrain(monkeypatch, old, new, **kw):
    monkeypatch.setattr(utils, "Ensemble", lambda cfg, model_params=None, n_jobs=-1: NS(
        **{**new.__dict__, "fit": lambda *a, **k: None, "save": new.save}))
    cfg = NS(risk=NS(), get_symbol_value=lambda s, k, d=None: d)
    return utils.safe_retrain_ensemble(cfg, SYM, old, None, None, None, model_type="long", model_params={}, staged=True, **kw)


# ---- the child's decision -----------------------------------------------------------------------------------------

def test_a_lower_auc_model_still_replaces_the_old_one(models, monkeypatch):
    new = fake(0.52)
    result = retrain(monkeypatch, fake(0.60), new)
    assert result is not None and result.ensemble_cv_auc_ == 0.52 and new.saved


def test_a_model_below_the_trading_gate_still_replaces_the_old_one(models, monkeypatch):
    new = fake(0.54)
    assert retrain(monkeypatch, fake(0.56), new).ensemble_cv_auc_ == 0.54 and new.saved


def test_a_model_with_other_features_replaces_the_old_one(models, monkeypatch):
    new = fake(0.56, names=("f1", "f3"))
    assert retrain(monkeypatch, fake(0.60, names=("f1", "f2")), new).ensemble_cv_auc_ == 0.56 and new.saved


@pytest.mark.parametrize("auc", [None, float("nan")])
def test_a_model_without_a_usable_auc_is_refused(models, monkeypatch, auc):
    old, new = fake(0.60), fake(auc)
    assert retrain(monkeypatch, old, new) is old and not new.saved


def test_a_model_that_never_fitted_is_refused(models, monkeypatch):
    """`Ensemble.fit` skips a small sample with a warning and reports 0.5; it must not replace a working model."""
    old, new = fake(0.60), fake(0.5, names=None)
    assert retrain(monkeypatch, old, new) is old and not new.saved


def test_the_threshold_argument_is_gone():
    assert "min_improvement" not in inspect.signature(utils.safe_retrain_ensemble).parameters


# ---- the live promotion -------------------------------------------------------------------------------------------

def accept(monkeypatch, staged_long, old_auc=0.60):
    promote, discard = MagicMock(), MagicMock()
    monkeypatch.setattr(retraining, "_load_staged", lambda cfg, sym, side, mp: staged_long if side == "long" else None)
    monkeypatch.setattr(retraining, "promote_staged_ensemble", promote)
    monkeypatch.setattr(retraining, "discard_staged_ensemble", discard)
    longs = {SYM: fake(old_auc)}
    cfg = NS(risk=NS(), get_symbol_value=lambda s, k, d=None: d)
    retraining._handle_model_acceptance(SYM, cfg, longs, {SYM: fake(0.6)}, {}, MagicMock(), None, {SYM: {}})
    return longs[SYM], promote, discard


def test_the_live_check_promotes_a_lower_auc_staged_model(monkeypatch):
    new = fake(0.53)
    live, promote, discard = accept(monkeypatch, new)
    assert live is new
    promote.assert_called_once_with(SYM, "long")


def test_the_live_check_discards_a_staged_model_that_never_fitted(monkeypatch):
    new = fake(0.5, names=None)
    live, promote, discard = accept(monkeypatch, new)
    assert live is not new and not promote.called
    discard.assert_called_once_with(SYM, "long")


# ---- feature names of a real ensemble ------------------------------------------------------------------------------

def test_a_real_ensemble_knows_its_feature_names_only_once_fitted():
    cfg = Cfg()
    cfg.models = [{"name": "lgbm", "defaults": {"n_estimators": 10, "max_depth": 3}}]
    cfg.cv_samples_per_split = 400
    ens = Ensemble(cfg)
    assert ens.feature_names() is None
    rng = np.random.default_rng(1)
    idx = pd.date_range("2026-01-05", periods=2400, freq="5min", tz="UTC")
    X = pd.DataFrame({"f1": rng.normal(size=2400), "f2": rng.normal(size=2400)}, index=idx)
    ens.fit(X, pd.Series((rng.normal(size=2400) > 0).astype(int), index=idx))
    assert ens.feature_names() == ["f1", "f2"]
    assert math.isfinite(ens.ensemble_cv_auc_)


def test_a_staged_model_loaded_from_disk_is_still_usable(models, monkeypatch):
    """The live promotion judges the ensemble the child SAVED, loaded back by `_load_staged`, not the one in memory: the
    members' feature names must survive the save and the load, or every daily retrain would be discarded as unfitted."""
    monkeypatch.setenv("MODEL_SIGNING_KEY", "test-key-1")
    cfg = Cfg()
    cfg.models = [{"name": "lgbm", "defaults": {"n_estimators": 10, "max_depth": 3}}]
    cfg.cv_samples_per_split = 400
    rng = np.random.default_rng(1)
    idx = pd.date_range("2026-01-05", periods=2400, freq="5min", tz="UTC")
    X = pd.DataFrame({"f1": rng.normal(size=2400), "f2": rng.normal(size=2400)}, index=idx)
    ens = Ensemble(cfg)
    ens.fit(X, pd.Series((rng.normal(size=2400) > 0).astype(int), index=idx))
    utils.save_ensemble(ens, SYM, "long", staged=True)
    loaded = retraining._load_staged(cfg, SYM, "long", {})
    assert loaded is not None and loaded.feature_names() == ["f1", "f2"]
    assert utils.retrain_is_usable(loaded) == (True, "")
