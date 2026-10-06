"""C2 (purge gap): a label looks `horizon` bars ahead, so a training row within `horizon` bars before a validation block shares
part of its future with that block. The cross-validation of the ensemble AND of each member (it nests: Ensemble CV calls member
fit, which runs its own split) leaves a gap of `horizon` rows between the training and the validation rows, so the CV AUC that
gates trading is not inflated by overlapping labels. The horizon is the one the labels were built with (a tuned value wins)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src import ensemble as ensemble_mod
from src import strategy_ml as strategy_mod
from src.config import Cfg
from src.ensemble import Ensemble
from src.strategy_ml import MLStrategy

N = 2400


class RecordingSplit(TimeSeriesSplit):
    calls = []

    def split(self, X, y=None, groups=None):
        for tr, va in super().split(X, y, groups):
            RecordingSplit.calls.append((self.gap, tr, va))
            yield tr, va


class EnsembleSplit(RecordingSplit):
    """Its own recorder, so the ensemble's own split is exercised and not only the members'."""
    calls = []

    def split(self, X, y=None, groups=None):
        for tr, va in TimeSeriesSplit.split(self, X, y, groups):
            EnsembleSplit.calls.append((self.gap, tr, va))
            yield tr, va


def data(n=N):
    rng = np.random.default_rng(1)
    idx = pd.date_range("2026-01-05", periods=n, freq="5min", tz="UTC")
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)}, index=idx)
    y = pd.Series((rng.normal(size=n) > 0).astype(int), index=idx)
    return X, y


def make_cfg(horizon=12):
    cfg = Cfg()
    cfg.models = [{"name": "lgbm", "defaults": {"n_estimators": 10, "max_depth": 3}}]
    cfg.cv_samples_per_split = 400
    cfg.prediction_horizon = horizon
    return cfg


def clear():
    RecordingSplit.calls = []
    EnsembleSplit.calls = []


def test_a_member_leaves_a_gap_between_its_training_and_validation_rows(monkeypatch):
    clear()
    monkeypatch.setattr(strategy_mod, "TimeSeriesSplit", RecordingSplit)
    member = MLStrategy(model="lgbm", calibrate=False, cv_samples_per_split=400, n_estimators=10, max_depth=3)
    member.purge_gap = 12
    X, y = data()
    member.fit(X, y)
    assert RecordingSplit.calls
    for gap, tr, va in RecordingSplit.calls:
        assert gap == 12 and va.min() - tr.max() > 12


def test_a_member_without_a_gap_splits_as_before(monkeypatch):
    clear()
    monkeypatch.setattr(strategy_mod, "TimeSeriesSplit", RecordingSplit)
    member = MLStrategy(model="lgbm", calibrate=False, cv_samples_per_split=400, n_estimators=10, max_depth=3)
    X, y = data()
    member.fit(X, y)
    assert all(gap == 0 for gap, _, _ in RecordingSplit.calls)


def test_the_ensemble_uses_the_label_horizon_as_its_gap_and_hands_it_to_the_members(monkeypatch):
    clear()
    monkeypatch.setattr(strategy_mod, "TimeSeriesSplit", RecordingSplit)
    monkeypatch.setattr(ensemble_mod, "TimeSeriesSplit", EnsembleSplit)
    cfg = make_cfg(horizon=12)
    ens = Ensemble(cfg)
    X, y = data()
    ens.fit(X, y)
    assert ens.purge_gap == 12 and all(m.purge_gap == 12 for m in ens.members.values())
    for calls in (RecordingSplit.calls, EnsembleSplit.calls):   # the members' splits and the ensemble's own
        assert calls and all(gap == 12 and va.min() - tr.max() > 12 for gap, tr, va in calls)


def test_a_tuned_horizon_wins_over_the_config(monkeypatch):
    ens = Ensemble(make_cfg(horizon=12), model_params={"prediction_horizon": 7})
    assert ens.purge_gap == 7


def test_the_tuned_horizon_is_not_passed_to_a_model_as_a_parameter():
    ens = Ensemble(make_cfg(horizon=12), model_params={"prediction_horizon": 7, "lgbm": {"n_estimators": 5}})
    assert ens.members and not ens.failed_members
