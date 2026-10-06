"""A retrain child used to save straight into the live model folder, so "rejected" models were already the ones on disk (a
reconnect or restart loaded them), acceptance compared a model with itself after a reload and sent a false "rejected"
alert, the child's gate (global `risk.min_auc_improvement`) could disagree with acceptance (per-symbol), and `Ensemble.save`
rewrote the live folder in place. Now the child saves to `models/_staging/`, acceptance decides with one threshold, and an
accepted model is promoted on disk before it is used in memory."""
import os
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src import retraining, utils

SYM = "EURUSD#"
LIVE = "EURUSD_ensemble_long"


@pytest.fixture
def models(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "MODEL_DIR", str(tmp_path))
    return tmp_path


def mark(folder, text):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "m.txt"), "w") as f:
        f.write(text)


def read(folder):
    with open(os.path.join(folder, "m.txt")) as f:
        return f.read()


class FakeEnsemble:
    """What safe_retrain_ensemble needs: fit, an AUC and save(path)."""
    auc = 0.62

    def __init__(self, cfg, model_params=None, n_jobs=-1):
        self.ensemble_cv_auc_ = type(self).auc

    def fit(self, *a, **k):
        pass

    def save(self, path):
        mark(path, "new")


def retrain(monkeypatch, old_auc=0.60, new_auc=0.62, **kw):
    monkeypatch.setattr(utils, "Ensemble", type("E", (FakeEnsemble,), {"auc": new_auc}))
    cfg = NS(risk=NS(min_auc_improvement=0.005))
    return utils.safe_retrain_ensemble(cfg, SYM, NS(ensemble_cv_auc_=old_auc), None, None, None, model_type="long",
                                       model_params={}, **kw)


# ---- the child's save ---------------------------------------------------------------------------------------------

def test_a_staged_retrain_saves_to_the_staging_folder_and_leaves_the_live_one_alone(models, monkeypatch):
    mark(models / LIVE, "old")
    retrain(monkeypatch, staged=True)
    assert read(models / "_staging" / LIVE) == "new"
    assert read(models / LIVE) == "old"


def test_the_default_retrain_still_saves_to_the_live_folder(models, monkeypatch):
    """trainer.py and backtester.py call it without `staged`."""
    mark(models / LIVE, "old")
    retrain(monkeypatch)
    assert read(models / LIVE) == "new"


def test_the_childs_gate_uses_the_threshold_it_is_given(models, monkeypatch):
    retrain(monkeypatch, old_auc=0.60, new_auc=0.62, staged=True, min_improvement=0.05)   # +0.02 < 0.05
    assert not (models / "_staging" / LIVE).exists()
    retrain(monkeypatch, old_auc=0.60, new_auc=0.62, staged=True)                         # default 0.005
    assert (models / "_staging" / LIVE).exists()


def test_a_dry_run_stages_nothing(models, monkeypatch):
    retrain(monkeypatch, staged=True, dry_run=True)
    assert not (models / "_staging").exists()


# ---- promotion on disk --------------------------------------------------------------------------------------------

def test_promote_replaces_the_live_folder_and_leaves_no_staged_or_backup_folder(models):
    mark(models / LIVE, "old")
    mark(models / "_staging" / LIVE, "new")
    utils.promote_staged_ensemble(SYM, "long")
    assert read(models / LIVE) == "new"
    assert not (models / "_staging" / LIVE).exists() and not (models / (LIVE + ".bak")).exists()


def test_promote_without_a_live_folder_just_moves_the_staged_one(models):
    mark(models / "_staging" / LIVE, "new")
    utils.promote_staged_ensemble(SYM, "long")
    assert read(models / LIVE) == "new"


def test_a_failed_second_rename_puts_the_old_model_back(models, monkeypatch):
    mark(models / LIVE, "old")
    mark(models / "_staging" / LIVE, "new")
    real, calls = os.rename, []

    def flaky(src, dst):
        calls.append(src)
        if len(calls) == 2:
            raise OSError("locked")
        return real(src, dst)

    monkeypatch.setattr(utils.os, "rename", flaky)
    with pytest.raises(OSError):
        utils.promote_staged_ensemble(SYM, "long")
    monkeypatch.setattr(utils.os, "rename", real)
    assert read(models / LIVE) == "old"


def test_load_restores_a_backup_left_by_an_interrupted_promotion(models, monkeypatch):
    mark(models / (LIVE + ".bak"), "old")                      # crashed between the two renames: no live folder
    loaded = []
    monkeypatch.setattr(utils.Ensemble, "load", classmethod(lambda cls, path, cfg, model_params=None: loaded.append(path) or "ens"))
    assert utils.load_ensemble(NS(), SYM, "long", model_params={}) == "ens"
    assert read(models / LIVE) == "old" and loaded == [str(models / LIVE)]


def test_discard_removes_the_staged_folder_and_is_quiet_when_there_is_none(models):
    mark(models / "_staging" / LIVE, "new")
    utils.discard_staged_ensemble(SYM, "long")
    utils.discard_staged_ensemble(SYM, "long")
    assert not (models / "_staging" / LIVE).exists()


# ---- the child process --------------------------------------------------------------------------------------------

def test_the_child_clears_old_staged_models_and_retrains_into_staging_with_the_symbol_threshold(monkeypatch):
    idx = pd.date_range("2026-01-05", periods=5, freq="5min")
    frame = pd.DataFrame({"close": 1.0}, index=idx)
    monkeypatch.setattr(retraining, "DataManager", lambda cfg: NS(load_cached=lambda *a, **k: (frame, frame, None)))
    monkeypatch.setattr(retraining, "generate_long_short_labels", lambda *a: (pd.Series(1, index=idx), pd.Series(0, index=idx)))
    monkeypatch.setattr(retraining, "load_ensemble", lambda *a, **k: NS(ensemble_cv_auc_=0.6))
    order = []
    monkeypatch.setattr(retraining, "discard_staged_ensemble", lambda sym, side: order.append(("discard", side)))
    safe = MagicMock(side_effect=lambda *a, **k: order.append(("retrain", k["model_type"])))
    monkeypatch.setattr(retraining, "safe_retrain_ensemble", safe)
    cfg = NS(prediction_horizon=3, retraining_window_bars=100, risk=NS(min_auc_improvement=0.005),
             get_symbol_value=lambda sym, key, default=None: 0.02 if key == "min_auc_improvement" else default)

    retraining.run_retraining_in_background(cfg, SYM, NS(min_pct_change=0.0), False, None, {SYM: {}})

    assert order[:2] == [("discard", "long"), ("discard", "short")] and len(order) == 4
    for call in safe.call_args_list:
        assert call.kwargs["staged"] is True and call.kwargs["min_improvement"] == 0.02


# ---- acceptance ---------------------------------------------------------------------------------------------------

def accept(monkeypatch, staged, old=(0.60, 0.60), processors=None, promote_error=None, notifier=None):
    """`staged` maps side -> ensemble, an Exception to raise, or None for nothing staged."""
    def load_staged(cfg, sym, side, model_params):
        if isinstance(staged[side], Exception):
            raise staged[side]
        return staged[side]

    promote, discard = MagicMock(side_effect=promote_error), MagicMock()
    monkeypatch.setattr(retraining, "_load_staged", load_staged)
    monkeypatch.setattr(retraining, "promote_staged_ensemble", promote)
    monkeypatch.setattr(retraining, "discard_staged_ensemble", discard)
    longs, shorts, aucs = {SYM: NS(ensemble_cv_auc_=old[0])}, {SYM: NS(ensemble_cv_auc_=old[1])}, {}
    cfg = NS(risk=NS(min_auc_improvement=0.005), get_symbol_value=lambda s, k, d=None: d)
    retraining._handle_model_acceptance(SYM, cfg, longs, shorts, aucs, MagicMock(), notifier, {SYM: {}}, processors=processors)
    return longs[SYM], shorts[SYM], promote, discard


def ens(auc):
    return NS(ensemble_cv_auc_=auc)


def test_an_accepted_side_is_promoted_on_disk_and_a_rejected_one_is_discarded(monkeypatch):
    new_long = ens(0.65)
    long_, short_, promote, discard = accept(monkeypatch, {"long": new_long, "short": ens(0.55)})
    assert long_ is new_long and short_.ensemble_cv_auc_ == 0.60
    promote.assert_called_once_with(SYM, "long")
    discard.assert_called_once_with(SYM, "short")


def test_nothing_staged_is_not_a_rejection(monkeypatch):
    notifier = MagicMock()
    long_, short_, promote, discard = accept(monkeypatch, {"long": None, "short": None}, notifier=notifier)
    assert long_.ensemble_cv_auc_ == 0.60 and short_.ensemble_cv_auc_ == 0.60
    assert not promote.called and not discard.called
    assert not any(c.kwargs.get("level") == "WARNING" for c in notifier.send_message.call_args_list)


def test_a_staged_model_that_cannot_be_loaded_is_discarded_and_not_used(monkeypatch):
    sp = NS(set_models=MagicMock())
    long_, short_, promote, discard = accept(monkeypatch, {"long": ValueError("bad manifest"), "short": None}, processors={SYM: sp})
    assert long_.ensemble_cv_auc_ == 0.60 and not promote.called and not sp.set_models.called
    discard.assert_called_once_with(SYM, "long")


def test_a_staged_model_without_an_auc_is_discarded(monkeypatch):
    _, _, promote, discard = accept(monkeypatch, {"long": NS(), "short": None})
    assert not promote.called
    discard.assert_called_once_with(SYM, "long")


def test_memory_is_only_updated_after_the_disk_promotion_succeeds(monkeypatch):
    sp = NS(set_models=MagicMock())
    long_, _, promote, _ = accept(monkeypatch, {"long": ens(0.65), "short": None}, processors={SYM: sp},
                                  promote_error=OSError("locked"))
    assert long_.ensemble_cv_auc_ == 0.60 and not sp.set_models.called


def test_a_promoted_folder_still_verifies_against_its_signed_manifest(models, monkeypatch):
    """The manifest covers file contents and relative paths, not the folder name, so the rename keeps it valid; a half-written
    staged folder (no manifest yet, `Ensemble.save` signs last) is refused."""
    from src.model_integrity import ModelIntegrityError, sign, verify
    monkeypatch.setenv("MODEL_SIGNING_KEY", "test-key-1")
    mark(models / LIVE, "old")
    mark(models / "_staging" / LIVE, "new")
    with pytest.raises(ModelIntegrityError):
        verify(str(models / "_staging" / LIVE))
    sign(str(models / "_staging" / LIVE))
    utils.promote_staged_ensemble(SYM, "long")
    verify(str(models / LIVE))


# ---- acceptance must never raise into main's trading loop (a raise there means a reconnect, and the finished child stays
# ---- tracked, so it would raise again every few seconds) -----------------------------------------------------------

def test_an_incumbent_without_an_auc_is_replaced_by_the_staged_model(monkeypatch):
    new_long = ens(0.60)
    long_, _, promote, _ = accept(monkeypatch, {"long": new_long, "short": None}, old=(None, 0.60))
    assert long_ is new_long
    promote.assert_called_once_with(SYM, "long")


def test_a_failure_while_applying_one_side_does_not_escape_and_does_not_skip_the_other_side(monkeypatch):
    sp = NS(set_models=MagicMock(side_effect=[RuntimeError("boom"), None]))
    new_short = ens(0.70)
    _, short_, promote, discard = accept(monkeypatch, {"long": ens(0.65), "short": new_short}, processors={SYM: sp})
    assert short_ is new_short                                     # the short side was still judged and applied
    assert promote.call_count == 2


def test_a_monitor_failure_does_not_escape(monkeypatch):
    monkeypatch.setattr(retraining, "_load_staged", lambda cfg, sym, side, params: ens(0.65) if side == "long" else None)
    monkeypatch.setattr(retraining, "promote_staged_ensemble", MagicMock())
    monkeypatch.setattr(retraining, "discard_staged_ensemble", MagicMock())
    monitor = MagicMock()
    monitor.update_ensemble_auc.side_effect = RuntimeError("boom")
    cfg = NS(risk=NS(min_auc_improvement=0.005), get_symbol_value=lambda s, k, d=None: d)
    retraining._handle_model_acceptance(SYM, cfg, {SYM: ens(0.60)}, {SYM: ens(0.60)}, {}, monitor, None, {SYM: {}})
