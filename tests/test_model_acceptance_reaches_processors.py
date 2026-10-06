"""K25: `_handle_model_acceptance` only updated `main`'s ensemble dicts, so an accepted retrain never reached the running
SymbolProcessors (each loads its own models in `__init__`) until the next reconnect. Accepted sides are now pushed into the
processor, and one decision uses one model per side even if a swap lands in the middle of it."""
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

from src import retraining
from src.retraining import _handle_model_acceptance
from test_trade_gate import FakeEnsemble, decide, make_rm, make_sp

SYM = "EURUSD#"
CFG = NS(risk=NS(), get_symbol_value=lambda sym, key, default=None: default)


def accept(monkeypatch, old_long, old_short, new_long, new_short, processors):
    loaded = {"long": new_long, "short": new_short}
    monkeypatch.setattr(retraining, "_load_staged", lambda cfg, sym, side, model_params: loaded[side])
    monkeypatch.setattr(retraining, "promote_staged_ensemble", MagicMock())
    monkeypatch.setattr(retraining, "discard_staged_ensemble", MagicMock())
    longs, shorts, aucs = {SYM: old_long}, {SYM: old_short}, {}
    _handle_model_acceptance(SYM, CFG, longs, shorts, aucs, MagicMock(), None, {SYM: None}, processors=processors)
    return longs, shorts


def real_processor(long, short):
    sp = make_sp(make_rm())
    sp.ens_long, sp.ens_short = long, short
    return sp


class Unfitted(FakeEnsemble):
    """A staged model whose fit was skipped: no member has feature names (C2 refuses it)."""
    def feature_names(self):
        return None


def test_an_accepted_long_reaches_the_processor_and_a_rejected_short_does_not(monkeypatch):
    old_long, old_short = FakeEnsemble(auc=0.56), FakeEnsemble(auc=0.56)
    new_long, new_short = FakeEnsemble(auc=0.60), Unfitted(auc=0.50)
    sp = real_processor(old_long, old_short)
    longs, shorts = accept(monkeypatch, old_long, old_short, new_long, new_short, {SYM: sp})
    assert sp.ens_long is new_long and sp.ens_short is old_short
    assert longs[SYM] is new_long and shorts[SYM] is old_short


def test_an_accepted_short_reaches_the_processor(monkeypatch):
    old_long, old_short = FakeEnsemble(auc=0.56), FakeEnsemble(auc=0.56)
    new_long, new_short = Unfitted(auc=0.50), FakeEnsemble(auc=0.60)
    sp = real_processor(old_long, old_short)
    accept(monkeypatch, old_long, old_short, new_long, new_short, {SYM: sp})
    assert sp.ens_short is new_short and sp.ens_long is old_long


def test_two_rejected_sides_leave_the_processor_alone(monkeypatch):
    old_long, old_short = FakeEnsemble(auc=0.60), FakeEnsemble(auc=0.60)
    sp = real_processor(old_long, old_short)
    accept(monkeypatch, old_long, old_short, Unfitted(auc=0.55), Unfitted(auc=0.55), {SYM: sp})
    assert sp.ens_long is old_long and sp.ens_short is old_short


def test_a_symbol_without_a_processor_still_updates_the_main_dicts(monkeypatch):
    """A symbol whose MT5 client failed to connect has no processor."""
    old_long, old_short = FakeEnsemble(auc=0.56), FakeEnsemble(auc=0.56)
    new_long = FakeEnsemble(auc=0.60)
    longs, _ = accept(monkeypatch, old_long, old_short, new_long, Unfitted(auc=0.50), {})
    assert longs[SYM] is new_long


def test_no_processors_argument_keeps_the_old_behaviour(monkeypatch):
    old_long, old_short = FakeEnsemble(auc=0.56), FakeEnsemble(auc=0.56)
    new_long = FakeEnsemble(auc=0.60)
    longs, _ = accept(monkeypatch, old_long, old_short, new_long, Unfitted(auc=0.50), None)
    assert longs[SYM] is new_long


# ---- one model per side inside one decision ------------------------------------------------------------------

class SwappingEnsemble(FakeEnsemble):
    """Replaces the processor's long model while it is being asked for a probability (a swap from the main thread)."""
    def __init__(self, sp, replacement, **kw):
        super().__init__(**kw)
        self.sp, self.replacement = sp, replacement

    def predict_proba(self, X):
        self.sp.ens_long = self.replacement
        return super().predict_proba(X)


def test_a_swap_in_the_middle_of_a_decision_does_not_mix_the_probability_of_one_model_with_the_auc_of_another():
    sp = make_sp(make_rm())
    replacement = FakeEnsemble(prob=0.9, auc=0.40)                   # would fail the AUC gate
    sp.ens_long = SwappingEnsemble(sp, replacement, prob=0.9, auc=0.60)
    decide(sp)
    assert len(sp.execution.calls) == 1                              # decided on the old model throughout
    assert sp.ens_long is replacement                                # and the next decision uses the new one


def test_set_models_replaces_only_the_sides_it_is_given():
    sp = make_sp(make_rm())
    old_long, old_short = sp.ens_long, sp.ens_short
    new_long = FakeEnsemble()
    sp.set_models(long=new_long)
    assert sp.ens_long is new_long and sp.ens_short is old_short
    new_short = FakeEnsemble()
    sp.set_models(short=new_short)
    assert sp.ens_short is new_short and sp.ens_long is new_long


def test_main_passes_the_running_processors_to_the_acceptance_step():
    src = Path(__file__).resolve().parent.parent.joinpath("main.py").read_text(encoding="utf-8")
    assert "processors=" in src.split("_handle_model_acceptance(", 1)[1].split(")", 1)[0]
