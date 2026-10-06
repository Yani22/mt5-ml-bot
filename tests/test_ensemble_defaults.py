"""`Ensemble.__init__` set `promising_thresholds_ = field(default_factory=list)`; `field` only works inside a dataclass, so a fresh ensemble held a
`dataclasses.Field` object instead of a list. `save` pickled it and `load` handed it back."""
import dataclasses
import os

from src.config import Cfg
from src.ensemble import Ensemble


def fresh():
    cfg = Cfg()
    cfg.models = [{"name": "lgbm", "defaults": {}}]
    return cfg, Ensemble(cfg)


def test_a_fresh_ensemble_has_an_empty_threshold_list():
    _, ens = fresh()
    assert ens.promising_thresholds_ == [] and not isinstance(ens.promising_thresholds_, dataclasses.Field)


def test_an_unfitted_ensemble_round_trips_an_empty_list(tmp_path):
    cfg, ens = fresh()
    path = os.path.join(str(tmp_path), "m")
    ens.save(path)
    assert Ensemble.load(path, cfg).promising_thresholds_ == []
