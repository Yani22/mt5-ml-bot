"""A retrain child started with spawn (Windows, and so Wine) re-imports main.py, which now opens no log file at import, and never reaches
`run`. The child must apply the config's `logging:` block itself."""
from types import SimpleNamespace as NS

from src import retraining


def test_the_child_sets_up_logging_from_the_config_before_anything_else(monkeypatch):
    order = []
    monkeypatch.setattr(retraining, "setup_logging_from_config", lambda block: order.append(("logging", block)))
    monkeypatch.setattr(retraining, "discard_staged_ensemble", lambda *a: order.append(("discard", a)))

    def stop(*a, **k):
        raise RuntimeError("stop here")

    monkeypatch.setattr(retraining, "DataManager", stop)
    cfg = NS(logging={"level": "INFO", "to_file": True})
    retraining.run_retraining_in_background(cfg, "EURUSD#", None, True, None, {"EURUSD#": None})
    assert order[0] == ("logging", {"level": "INFO", "to_file": True})
