"""Dry-run must not read or write the bandit/monitor state a live run uses (K1)."""
import ast
import fnmatch
import os

import pytest

import main
from src.config import Cfg
from src.risk_controller import RiskController
from src.state_paths import apply_mode_state_paths, mode_state_path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_live_path_is_unchanged():
    assert mode_state_path("results/ts_risk_controller_state.json", False) == "results/ts_risk_controller_state.json"


def test_dry_run_path_is_a_separate_sibling():
    assert mode_state_path("results/ts_risk_controller_state.json", True) == "results/ts_risk_controller_state_dryrun.json"


def test_dry_run_path_is_idempotent():
    once = mode_state_path("results/x.json", True)
    assert mode_state_path(once, True) == once


def test_dry_run_path_cannot_match_the_backtest_warmstart_glob():
    p = os.path.basename(mode_state_path("results/ts_risk_controller_state.json", True))
    assert not fnmatch.fnmatch(p, "ts_risk_controller_state_backtest_*.json")


def _cfg():
    return Cfg.from_yaml(os.path.join(ROOT, "config.yaml"))


def test_apply_changes_bandit_and_monitor_paths_only_in_dry_run():
    live, dry = _cfg(), _cfg()
    live_ts, live_mon = live.thompson_sampling.state_file, live.monitoring.monitor_state_file
    apply_mode_state_paths(live, False)
    apply_mode_state_paths(dry, True)
    assert (live.thompson_sampling.state_file, live.monitoring.monitor_state_file) == (live_ts, live_mon)
    assert dry.thompson_sampling.state_file != live_ts
    assert dry.monitoring.monitor_state_file != live_mon


def test_live_controller_starts_fresh_after_a_dry_run_save(tmp_path):
    dry, live = _cfg(), _cfg()
    dry.thompson_sampling.state_file = str(tmp_path / "ts.json")
    live.thompson_sampling.state_file = str(tmp_path / "ts.json")
    apply_mode_state_paths(dry, True)
    apply_mode_state_paths(live, False)
    RiskController(dry).save_state({1: {"symbol": "EURUSD#"}})
    assert os.path.exists(dry.thompson_sampling.state_file)
    assert not os.path.exists(live.thompson_sampling.state_file)
    assert RiskController(live).load_state() == {}


class _Stop(BaseException):
    """Escapes run() past its `except Exception` reconnect loop."""


@pytest.mark.parametrize("dry_run", [True, False])
def test_run_wires_mode_paths_into_cfg_before_anything_else(monkeypatch, dry_run):
    """Stops at setup_logging, the statement right after the config load, so no MT5 code is reached."""
    seen = {}
    cfg = _cfg()
    configured = cfg.thompson_sampling.state_file, cfg.monitoring.monitor_state_file
    monkeypatch.setenv("ALLOW_LIVE_TRADING", "1")
    monkeypatch.setattr(main.Cfg, "from_yaml", staticmethod(lambda _p: cfg))

    def stop(*_a, **_k):
        seen["paths"] = (cfg.thompson_sampling.state_file, cfg.monitoring.monitor_state_file)
        raise _Stop

    monkeypatch.setattr(main, "setup_logging", stop)
    with pytest.raises(_Stop):
        main.run(dry_run=dry_run)
    expected = tuple(mode_state_path(p, True) for p in configured) if dry_run else configured
    assert seen["paths"] == expected


def test_apply_runs_right_after_config_load():
    src = open(os.path.join(ROOT, "main.py")).read()
    run = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "run")
    calls = [ast.unparse(n) for n in run.body]
    assert any("apply_mode_state_paths(cfg, dry_run)" in c for c in calls)
