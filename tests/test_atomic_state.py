"""State files are written through a temp file and `os.replace`, a bad file is moved aside instead of being overwritten, and a
failed monitor load keeps the account stamp (without it `save_state` refuses to write for the rest of the run)."""
import json
import os
from types import SimpleNamespace as NS

import pytest

from src import atomic_io
from src.config import Cfg
from src.live_performance_monitor import LivePerformanceMonitor
from src.risk_controller import RiskController


def test_a_write_that_fails_halfway_leaves_the_old_file_untouched(tmp_path):
    path = tmp_path / "state.json"
    atomic_io.atomic_write_json(str(path), {"a": 1})

    class Unserialisable:
        pass

    with pytest.raises(TypeError):
        atomic_io.atomic_write_json(str(path), {"a": Unserialisable()})
    assert json.loads(path.read_text()) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]      # no temp file left behind


def test_the_write_creates_the_missing_directory(tmp_path):
    path = tmp_path / "results" / "deep" / "state.json"
    atomic_io.atomic_write_json(str(path), {"x": 2}, indent=4)
    assert json.loads(path.read_text()) == {"x": 2}


def test_quarantine_moves_a_file_aside_and_keeps_its_bytes(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{ truncated")
    moved = atomic_io.quarantine(str(path))
    assert not path.exists() and open(moved).read() == "{ truncated" and moved.endswith(".corrupt")
    assert atomic_io.quarantine(str(tmp_path / "missing.json")) is None


def monitor(tmp_path):
    cfg = NS(initial_equity=130.0, monitoring=NS(monitor_state_file=str(tmp_path / "m.json"), lookback_days=30))
    m = LivePerformanceMonitor(cfg)
    m.account_id = "123@XM"
    return m, cfg


def test_a_corrupt_monitor_file_keeps_the_account_id_so_saving_still_works(tmp_path):
    m, cfg = monitor(tmp_path)
    (tmp_path / "m.json").write_text('{"account_id": "123@XM", "closed_trades": [{"entry_time": "garbage"}], "peak_equity": 200}')
    m.load_state()
    assert m.account_id == "123@XM" and m.peak_equity == 130.0
    m.peak_equity = 150.0
    m.save_state()
    assert json.loads((tmp_path / "m.json").read_text())["peak_equity"] == 150.0
    assert any(p.name.endswith(".corrupt") for p in tmp_path.iterdir())


def test_a_truncated_monitor_file_is_handled_the_same_way(tmp_path):
    m, cfg = monitor(tmp_path)
    (tmp_path / "m.json").write_text('{"account_id": "123@XM", "peak_eq')
    m.load_state()
    assert m.account_id == "123@XM"
    m.save_state()
    assert json.loads((tmp_path / "m.json").read_text())["account_id"] == "123@XM"


def test_the_monitor_saves_atomically(tmp_path, monkeypatch):
    m, cfg = monitor(tmp_path)
    calls = []
    real = atomic_io.atomic_write_json
    monkeypatch.setattr("src.live_performance_monitor.atomic_write_json", lambda *a, **k: calls.append(a[0]) or real(*a, **k))
    m.save_state()
    assert calls == [str(tmp_path / "m.json")]


def bandit_controller(tmp_path):
    cfg = Cfg()
    cfg.symbols = ["EURUSD#"]
    cfg.thompson_sampling.state_file = str(tmp_path / "ts.json")
    return RiskController(cfg), cfg


def test_the_bandit_state_is_saved_atomically(tmp_path, monkeypatch):
    rc, cfg = bandit_controller(tmp_path)
    calls = []
    real = atomic_io.atomic_write_json
    monkeypatch.setattr("src.risk_controller.atomic_write_json", lambda *a, **k: calls.append(a[0]) or real(*a, **k))
    rc.save_state({})
    assert calls == [cfg.thompson_sampling.state_file]
    assert "symbol_states" in json.loads((tmp_path / "ts.json").read_text())


def test_a_corrupt_bandit_file_is_moved_aside_before_the_next_save_can_overwrite_it(tmp_path):
    rc, cfg = bandit_controller(tmp_path)
    (tmp_path / "ts.json").write_text('{"symbol_states": {"EURUSD#": ')
    assert rc.load_state() == {}
    corrupt = [p for p in tmp_path.iterdir() if p.name.endswith(".corrupt")]
    assert len(corrupt) == 1 and corrupt[0].read_text() == '{"symbol_states": {"EURUSD#": '
    assert not (tmp_path / "ts.json").exists()


def test_every_state_writer_goes_through_the_helper():
    import re
    for path in ("src/live_performance_monitor.py", "src/risk_controller.py"):
        source = open(os.path.join(os.path.dirname(__file__), "..", path)).read()
        assert not re.search(r"open\([^)]*['\"]w['\"]\)", source), f"{path} still writes a file in place"
