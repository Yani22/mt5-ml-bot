"""K22: a reconnect used to reset `retraining_status` to all-False while `retraining_processes` carried over. A child still
running was never accepted when it ended, and if the once-per-day date had not been saved yet a second child was started for
the same symbol (and the first one orphaned). The status is now derived from the process table."""
import datetime
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

from src import retraining
from src.retraining import _check_and_trigger_retraining, retraining_status_for

SYMS = ["EURUSD#", "GBPUSD#"]
NOW = datetime.datetime(2026, 1, 5, 10, 0, tzinfo=datetime.timezone.utc)


class Child:
    def __init__(self, alive):
        self.alive = alive

    def is_alive(self):
        return self.alive


def test_status_follows_the_process_table_running_or_finished():
    status = retraining_status_for(SYMS, {"EURUSD#": Child(True), "GBPUSD#": Child(False)})
    assert status == {"EURUSD#": True, "GBPUSD#": True}


def test_no_process_means_no_status():
    assert retraining_status_for(SYMS, {}) == {"EURUSD#": False, "GBPUSD#": False}


def trigger(monkeypatch, status):
    process_cls = MagicMock()
    monkeypatch.setattr(retraining, "Process", process_cls)
    procs = {"EURUSD#": Child(True)}
    cfg = NS(get_symbol_value=lambda sym, key, default=None: "00:00")
    _check_and_trigger_retraining(cfg, "EURUSD#", {"EURUSD#": NS()}, True, MagicMock(), {}, procs,
                                  status if status is not None else retraining_status_for(SYMS, procs),
                                  {"EURUSD#": None}, MagicMock(), NS(), now_fn=lambda: NOW)
    return process_cls


def test_the_old_all_false_status_starts_a_second_child_for_a_running_one(monkeypatch):
    trigger(monkeypatch, {sym: False for sym in SYMS}).assert_called_once()


def test_the_derived_status_does_not_start_a_second_child(monkeypatch):
    trigger(monkeypatch, None).assert_not_called()


def test_a_finished_child_meets_the_acceptance_condition_after_a_reconnect():
    procs = {"EURUSD#": Child(False)}
    status = retraining_status_for(SYMS, procs)
    assert status["EURUSD#"] and not procs["EURUSD#"].is_alive()   # main.py's "finished" branch condition


def test_main_derives_the_status_from_the_process_table():
    src = (Path(__file__).resolve().parents[1] / "main.py").read_text()
    assert "retraining_status_for(cfg.symbols, retraining_processes)" in src
    assert src.count("retraining_status = {sym: False") == 1
