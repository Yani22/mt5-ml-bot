"""K30: `MT5Client.now_utc` asked `EURUSDm#` for the server time on every call. That symbol is not on the account, so every call
logged a WARNING (the main loop calls it every 5 s) and scanned all 1,645 symbols of the server for any tick. It now asks the
last symbol that worked, then the symbols the bot trades (`time_symbols`), and only then scans Market Watch for the freshest tick."""
import datetime
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from loguru import logger

import src.mt5_client as mc
from src.mt5_client import MT5Client

UTC = datetime.timezone.utc
T1, T2 = 1_700_000_000, 1_700_000_900


def at(ts):
    return datetime.datetime.fromtimestamp(ts, tz=UTC)


class FakeTerminal:
    """symbol -> last tick time (None: no tick); `selected` are the Market Watch symbols."""
    def __init__(self, ticks, selected=None):
        self.ticks, self.asked, self.scans = ticks, [], 0
        self.selected = set(ticks if selected is None else selected)

    def symbol_info_tick(self, name):
        self.asked.append(name)
        ts = self.ticks.get(name)
        return NS(time=ts) if ts else None

    def symbols_get(self):
        self.scans += 1
        return [NS(name=n, select=n in self.selected) for n in self.ticks]


def client(monkeypatch, terminal, time_symbols=None):
    monkeypatch.setattr(mc.mt5, "symbol_info_tick", terminal.symbol_info_tick, raising=False)
    monkeypatch.setattr(mc.mt5, "symbols_get", terminal.symbols_get, raising=False)
    c = MT5Client(login=None, password=None, server=None, time_symbols=time_symbols)
    c._connected = True
    return c


def test_a_configured_symbol_is_asked_first_and_nothing_is_scanned(monkeypatch):
    t = FakeTerminal({"USDJPY#": T1, "GOLD#": T2})
    assert client(monkeypatch, t, ["USDJPY#"]).now_utc() == at(T1)
    assert t.asked == ["USDJPY#"] and t.scans == 0


def test_the_next_call_asks_only_the_symbol_that_worked(monkeypatch):
    t = FakeTerminal({"EURUSD#": None, "USDJPY#": T1})
    c = client(monkeypatch, t, ["EURUSD#", "USDJPY#"])
    c.now_utc()
    t.asked.clear()
    assert c.now_utc() == at(T1)
    assert t.asked == ["USDJPY#"]


def test_the_old_hard_coded_symbol_is_never_asked(monkeypatch):
    t = FakeTerminal({"USDJPY#": T1})
    client(monkeypatch, t, ["USDJPY#"]).now_utc()
    client(monkeypatch, FakeTerminal({"X": None}), None).now_utc()
    assert "EURUSDm#" not in t.asked


def test_the_scan_only_looks_at_market_watch_and_keeps_the_symbol_with_the_newest_tick(monkeypatch):
    t = FakeTerminal({"CLOSED_INDEX": T1, "GOLD#": T2 - 60, "USDMXN": T2, "NOT_WATCHED": T2 + 999},
                     selected={"CLOSED_INDEX", "GOLD#", "USDMXN"})
    c = client(monkeypatch, t)                              # no configured symbols
    assert c.now_utc() == at(T2)                            # the newest watched tick, not the first one that answers
    assert "NOT_WATCHED" not in t.asked
    t.asked.clear()
    c.now_utc()
    assert t.asked == ["USDMXN"] and t.scans == 1           # cached: no second scan


def test_the_fallback_warning_is_logged_once_not_on_every_call(monkeypatch):
    t = FakeTerminal({"EURUSD#": None, "USDJPY#": T1}, selected={"USDJPY#"})
    c = client(monkeypatch, t, ["EURUSD#"])
    lines = []
    sink = logger.add(lambda m: lines.append(str(m)), format="{level} {message}", level="DEBUG")
    try:
        for _ in range(4):
            c.now_utc()
    finally:
        logger.remove(sink)
    assert sum(line.startswith("WARNING") for line in lines) <= 1


def test_with_no_tick_anywhere_the_system_clock_answers(monkeypatch):
    c = client(monkeypatch, FakeTerminal({"USDJPY#": None}), ["USDJPY#"])
    before = datetime.datetime.now(UTC)
    assert before <= c.now_utc() <= datetime.datetime.now(UTC)


def test_a_disconnected_client_does_not_touch_the_terminal(monkeypatch):
    t = FakeTerminal({"USDJPY#": T1})
    c = client(monkeypatch, t, ["USDJPY#"])
    c._connected = False
    c.now_utc()
    assert t.asked == [] and t.scans == 0


def test_main_gives_the_scheduling_client_the_symbols_it_trades():
    src = Path(__file__).resolve().parent.parent.joinpath("main.py").read_text(encoding="utf-8")
    assert "time_symbols=" in src.split("mt5c = MT5Client(", 1)[1].split("mt5c.connect()", 1)[0]
