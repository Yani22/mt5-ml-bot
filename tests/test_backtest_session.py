"""AUDIT fix 89: the backtester applies `risk.session_filter` like live. Live gates a signal in `RiskManager.should_trade` with the system's real UTC
clock at the moment the bar has closed; the backtest bars are stamped in the broker's server time (New York + 7 hours: UTC+3 in US summer time,
UTC+2 in winter), labelled UTC by this code base. So a bar's decision time is the bar's stamp plus one bar, converted from server time to real UTC."""
import pandas as pd
import pytest

from src.risk import RiskManager
from src.time_utils import server_time_to_utc
from tests.test_backtest_walkforward import build, make_bt, run


# ---- the clock ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("server, utc", [
    ("2026-07-01 12:00", "2026-07-01 09:00"),     # US summer time: server = UTC + 3
    ("2026-01-05 12:00", "2026-01-05 10:00"),     # winter: server = UTC + 2
    ("2026-03-09 00:00", "2026-03-08 21:00"),     # the Monday after the US clock change (2026-03-08): already summer
    ("2026-11-02 00:00", "2026-11-01 22:00"),     # the Monday after the US clock change back (2026-11-01): winter
])
def test_server_time_converts_to_real_utc(server, utc):
    assert server_time_to_utc(pd.Timestamp(server, tz="UTC")) == pd.Timestamp(utc, tz="UTC")


# ---- the shared session check ----------------------------------------------------------------------------------------------

def rm_with(session):
    import threading
    from types import SimpleNamespace as NS
    from src.config import Cfg
    cfg = Cfg()
    cfg.risk.session_filter = session
    return RiskManager(cfg, NS(), threading.Lock())


@pytest.mark.parametrize("hhmm, allowed", [("06:59", False), ("07:00", True), ("12:00", True), ("17:00", True), ("17:01", False)])
def test_session_allows_a_same_day_window_with_both_ends_inclusive(hhmm, allowed):
    rm = rm_with({"start": "07:00", "end": "17:00"})
    assert rm.session_allows(pd.Timestamp(f"2026-01-05 {hhmm}", tz="UTC").to_pydatetime()) is allowed


def test_session_allows_everything_without_a_filter_and_on_an_invalid_one():
    t = pd.Timestamp("2026-01-05 03:00", tz="UTC").to_pydatetime()
    assert rm_with(None).session_allows(t) is True
    assert rm_with({"start": "nonsense", "end": "17:00"}).session_allows(t) is True


def test_should_trade_uses_the_shared_session_check():
    rm = rm_with({"start": "07:00", "end": "17:00"})
    rm.watchdog_cfg.enabled = False
    assert rm.should_trade(pd.Timestamp("2026-01-05 03:00", tz="UTC").to_pydatetime(), 0.0) is False


# ---- the backtester --------------------------------------------------------------------------------------------------------

def session_bt(monkeypatch, tmp_path, start, end):
    bt = make_bt(monkeypatch, tmp_path)
    bt.cfg.risk.session_filter = {"start": start, "end": end}
    bt.risk_manager.risk_cfg.session_filter = {"start": start, "end": end}
    return bt


def test_a_signal_outside_the_session_is_refused_and_counted(monkeypatch, tmp_path):
    bt = session_bt(monkeypatch, tmp_path, "10:00", "11:00")
    frame, models, _ = build(p={40: 0.9})                      # bar 40 is stamped 03:20 server time = 01:20 UTC, decided at 01:25 UTC
    assert run(bt, frame, models) == [] and bt.blocked_by_session == 1 and bt.signals == 0


def test_a_signal_inside_the_session_trades(monkeypatch, tmp_path):
    bt = session_bt(monkeypatch, tmp_path, "01:00", "02:00")
    frame, models, _ = build(p={40: 0.9})
    assert len(run(bt, frame, models)) == 1 and bt.blocked_by_session == 0


def test_the_decision_time_is_the_bar_close_in_real_utc(monkeypatch, tmp_path):
    """Row 40 (the first row the backtester decides on) is stamped 03:20 server time: the bar closes at 03:25 server = 01:25 UTC (winter, UTC+2).
    A window starting at 01:25 includes it, one starting at 01:26 does not; with the stamp itself (01:20 UTC) as the decision time both would be refused."""
    bt = session_bt(monkeypatch, tmp_path, "01:25", "02:00")
    frame, models, _ = build(p={40: 0.9})
    assert len(run(bt, frame, models)) == 1
    bt = session_bt(monkeypatch, tmp_path, "01:26", "02:00")
    frame, models, _ = build(p={40: 0.9})
    assert run(bt, frame, models) == [] and bt.blocked_by_session == 1


def test_the_end_of_the_window_is_refused_like_live_which_decides_seconds_after_the_close(monkeypatch, tmp_path):
    """Live decides a few seconds after the bar closes, so a bar closing exactly at the window's end (01:25:00) is seen at 01:25:0x and refused."""
    bt = session_bt(monkeypatch, tmp_path, "01:00", "01:25")
    frame, models, _ = build(p={40: 0.9})
    assert run(bt, frame, models) == [] and bt.blocked_by_session == 1
    bt = session_bt(monkeypatch, tmp_path, "01:00", "01:26")
    frame, models, _ = build(p={40: 0.9})
    assert len(run(bt, frame, models)) == 1


def test_without_a_filter_nothing_is_refused(monkeypatch, tmp_path):
    bt = make_bt(monkeypatch, tmp_path)
    bt.risk_manager.risk_cfg.session_filter = None
    frame, models, _ = build(p={40: 0.9})
    assert len(run(bt, frame, models)) == 1 and bt.blocked_by_session == 0
