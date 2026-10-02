"""B7: the monitor state file belongs to one broker account. A peak equity saved on another account (a $10k demo, then a
$30 account) gives a fake 99% drawdown, so a state file from another or an unknown account is not loaded."""
import datetime
import json

from src.config import Cfg
from src.live_performance_monitor import LivePerformanceMonitor

NOW = datetime.datetime(2026, 1, 5, 10, 0, tzinfo=datetime.timezone.utc)


def make_monitor(tmp_path, equity, account_id):
    cfg = Cfg()
    cfg.initial_equity = equity
    cfg.monitoring.monitor_state_file = str(tmp_path / "monitor_state.json")
    monitor = LivePerformanceMonitor(cfg)
    monitor.account_id = account_id
    return monitor


def saved_demo_state(tmp_path, account_id="111@Demo-Server"):
    demo = make_monitor(tmp_path, 10_000.0, account_id)
    demo.update_equity(NOW, 10_500.0)
    demo.save_state()


def test_the_state_file_records_the_account(tmp_path):
    saved_demo_state(tmp_path)
    assert json.loads((tmp_path / "monitor_state.json").read_text())["account_id"] == "111@Demo-Server"


def test_the_same_account_keeps_its_peak(tmp_path):
    saved_demo_state(tmp_path)
    monitor = make_monitor(tmp_path, 10_400.0, "111@Demo-Server")
    monitor.load_state()
    assert monitor.peak_equity == 10_500.0


def test_another_account_starts_fresh_at_its_own_equity(tmp_path):
    saved_demo_state(tmp_path)
    monitor = make_monitor(tmp_path, 30.0, "222@Live-Server")
    monitor.load_state()
    assert monitor.peak_equity == 30.0 and monitor.current_equity == 30.0 and len(monitor.equity_curve) == 0


def test_the_same_login_on_another_server_is_another_account(tmp_path):
    saved_demo_state(tmp_path)
    monitor = make_monitor(tmp_path, 30.0, "111@Other-Server")
    monitor.load_state()
    assert monitor.peak_equity == 30.0


def test_a_state_file_without_an_account_is_not_trusted(tmp_path):
    saved_demo_state(tmp_path, account_id=None)             # what an older version wrote: no account field
    monitor = make_monitor(tmp_path, 30.0, "222@Live-Server")
    monitor.load_state()
    assert monitor.peak_equity == 30.0


def test_an_unknown_current_account_does_not_load_the_saved_peak(tmp_path):
    saved_demo_state(tmp_path)
    monitor = make_monitor(tmp_path, 30.0, None)
    monitor.load_state()
    assert monitor.peak_equity == 30.0


def test_main_stamps_the_monitor_with_the_account_before_loading_its_state():
    """main.py cannot be imported on Linux, so check the wiring in its source."""
    src = open("main.py").read()
    assert 0 <= src.index("live_monitor.account_id") < src.index("live_monitor.load_state()")


def test_an_unknown_account_never_overwrites_the_saved_state(tmp_path):
    """A transient account_info failure must not replace the real account's file with a fresh unstamped one."""
    saved_demo_state(tmp_path)
    monitor = make_monitor(tmp_path, 100.0, None)
    monitor.save_state()
    state = json.loads((tmp_path / "monitor_state.json").read_text())
    assert state["account_id"] == "111@Demo-Server" and state["peak_equity"] == 10_500.0
