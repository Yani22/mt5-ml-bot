"""`RiskController.save_state` / `load_state` take the file as an argument (default: the config's path), so the backtester writes
its own file without editing the shared config for the duration of the save."""
import os


from src.config import Cfg
from src.risk_controller import RiskController

SYM = "EURUSD#"


def make_rc(tmp_path):
    cfg = Cfg()
    cfg.symbols = [SYM]
    cfg.thompson_sampling.state_file = str(tmp_path / "config_path.json")
    return RiskController(cfg)


def test_save_and_load_use_the_path_argument_not_the_config_path(tmp_path):
    rc = make_rc(tmp_path)
    other = str(tmp_path / "other.json")
    rc.save_state({1: {"symbol": SYM}}, path=other)
    assert os.path.exists(other) and not os.path.exists(rc.cfg.thompson_sampling.state_file)
    assert list(make_rc(tmp_path).load_state(path=other).values()) == [{"symbol": SYM}]


def test_without_an_argument_the_config_path_is_used(tmp_path):
    rc = make_rc(tmp_path)
    rc.save_state()
    assert os.path.exists(rc.cfg.thompson_sampling.state_file)


def test_the_backtester_persist_leaves_the_config_path_alone_even_while_saving(tmp_path, monkeypatch):
    import tests.test_backtest_walkforward as twf
    bt = twf.make_bt(monkeypatch, tmp_path)
    bt.ts_history_csv = str(tmp_path / "history.csv")
    rc = make_rc(tmp_path)
    rc.cfg = bt.cfg
    bt.cfg.symbols = [SYM]
    original = bt.cfg.thompson_sampling.state_file
    seen = []
    real = rc.save_state

    def spy(*a, **k):
        seen.append(bt.cfg.thompson_sampling.state_file)
        return real(*a, **k)

    rc.save_state = spy
    bt.risk_controller = rc
    target = str(tmp_path / "bt_state.json")
    bt._persist_bandit_state(force_path=target)
    assert os.path.exists(target)
    assert seen == [original] and bt.cfg.thompson_sampling.state_file == original
    assert not os.path.exists(original)


def test_the_backtester_entry_point_no_longer_overwrites_the_config_state_path():
    src = open("backtester.py").read()
    assert "cfg.thompson_sampling.state_file =" not in src


def test_the_backtester_persist_without_a_path_writes_its_own_file_not_the_config_path(tmp_path, monkeypatch):
    import tests.test_backtest_walkforward as twf
    bt = twf.make_bt(monkeypatch, tmp_path)
    bt.ts_history_csv = str(tmp_path / "history.csv")
    rc = make_rc(tmp_path)
    rc.cfg = bt.cfg
    bt.cfg.symbols = [SYM]
    bt.cfg.thompson_sampling.state_file = str(tmp_path / "live.json")
    bt.risk_controller = rc
    bt._persist_bandit_state()
    assert os.path.exists(bt.backtest_ts_state_file) and not os.path.exists(bt.cfg.thompson_sampling.state_file)
