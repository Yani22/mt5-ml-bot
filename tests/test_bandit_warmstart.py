"""B8: warm-start merges backtest bandit priors into the live state file. It must keep everything else in that file
(open positions, retrain dates, bar counters), merge each symbol once per state file (not on every start), and be
switchable off with a weight of 0."""
import datetime
import json

from src.bandit_warmstart import merge_warmstart
from src.config import Cfg
from src.risk_controller import RiskController
from test_risk_controller import closed

SYM = "EURUSD#"
DAY = datetime.date(2026, 1, 5)
POSITION = {"ticket": 5, "symbol": SYM, "direction": "long", "entry_price": 1.1, "atr": 0.001, "sl": 1.099,
            "tp": 1.102, "lots": 0.01, "risk": 1.0, "sl_atr_mult": 1.0}


def make_rc(path, trades=0):
    cfg = Cfg()
    cfg.symbols = [SYM]
    cfg.thompson_sampling.state_file = str(path)
    cfg.thompson_sampling.bandit_reset_enabled = False
    cfg.thompson_sampling.decay = 1.0   # these tests count arm pulls, not decayed statistics
    rc = RiskController(cfg)
    for _ in range(trades):
        rc.update(closed("long", long_idx=2))
    return rc


def read(path):
    return json.loads(path.read_text())


def long_counts(path):
    return read(path)["symbol_states"][SYM]["min_prob_bandit_long"]["counts"]


def setup(tmp_path, live_trades=1, back_trades=3):
    live_path, back_path = tmp_path / "live.json", tmp_path / "ts_risk_controller_state_backtest_x.json"
    live = make_rc(live_path, live_trades)
    live.update_last_daily_retrain_date(SYM, DAY)
    live.bar_counters[SYM] = 77
    live.save_state({5: dict(POSITION)})
    make_rc(back_path, back_trades).save_state()
    return live_path, back_path


def test_a_merge_adds_the_backtest_counts_to_the_live_ones(tmp_path):
    live_path, back_path = setup(tmp_path)
    merge_warmstart(str(back_path), str(live_path))
    assert long_counts(live_path)[2] == 4                   # 1 live + 3 backtest


def test_the_merge_keeps_open_positions_retrain_dates_and_bar_counters(tmp_path):
    live_path, back_path = setup(tmp_path)
    merge_warmstart(str(back_path), str(live_path))
    rc = make_rc(live_path)
    cache = rc.load_state()
    assert cache == {5: POSITION}
    assert rc.last_daily_retrain_date[SYM] == DAY
    assert rc.bar_counters[SYM] == 77


def test_merging_again_after_a_normal_save_does_not_double_count(tmp_path):
    live_path, back_path = setup(tmp_path)
    merge_warmstart(str(back_path), str(live_path))
    first = long_counts(live_path)
    rc = make_rc(live_path)                                 # a restart: load, run, save every few seconds
    rc.save_state(rc.load_state())
    merge_warmstart(str(back_path), str(live_path))         # the next start merges again
    assert long_counts(live_path) == first


def test_a_weight_of_zero_leaves_the_live_file_alone(tmp_path):
    live_path, back_path = setup(tmp_path)
    before = live_path.read_text()
    merge_warmstart(str(back_path), str(live_path), warmstart_weight=0.0)
    assert live_path.read_text() == before


def test_a_weight_of_zero_does_not_seed_a_fresh_state(tmp_path):
    back_path = tmp_path / "ts_risk_controller_state_backtest_x.json"
    make_rc(back_path, 3).save_state()
    live_path = tmp_path / "live.json"
    merge_warmstart(str(back_path), str(live_path), warmstart_weight=0.0)
    assert not live_path.exists()
