"""The rebuilt bar loop: a decision on bar i fills at the NEXT bar's open (a long at the ask), exits are checked on each bar's high
and low with that bar's spread, models are trained on past bars only, and nothing is traded before the first training window."""
import threading
from collections import defaultdict
from types import SimpleNamespace as NS

import pandas as pd
import pytest

import backtester
from backtester import HybridBacktester
from src.backtest_data import BacktestFrame
from src.backtest_models import WalkForwardModels
from src.backtest_symbols import BacktestSymbolClient
from src.config import Cfg
from src.risk import RiskManager

N, TRAIN = 80, 40
PARAMS = {"atr_multiplier_sl": 1.0, "atr_multiplier_tp": 2.0, "trailing_atr_mult": 0.0, "min_prob_long": 0.55, "min_prob_short": 0.55,
          "atr_idx": 0, "min_prob_long_idx": -1, "min_prob_short_idx": -1, "exploration_risk_mult": 1.0, "ac_multiplier": 1.0,
          "context_vector": None}


class Controller:
    def __init__(self):
        self.trades = []
        self.symbol_states = defaultdict(lambda: NS(consecutive_losses=0))

    def get_params(self, sym, context):
        return dict(PARAMS)

    def update(self, trade):
        self.trades.append(trade)


class FakeEns:
    def __init__(self, col, auc):
        self.col, self.ensemble_cv_auc_ = col, auc

    def predict_proba(self, X):
        return X[self.col].astype(float)


def make_bt(monkeypatch, tmp_path, spread_pips=None):
    monkeypatch.chdir(tmp_path)
    cfg = Cfg()
    cfg.symbols = ["USDJPY#"]
    cfg.trading_costs.defaults.slippage_pips = 0.0
    cfg.trading_costs.defaults.commission_per_trade = 0.0
    if spread_pips is not None:
        cfg.trading_costs.defaults.spread_pips = spread_pips
    bt = object.__new__(HybridBacktester)
    bt.cfg, bt.equity, bt.initial_equity, bt.positions = cfg, 10_000.0, 10_000.0, []
    bt.equity_curve, bt.pending, bt.signals, bt.skipped_for_size, bt.blocked_by_auc = [], {}, 0, 0, 0
    bt.skipped_for_spread = 0
    bt._closed_results = []
    bt.blocked_by_session = 0
    bt._closed_R, bt._early_stop_seen, bt.stopped_early = {}, {}, {}
    bt.bar_counters, bt.save_state_every_bars, bt.ts_param_history = {"USDJPY#": 0}, 10 ** 9, []
    bt.backtest_ts_state_file = str(tmp_path / "state.json")
    bt.risk_manager = RiskManager(cfg, BacktestSymbolClient({"USDJPY#": 150.0}), threading.Lock())
    bt.risk_controller = Controller()
    return bt


def build(p=None, q=None, spread=2.0, auc=0.6, edits=None):
    """150.0 flat; `p` / `q` map decision row -> long / short probability; `edits` map (row, column) -> value."""
    idx = pd.date_range("2026-01-05", periods=N, freq="5min", tz="UTC")
    bars = pd.DataFrame({"open": 150.0, "high": 150.02, "low": 149.98, "close": 150.0, "spread": spread}, index=idx)
    X = pd.DataFrame({"atr_14": 0.10, "p": 0.0, "q": 0.0}, index=idx)
    for row, value in (p or {}).items():
        X.iloc[row, X.columns.get_loc("p")] = value
    for row, value in (q or {}).items():
        X.iloc[row, X.columns.get_loc("q")] = value
    for (row, col), value in (edits or {}).items():
        bars.iloc[row, bars.columns.get_loc(col)] = value
    y = pd.Series(0, index=idx)
    frame = BacktestFrame(bars=bars, X=X, y_long=y, y_short=y)
    models = WalkForwardModels(X, y, y, bars["close"], TRAIN, 20, 12,
                               lambda Xt, yt, pt, side, prev: FakeEns("p" if side == "long" else "q", auc))
    return frame, models, idx


def run(bt, frame, models):
    bt._process_bar("USDJPY#", frame, models)
    return bt.positions


def test_a_signal_on_bar_i_fills_at_the_next_open_plus_the_spread_for_a_long(monkeypatch, tmp_path):
    bt = make_bt(monkeypatch, tmp_path)
    frame, models, idx = build(p={40: 0.9})
    (pos,) = run(bt, frame, models)
    assert pos.entry_time == idx[41] and pos.entry_price == pytest.approx(150.002)


def test_a_short_fills_at_the_next_open_without_the_spread(monkeypatch, tmp_path):
    bt = make_bt(monkeypatch, tmp_path)
    frame, models, idx = build(q={40: 0.9})
    (pos,) = run(bt, frame, models)
    assert pos.direction == "short" and pos.entry_time == idx[41] and pos.entry_price == pytest.approx(150.0)


def test_nothing_is_traded_before_the_first_training_window(monkeypatch, tmp_path):
    bt = make_bt(monkeypatch, tmp_path)
    frame, models, idx = build(p={i: 0.9 for i in range(N)})
    positions = run(bt, frame, models)
    assert positions and min(p.entry_time for p in positions) >= idx[TRAIN + 1]


def test_a_signal_on_the_last_row_opens_nothing(monkeypatch, tmp_path):
    bt = make_bt(monkeypatch, tmp_path)
    frame, models, _ = build(p={N - 1: 0.9})
    assert run(bt, frame, models) == []


def test_a_second_signal_while_a_position_is_open_on_the_symbol_opens_nothing(monkeypatch, tmp_path):
    bt = make_bt(monkeypatch, tmp_path)
    frame, models, _ = build(p={40: 0.9, 45: 0.9, 50: 0.9})
    assert len(run(bt, frame, models)) == 1


def test_the_entry_bar_itself_can_stop_the_position_out(monkeypatch, tmp_path):
    bt = make_bt(monkeypatch, tmp_path)
    frame, models, idx = build(p={40: 0.9}, edits={(41, "low"): 149.5})
    (pos,) = run(bt, frame, models)
    assert pos.status == "closed" and pd.Timestamp(pos.exit_time) == idx[41]
    assert pos.exit_price == pytest.approx(pos.initial_sl) and pos.pnl < 0


def test_a_short_pays_the_spread_of_its_exit_bar(monkeypatch, tmp_path):
    # short entered at 150.0, stop 150.1; bar 45's high is 150.098: the ask is high + spread
    wide = make_bt(monkeypatch, tmp_path)
    frame, models, idx = build(q={40: 0.9}, edits={(45, "high"): 150.098, (45, "spread"): 4.0})
    (stopped,) = run(wide, frame, models)
    narrow = make_bt(monkeypatch, tmp_path)
    frame, models, idx = build(q={40: 0.9}, edits={(45, "high"): 150.098, (45, "spread"): 1.0})
    (survived,) = run(narrow, frame, models)
    assert pd.Timestamp(stopped.exit_time) == idx[45]
    assert pd.Timestamp(survived.exit_time) != idx[45]


def test_a_bar_without_a_spread_uses_the_trading_costs_spread(monkeypatch, tmp_path):
    bt = make_bt(monkeypatch, tmp_path, spread_pips=1.0)
    frame, models, _ = build(p={40: 0.9}, spread=float("nan"))
    (pos,) = run(bt, frame, models)
    assert pos.entry_price == pytest.approx(150.0 + 1.0 * 10 * 0.001)     # one pip is ten points on a 3-digit symbol


def test_signals_blocked_by_the_auc_gate_are_counted(monkeypatch, tmp_path):
    bt = make_bt(monkeypatch, tmp_path)
    frame, models, _ = build(p={40: 0.9}, auc=0.50)
    assert run(bt, frame, models) == [] and bt.blocked_by_auc == 1


def test_a_new_backtester_reads_no_saved_bandit_state_and_loads_no_model(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    from src.risk_controller import RiskController

    def boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(RiskController, "load_state", boom)
    monkeypatch.setattr(backtester, "load_ensemble", boom, raising=False)
    cfg = Cfg()
    cfg.symbols = ["USDJPY#"]
    assert HybridBacktester(cfg, None).positions == []


def test_the_default_training_window_is_a_config_key_that_loads():
    assert Cfg().backtesting.train_bars == 45000


def test_an_entry_on_a_bar_whose_spread_is_wider_than_the_atr_cap_is_skipped_like_live(monkeypatch, tmp_path):
    # live refuses an entry when the spread is above `max_spread_atr` (1.0) x the decision ATR: here 150 points = 0.15 > 0.10
    bt = make_bt(monkeypatch, tmp_path)
    frame, models, _ = build(p={40: 0.9}, edits={(41, "spread"): 150.0})
    assert run(bt, frame, models) == [] and bt.skipped_for_spread == 1


def test_the_spread_cap_can_be_turned_off_per_symbol(monkeypatch, tmp_path):
    bt = make_bt(monkeypatch, tmp_path)
    bt.cfg.symbol_overrides = {"USDJPY#": {"max_spread_atr": 0}}
    frame, models, _ = build(p={40: 0.9}, edits={(41, "spread"): 150.0})
    assert len(run(bt, frame, models)) == 1 and bt.skipped_for_spread == 0


# ---- the risk gates are re-checked on the bar a cooldown expires (live `should_trade` clears an expired cooldown FIRST) --------

def test_a_drawdown_past_the_block_keeps_the_backtester_out_after_each_cooldown_expires(monkeypatch, tmp_path):
    """Equity 7,000 against a peak of 10,000 (30% > the block). The cooldown (1 hour = 12 bars) starts on the first decision bar and
    expires on bar 52; that bar must start a new one, not decide. Sized at 1% of 7,000 a trade is possible, so a missing position
    is the gate, not the sizing."""
    bt = make_bt(monkeypatch, tmp_path)
    bt.equity = 7_000.0
    bt.risk_manager.equity_peak = 10_000.0
    bt.risk_manager.watchdog_cfg.cooldown_hours = 1.0
    frame, models, _ = build(p={i: 0.9 for i in range(40, N)})
    assert run(bt, frame, models) == []
    assert bt.signals == 0 and bt.pending == {} and bt.risk_manager.cooldown_until is not None


def test_a_loss_streak_cooldown_is_started_again_on_the_bar_it_expires(monkeypatch, tmp_path):
    """The consecutive-loss watchdog uses the same `if cooldown_until is None` pattern as the drawdown block and had the same leak."""
    bt = make_bt(monkeypatch, tmp_path)
    bt.risk_manager.watchdog_cfg.enabled = True
    bt.risk_manager.watchdog_cfg.max_consecutive_losses = 3
    bt.risk_manager.watchdog_cfg.cooldown_hours = 1.0
    frame, models, idx = build(p={i: 0.9 for i in range(40, N)})
    closed(bt, *[(-1.0, idx[0] - pd.Timedelta(hours=1, minutes=k)) for k in range(5)])
    run(bt, frame, models)
    assert bt.signals == 0 and bt.risk_manager.cooldown_until is not None


def closed(bt, *results):
    """Record closed trades (pnl, exit time) the way `_update_positions` does."""
    for pnl, exit_time in results:
        bt._record_close(exit_time, pnl)


def test_a_loss_streak_older_than_the_lookback_no_longer_blocks_the_backtester(monkeypatch, tmp_path):
    """Live counts only the losing deals of the last 48 hours (`_count_consecutive_losses`), so its watchdog releases. The backtester read
    a counter with no window; with no trade it never changed, and a streak of 5 blocked every later bar (158,539 of them on USDJPY#)."""
    bt = make_bt(monkeypatch, tmp_path)
    bt.risk_manager.watchdog_cfg.enabled = True
    bt.risk_manager.watchdog_cfg.max_consecutive_losses = 3
    frame, models, idx = build(p={i: 0.9 for i in range(40, N)})
    bt.risk_controller.symbol_states["USDJPY#"].consecutive_losses = 5   # the old counter: it blocked forever
    closed(bt, *[(-1.0, idx[0] - pd.Timedelta(hours=49 + k)) for k in range(5)])
    run(bt, frame, models)
    assert bt.signals >= 1 and bt.risk_manager.cooldown_until is None


def test_a_loss_streak_inside_the_lookback_still_blocks_the_backtester(monkeypatch, tmp_path):
    bt = make_bt(monkeypatch, tmp_path)
    bt.risk_manager.watchdog_cfg.enabled = True
    bt.risk_manager.watchdog_cfg.max_consecutive_losses = 3
    frame, models, idx = build(p={i: 0.9 for i in range(40, N)})
    closed(bt, *[(-1.0, idx[0] - pd.Timedelta(hours=1 + k)) for k in range(5)])
    run(bt, frame, models)
    assert bt.signals == 0


def test_a_win_inside_the_lookback_ends_the_streak_for_the_backtester(monkeypatch, tmp_path):
    bt = make_bt(monkeypatch, tmp_path)
    bt.risk_manager.watchdog_cfg.enabled = True
    bt.risk_manager.watchdog_cfg.max_consecutive_losses = 3
    frame, models, idx = build(p={i: 0.9 for i in range(40, N)})
    closed(bt, (-1.0, idx[0] - pd.Timedelta(hours=3)), (-1.0, idx[0] - pd.Timedelta(hours=2)),
           (-1.0, idx[0] - pd.Timedelta(hours=1)), (2.0, idx[0] - pd.Timedelta(minutes=30)))
    run(bt, frame, models)
    assert bt.signals >= 1


def test_the_streak_is_counted_over_all_symbols_like_live(monkeypatch, tmp_path):
    """Live counts every closed deal of the bot's magic number (all symbols), not one symbol's."""
    bt = make_bt(monkeypatch, tmp_path)
    bt.risk_manager.watchdog_cfg.enabled = True
    bt.risk_manager.watchdog_cfg.max_consecutive_losses = 3
    frame, models, idx = build(p={i: 0.9 for i in range(40, N)})
    closed(bt, (-1.0, idx[0] - pd.Timedelta(hours=3)), (-1.0, idx[0] - pd.Timedelta(hours=2)), (-1.0, idx[0] - pd.Timedelta(hours=1)))
    run(bt, frame, models)
    assert bt.signals == 0


def test_without_a_drawdown_or_a_loss_streak_the_gates_let_the_decision_through(monkeypatch, tmp_path):
    bt = make_bt(monkeypatch, tmp_path)
    frame, models, _ = build(p={40: 0.9})
    assert len(run(bt, frame, models)) == 1


def test_the_watchdog_window_edges_and_the_future(monkeypatch, tmp_path):
    bt = make_bt(monkeypatch, tmp_path)
    now = pd.Timestamp("2026-01-10 12:00", tz="UTC")
    closed(bt, (-1.0, now - pd.Timedelta(hours=49)), (-1.0, now - pd.Timedelta(hours=47)), (-1.0, now - pd.Timedelta(hours=1)),
           (-1.0, now + pd.Timedelta(hours=1)))   # the last one closes after `now` (another symbol, run later)
    assert bt._watchdog_losses(now.to_pydatetime()) == 2
    closed(bt, (0.0, now - pd.Timedelta(minutes=10)))   # a zero result is skipped, not a win
    assert bt._watchdog_losses(now.to_pydatetime()) == 2
    closed(bt, (0.5, now - pd.Timedelta(minutes=5)))
    assert bt._watchdog_losses(now.to_pydatetime()) == 0


def test_the_backtester_sizes_with_the_risk_scale_of_the_params(monkeypatch, tmp_path):
    """A drawdown lowers the risk of the trade (fix 87): with a risk scale far below one minimum lot the signal is skipped at sizing."""
    for scale, trades in ((1.0, 1), (0.0001, 0)):
        bt = make_bt(monkeypatch, tmp_path)
        bt.risk_controller.get_params = lambda sym, context, s=scale: dict(PARAMS, risk_scale=s)
        frame, models, _ = build(p={40: 0.9})
        assert len(run(bt, frame, models)) == trades
        assert bt.skipped_for_size == 1 - trades
