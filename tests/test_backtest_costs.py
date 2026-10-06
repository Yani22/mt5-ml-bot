"""A1: the backtester charges real trading costs. Slippage and spread are in PIPS (10 points on 3 and 5 digit symbols) and
are turned into account currency with the tick value; the bars are bid-only and entry and exit both use the bar close, so a
long or a short pays exactly one spread per round trip. Slippage and spread are charged on every close, including the
forced close at the end of a run."""
import datetime
import threading
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from backtester import HybridBacktester
from src.config import Cfg
from src.risk import RiskManager
from src.trade import SimPosition

T0 = datetime.datetime(2026, 1, 5, 9, 0, tzinfo=datetime.timezone.utc)
ROW_TIME = pd.Timestamp("2026-01-05 10:00")

EURUSD = dict(point=1e-5, digits=5, trade_tick_size=1e-5, trade_tick_value=1.0, trade_contract_size=100_000)
USDJPY = dict(point=1e-3, digits=3, trade_tick_size=1e-3, trade_tick_value=2 / 3, trade_contract_size=100_000)
TWO_DIGIT = dict(point=1e-2, digits=2, trade_tick_size=1e-2, trade_tick_value=1.0, trade_contract_size=100)


class FakeClient:
    def __init__(self, info):
        self.info = NS(trade_stops_level=0, volume_min=0.01, volume_step=0.01, volume_max=100, **info)

    def symbol_info(self, symbol):
        return self.info


class RecordingController:
    def __init__(self):
        self.trades = []

    def update(self, trade):
        self.trades.append(trade)


def make_bt(info=EURUSD, spread=1.0, slippage=0.1, overrides=None):
    cfg = Cfg()
    cfg.trading_costs.defaults.slippage_pips = slippage
    cfg.trading_costs.defaults.commission_per_trade = 0.0
    if spread is not None and hasattr(cfg.trading_costs.defaults, "spread_pips"):
        cfg.trading_costs.defaults.spread_pips = spread
    cfg.symbol_overrides = overrides or {}
    bt = object.__new__(HybridBacktester)
    bt.cfg, bt.equity, bt.positions = cfg, 10_000.0, []
    bt.risk_manager = RiskManager(cfg, FakeClient(info), threading.Lock())
    bt.risk_controller = RecordingController()
    return bt


def open_long(bt, entry, sl, tp, sym="EURUSD#", lots=1.0):
    pos = SimPosition(sym, "long", lots, entry, sl, tp, T0, 0.0010, 0.6, 0.01, entry_equity=10_000.0)
    bt.positions.append(pos)
    return pos


def at(price, spread=0.0):
    """A bar that opens, trades and closes at `price` (a stop or target is hit exactly when `price` reaches it); `spread` in points."""
    return pd.Series({"open": price, "high": price, "low": price, "close": price, "spread": spread}, name=ROW_TIME)


# ---- costs at a stop or target -------------------------------------------------------------------------------------------

def test_a_target_hit_pays_the_slippage_in_pips_and_no_spread_at_the_close():
    bt = make_bt()
    pos = open_long(bt, 1.1000, 1.0990, 1.1020)
    bt._update_positions("EURUSD#", at(1.1020))
    # gross $200; slippage 0.1 pip = 1 point = $1; the spread is in the entry price, not charged here (changed: it was $10 here)
    assert pos.pnl == pytest.approx(199.0)


def test_slippage_alone_is_charged_in_pips_not_points():
    bt = make_bt(spread=0.0)
    pos = open_long(bt, 1.1000, 1.0990, 1.1020)
    bt._update_positions("EURUSD#", at(1.1020))
    assert pos.pnl == pytest.approx(199.0)       # 0.1 pip = $1 (the points-for-pips bug charged $0.10)


def test_a_jpy_pair_is_charged_in_account_currency():
    bt = make_bt(USDJPY)
    pos = open_long(bt, 150.000, 149.850, 150.150, sym="USDJPY#")
    bt._update_positions("USDJPY#", at(150.150))
    # gross 150 points x $0.6667 = $100; slippage 1 point = $0.6667 (the spread is in the fill prices; it was 11 points here)
    assert pos.pnl == pytest.approx(100.0 - 2 / 3)


def test_a_symbol_override_sets_the_spread_used_for_a_bar_that_has_none():
    bt = make_bt(overrides={"EURUSD#": {"spread_pips": 2.0}})
    assert bt._spread_price("EURUSD#", float("nan")) == pytest.approx(2.0 * 10 * 1e-5)    # pips -> points -> price
    assert bt._spread_price("EURUSD#", 3.0) == pytest.approx(3.0 * 1e-5)                  # a bar's own spread (points) wins


def test_the_default_spread_fills_in_for_a_bar_with_a_zero_spread():
    bt = make_bt(spread=1.0)
    assert bt._spread_price("EURUSD#", 0.0) == pytest.approx(10 * 1e-5)


def test_the_closed_trade_sent_to_the_bandit_carries_the_net_pnl():
    bt = make_bt()
    pos = open_long(bt, 1.1000, 1.0990, 1.1020)
    bt._update_positions("EURUSD#", at(1.1020))
    (trade,) = bt.risk_controller.trades
    assert trade.pnl == pytest.approx(pos.pnl) == pytest.approx(199.0)      # the net figure is what is recorded


def test_the_risk_amount_of_a_jpy_trade_is_in_account_currency():
    bt = make_bt(USDJPY)
    open_long(bt, 150.000, 149.850, 150.150, sym="USDJPY#")
    bt._update_positions("USDJPY#", at(150.150))
    (trade,) = bt.risk_controller.trades
    assert trade.risk_amount == pytest.approx(100.0)                         # guard (fix 19): 150 points x $0.6667


# ---- the forced close at the end of a run ----------------------------------------------------------------------------------

def test_the_forced_close_charges_the_same_costs_as_any_other_close():
    bt = make_bt()
    pos = open_long(bt, 1.1000, 1.0990, 1.1050)
    bt._force_close_open_positions("EURUSD#", pd.DataFrame({"close": [1.1020], "spread": [10.0]}, index=[ROW_TIME]))
    assert pos.pnl == pytest.approx(199.0)       # a long closes at the bid (the close); slippage only
    assert bt.equity == pytest.approx(10_000.0 + 199.0)


def test_a_short_is_closed_at_the_ask_so_the_last_bars_spread_costs_it():
    bt = make_bt()
    pos = SimPosition("EURUSD#", "short", 1.0, 1.1000, 1.1010, 1.0950, T0, 0.0010, 0.6, 0.01, entry_equity=10_000.0)
    bt.positions.append(pos)
    bt._force_close_open_positions("EURUSD#", pd.DataFrame({"close": [1.0980], "spread": [10.0]}, index=[ROW_TIME]))
    assert pos.exit_price == pytest.approx(1.0981)              # close + 10 points
    assert pos.pnl == pytest.approx(190.0 - 1.0)


# ---- pips and config ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("info, points_per_pip", [(EURUSD, 10), (USDJPY, 10), (TWO_DIGIT, 1)])
def test_a_pip_is_ten_points_on_three_and_five_digit_symbols_and_one_otherwise(info, points_per_pip):
    rm = make_bt(info).risk_manager
    one_point = rm.get_pip_value("X")
    assert rm.pips_to_money("X", 1.0, 1.0) == pytest.approx(points_per_pip * one_point)


def test_unusable_tick_data_gives_zero_money_for_pips():
    bad = dict(EURUSD, trade_tick_value=0.0)
    assert make_bt(bad).risk_manager.pips_to_money("X", 1.0, 1.0) == 0.0


def test_the_yaml_spread_key_loads_and_keeps_the_other_costs(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("trading_costs:\n  defaults:\n    spread_pips: 0.5\n    slippage_pips: 0.2\n")
    d = Cfg.from_yaml(str(path)).trading_costs.defaults
    assert (d.spread_pips, d.slippage_pips) == (0.5, 0.2)


# ---- the threshold search reads the same cost block ------------------------------------------------------------------------

def test_the_threshold_search_gets_its_spread_from_the_same_trading_costs_block():
    # Ensemble hands asdict(trading_costs.defaults) to calculate_sharpe_ratio, which reads "spread_pips" (it used to fall back
    # to 2.0 because the key did not exist). Pinned so a change to the default is a visible decision (C7: that code is in price units).
    from src.ensemble import Ensemble
    cfg = Cfg()
    cfg.models = [{"name": "lgbm", "defaults": {}}]
    cfg.trading_costs.defaults.spread_pips = 1.5
    assert Ensemble(cfg).trading_costs["spread_pips"] == 1.5
    assert Cfg().trading_costs.defaults.spread_pips == 1.0
