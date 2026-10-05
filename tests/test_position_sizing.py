"""Position sizing: the lot size must risk the intended money at the stop that is actually used, and the
portfolio risk cap must compare like with like (money at risk as a fraction of equity)."""
import threading
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from src.config import Cfg
from src.risk import RiskManager
from src.symbol_processor import SymbolProcessor

EQUITY, CONTRACT = 10_000.0, 100_000


class FakeClient:
    def symbol_info_tick(self, symbol):
        return NS(ask=1.1001, bid=1.1000)

    def symbol_info(self, symbol):
        return NS(point=1e-5, trade_contract_size=CONTRACT, digits=5, trade_stops_level=0,
                  volume_min=0.01, volume_step=0.01, volume_max=100)


def make_cfg():
    cfg = Cfg()
    cfg.risk.risk_per_trade = 0.01          # 1% of equity = $100
    cfg.risk.dynamic_risk = {"enabled": False}
    cfg.risk.max_portfolio_risk = 0.05      # 5% = $500 across all open positions
    cfg.risk.atr_multiplier_sl = 1.0
    return cfg


def make_rm():
    return RiskManager(make_cfg(), FakeClient(), threading.Lock())


def size(rm, symbol="EURUSD#", **kw):
    args = dict(equity=EQUITY, atr=0.0010, auc_score=0.6, total_open_risk=0.0, symbol=symbol)
    args.update(kw)
    return rm.position_size(**args)


# ---- B3: size on the stop that is really placed ---------------------------------------------------------------

def test_without_a_stop_distance_the_configured_atr_multiple_is_used():
    lots, risk = size(make_rm())
    assert lots == pytest.approx(1.0) and risk == pytest.approx(0.01)   # $100 / (0.0010 * 100000 = $100 per lot)


@pytest.mark.parametrize("sl_distance", [0.0005, 0.0010, 0.0030])
def test_money_at_risk_at_the_stop_equals_the_budget(sl_distance):
    lots, risk = size(make_rm(), sl_distance=sl_distance)
    assert lots * sl_distance * CONTRACT == pytest.approx(EQUITY * risk, rel=0.02)


def test_a_wider_stop_means_a_smaller_position():
    assert size(make_rm(), sl_distance=0.0020)[0] == pytest.approx(0.5)


# ---- B4: portfolio cap compares money with money --------------------------------------------------------------

def test_a_second_position_is_allowed_while_total_open_risk_is_under_the_cap():
    lots, risk = size(make_rm(), symbol="GBPUSD#", total_open_risk=100.0)   # $100 open = 1% of the 5% cap
    assert lots == pytest.approx(1.0) and risk == pytest.approx(0.01)


def test_the_cap_limits_the_next_position_to_the_remaining_budget():
    lots, risk = size(make_rm(), total_open_risk=450.0)                      # 4.5% open, 0.5% left
    assert risk == pytest.approx(0.005) and lots == pytest.approx(0.5)


@pytest.mark.parametrize("open_risk", [500.0, 800.0])
def test_no_new_position_when_the_cap_is_used_up(open_risk):
    assert size(make_rm(), total_open_risk=open_risk) == (0.0, 0.0)


def test_streak_multiplier_cannot_push_risk_past_the_remaining_cap():
    _, risk = size(make_rm(), total_open_risk=450.0, ac_multiplier=2.0)
    assert risk <= 0.005 + 1e-12


def test_exploration_multiplier_still_scales_risk_down_under_the_cap():
    _, risk = size(make_rm(), exploration_mult=0.5)
    assert risk == pytest.approx(0.005)


# ---- the live caller must size on the stop it sends -----------------------------------------------------------

class FakeEnsemble:
    def __init__(self, prob=0.9):
        self.prob, self.ensemble_cv_auc_ = prob, 0.6

    def predict_proba(self, X):
        return pd.Series([self.prob], index=X.index)


class FakeRiskController:
    def get_params(self, symbol, context):
        return {"atr_multiplier_sl": 2.0, "atr_multiplier_tp": 2.0, "min_prob_long": 0.55, "min_prob_short": 0.55,
                "atr_idx": 0, "min_prob_long_idx": 0, "min_prob_short_idx": 0,
                "exploration_risk_mult": 1.0, "ac_multiplier": 1.0, "context_vector": None}


class RecordingExecution:
    def __init__(self):
        self.calls = []

    def trade(self, **kw):
        self.calls.append(kw)


def test_symbol_processor_sizes_on_the_bandit_stop_not_the_config_stop():
    sp = object.__new__(SymbolProcessor)
    sp.stop_event = threading.Event()
    sp.symbol, sp.dry_run = "EURUSD#", True
    sp.mt5_client, sp.risk_controller = FakeClient(), FakeRiskController()
    sp.risk_manager = make_rm()
    sp.monitor = NS(current_equity=EQUITY, peak_equity=EQUITY)
    sp.execution = RecordingExecution()
    sp.ens_long, sp.ens_short = FakeEnsemble(), FakeEnsemble(prob=0.1)
    idx = pd.date_range("2026-01-05", periods=3, freq="5min")
    X = pd.DataFrame({"atr_14": 0.0010}, index=idx)
    data = pd.DataFrame({"close": 1.1000}, index=idx)

    sp._make_trade_decision(data, X)

    (call,) = sp.execution.calls
    stop_distance = abs(call["price"] - call["sl"])
    assert stop_distance == pytest.approx(0.0020)            # bandit arm 2.0 x ATR, not the config 1.0 x ATR
    assert call["lots"] * stop_distance * CONTRACT == pytest.approx(EQUITY * 0.01, rel=0.02)


# ---- K4: floor lots to the broker step, never round up past the risk budget -----------------------------------

def _rm_with_volume(step, vmin=0.01, vmax=100):
    class Client(FakeClient):
        def symbol_info(self, symbol):
            return NS(point=1e-5, trade_contract_size=CONTRACT, digits=5, trade_stops_level=0,
                      volume_min=vmin, volume_step=step, volume_max=vmax)
    return RiskManager(make_cfg(), Client(), threading.Lock())


def _lots_for_units(units, rm=None):
    """Equity chosen so the exact (unrounded) lot size is `units`: atr 0.01 means $1000 per lot, risk 1%."""
    return size(rm or make_rm(), equity=units * 1000 / 0.01, atr=0.0100)


@pytest.mark.parametrize("units, expected", [(0.015, 0.01), (0.0251, 0.02), (0.0299, 0.02), (1.239, 1.23)])
def test_lots_are_floored_to_the_step_not_rounded_to_nearest(units, expected):
    assert _lots_for_units(units)[0] == pytest.approx(expected)


@pytest.mark.parametrize("units", [0.03, 0.07, 0.29, 0.58, 1.1])
def test_an_exact_multiple_of_the_step_is_not_floored_down_by_float_noise(units):
    assert _lots_for_units(units)[0] == pytest.approx(units)


def test_below_the_minimum_lot_the_trade_is_skipped():
    assert _lots_for_units(0.0099) == (0.0, 0.0)


def test_floored_lots_never_risk_more_than_the_budget():
    for units in (0.011, 0.0199, 0.0451, 0.333, 2.999):
        lots, risk = _lots_for_units(units)
        assert lots * 1000 <= units * 1000 + 1e-9


def test_a_step_finer_than_two_decimals_is_respected():
    lots, _ = _lots_for_units(0.0157, rm=_rm_with_volume(step=0.001, vmin=0.001))
    assert lots == pytest.approx(0.015)


def test_a_coarse_step_floors_to_it():
    assert _lots_for_units(0.47, rm=_rm_with_volume(step=0.1, vmin=0.1))[0] == pytest.approx(0.4)


def test_lots_are_capped_at_the_broker_maximum():
    assert _lots_for_units(5.0, rm=_rm_with_volume(step=0.01, vmax=2.0))[0] == pytest.approx(2.0)
