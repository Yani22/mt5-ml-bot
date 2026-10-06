"""AUDIT fix 87: the drawdown and loss-streak cuts lower the RISK of a trade, not the distance of its stop.
Sizing is risk-based (lots = risk / stop), so a smaller stop bought more lots, was hit more often and paid a larger share of the spread: on USDJPY#
the median stop went from 5.5 pips to 0.7 pips (below the 0.9-pip spread) once equity fell, and 203 trades lost 2.28 R each. The stop keeps the
bandit's multiple and the volatility scale; `risk_scale` (drawdown x loss streak) goes to `RiskManager.position_size`."""
import inspect
import threading
from types import SimpleNamespace as NS

import pytest

import backtester
from src.config import Cfg
from src.risk import RiskManager
from src.risk_controller import RiskController
from src.symbol_processor import SymbolProcessor

SYM = "EURUSD#"
CONTRACT = 100_000


def make(arm=2.0, losses=0):
    cfg = Cfg()
    cfg.symbols = [SYM]
    cfg.thompson_sampling.bandit_reset_enabled = False
    cfg.thompson_sampling.contextual_enabled = False
    cfg.thompson_sampling.atr_grid = [arm]
    cfg.risk.atr_multiplier_sl = 1.0
    cfg.risk.atr_multiplier_tp = 2.0
    cfg.risk.trailing_atr_mult = 1.0
    rc = RiskController(cfg)
    rc.symbol_states[SYM].consecutive_losses = losses
    return rc


def params(rc, vol=0.0005, equity=1000.0, peak=1000.0):
    return rc.get_params(SYM, {"vol": vol, "price": 1.0, "equity": equity, "peak_equity": peak, "ensemble_auc": 0.6})


def test_a_drawdown_and_a_loss_streak_do_not_shrink_the_stop():
    p = params(make(arm=2.0, losses=3), equity=50.0)          # 95% drawdown, 3 losses in a row
    assert p["atr_multiplier_sl"] == pytest.approx(2.0)


def test_they_do_not_shrink_the_trailing_stop_or_the_take_profit_either():
    p = params(make(arm=2.0, losses=3), equity=50.0)
    assert p["trailing_atr_mult"] == pytest.approx(1.0)
    assert p["atr_multiplier_tp"] / p["atr_multiplier_sl"] == pytest.approx(2.0)


def test_the_volatility_scale_still_shrinks_the_stop_and_the_trailing_stop():
    p = params(make(arm=2.0), vol=0.01)                        # 0.0005 / 0.01 + 0.5 = 0.55
    assert p["atr_multiplier_sl"] == pytest.approx(2.0 * 0.55)
    assert p["trailing_atr_mult"] == pytest.approx(0.55)
    assert p["risk_scale"] == pytest.approx(1.0)


def test_the_drawdown_and_the_loss_streak_become_the_risk_scale():
    assert params(make(losses=3), equity=50.0)["risk_scale"] == pytest.approx(0.1 * 0.88)       # max(0.1, 1 - 2 x 0.95) x (1 - 0.2 x 3 / 5)
    assert params(make(), equity=800.0)["risk_scale"] == pytest.approx(0.6)
    assert params(make(losses=50), equity=1000.0)["risk_scale"] == pytest.approx(0.1)


def test_with_no_drawdown_and_no_losses_nothing_changes():
    p = params(make(arm=2.0))
    assert p["atr_multiplier_sl"] == pytest.approx(2.0) and p["risk_scale"] == pytest.approx(1.0) and p["rule_scale"] == pytest.approx(1.0)


def test_the_logged_rule_scale_is_still_the_product_of_all_three_factors():
    p = params(make(losses=3), vol=0.01, equity=800.0)
    assert p["rule_scale"] == pytest.approx(0.55 * 0.6 * 0.88)


# ---- sizing ------------------------------------------------------------------------------------------------------------

class FakeClient:
    def symbol_info(self, symbol):
        return NS(point=1e-5, trade_contract_size=CONTRACT, digits=5, trade_stops_level=0, volume_min=0.01, volume_step=0.01, volume_max=100)


def sized(**kw):
    cfg = Cfg()
    cfg.risk.risk_per_trade = 0.01
    cfg.risk.dynamic_risk = {"enabled": False}
    cfg.risk.max_portfolio_risk = 0.05
    rm = RiskManager(cfg, FakeClient(), threading.Lock())
    return rm.position_size(10_000.0, 0.0010, 0.6, symbol=SYM, sl_distance=0.0010, **kw)


def test_the_risk_scale_lowers_the_lots_and_the_effective_risk():
    lots_full, risk_full = sized()
    lots_half, risk_half = sized(risk_scale=0.5)
    assert lots_half == pytest.approx(lots_full / 2) and risk_half == pytest.approx(risk_full / 2)


def test_a_risk_below_one_minimum_lot_skips_the_trade():
    lots, _ = sized(risk_scale=0.001)
    assert lots == 0.0


def test_the_default_risk_scale_changes_nothing():
    assert sized() == sized(risk_scale=1.0)


# ---- both call sites pass it ----------------------------------------------------------------------------------------------

def test_the_live_decision_passes_the_risk_scale_to_position_size():
    src = inspect.getsource(SymbolProcessor._make_trade_decision)
    assert 'risk_scale=dynamic_risk_params.get("risk_scale"' in src


def test_the_backtester_passes_the_risk_scale_to_position_size():
    src = inspect.getsource(backtester.HybridBacktester._open_pending)
    assert 'risk_scale=params.get("risk_scale"' in src
