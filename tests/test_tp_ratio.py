"""A5: the take-profit is the stop times the configured ratio (atr_multiplier_tp / atr_multiplier_sl), whatever arm the
bandit picked and however far rule_scale shrank the stop. It used to be 1.2 for every arm (the stop floor in stop_targets)."""
import threading
from types import SimpleNamespace as NS

import pytest

from src.config import Cfg
from src.risk import RiskManager
from src.risk_controller import RiskController

SYM = "EURUSD#"
PRICE, ATR = 1.1, 0.01


class FakeClient:
    def symbol_info(self, symbol):
        return NS(point=1e-5, digits=5, trade_stops_level=0, trade_contract_size=100_000)


def make(ts_enabled=True, sl=1.0, tp=2.0):
    cfg = Cfg()
    cfg.symbols = [SYM]
    cfg.thompson_sampling.enabled = ts_enabled
    cfg.thompson_sampling.bandit_reset_enabled = False
    cfg.risk.atr_multiplier_sl = sl
    cfg.risk.atr_multiplier_tp = tp
    return RiskController(cfg), RiskManager(cfg, FakeClient(), threading.Lock())


def ratio(rc, rm, vol):
    ctx = {"vol": vol, "equity": 1000.0, "peak_equity": 1000.0, "ensemble_auc": 0.6}
    p = rc.get_params(SYM, ctx)
    sl, tp = rm.stop_targets(PRICE, ATR, "long", 0.6, SYM, sl_mult=p["atr_multiplier_sl"], tp_mult=p["atr_multiplier_tp"])
    return (tp - PRICE) / (PRICE - sl)


@pytest.mark.parametrize("vol", [0.0005, 0.01])         # rule_scale 1.0 and about 0.55
def test_every_bandit_arm_gets_the_configured_ratio(vol):
    rc, rm = make()
    for _ in range(30):                                  # arms are sampled at random
        assert ratio(rc, rm, vol) == pytest.approx(2.0, rel=1e-3)


def test_the_ratio_follows_the_config_multiples():
    rc, rm = make(sl=1.0, tp=3.0)
    assert ratio(rc, rm, 0.0005) == pytest.approx(3.0, rel=1e-3)


def test_with_the_bandit_off_the_config_multiples_apply():
    rc, rm = make(ts_enabled=False)
    assert ratio(rc, rm, 0.0005) == pytest.approx(2.0, rel=1e-3)


def test_a_ratio_below_the_minimum_reward_to_risk_is_lifted_to_it():
    """Pinned so nobody configures 1.0 and silently gets 1.2: stop_targets floors the take-profit at 1.2 x the stop."""
    rc, rm = make(ts_enabled=False, sl=1.0, tp=1.0)
    assert ratio(rc, rm, 0.0005) == pytest.approx(1.2, rel=1e-3)
