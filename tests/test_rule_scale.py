"""Characterization of `RiskController._calculate_rule_scale` (volatility, drawdown and loss-streak scaling of the stop), written
before the unreachable second copy of its body was deleted: the values below passed on the code as it was and must not move.
Each one is also the formula by hand: inverse vol = min(1, vol_threshold / vol + 0.5); drawdown = max(0.1, 1 - dd_cut x drawdown);
losses = max(0.1, 1 - consec_cut x losses / 5); product clipped to [0.01, 1]. Defaults: vol_threshold 0.0005, dd_cut 2.0, consec_cut 0.2.
These pin TODAY's behaviour, including the price-unit `vol_threshold` (AUDIT C1: at USDJPY's ATR of about 0.057 the vol factor is about 0.51);
a fix for C1 will change the `vol=0.01` values on purpose and should update them."""
import pytest

from src.config import Cfg
from src.risk_controller import RiskController

SYM = "EURUSD#"


def scale(vol, equity=1000.0, peak=1000.0, losses=0):
    cfg = Cfg()
    cfg.symbols = [SYM]
    cfg.thompson_sampling.bandit_reset_enabled = False
    rc = RiskController(cfg)
    rc.symbol_states[SYM].consecutive_losses = losses
    return rc._calculate_rule_scale(SYM, {"vol": vol, "equity": equity, "peak_equity": peak})


@pytest.mark.parametrize("kwargs, expected", [
    (dict(vol=0.0005), 1.0),                                   # at the threshold: no shrink
    (dict(vol=0.01), 0.55),                                    # 0.0005 / 0.01 + 0.5
    (dict(vol=0.0), 1.0),                                      # no volatility reading: no vol scaling
    (dict(vol=0.0005, equity=800.0), 0.6),                     # 20% drawdown: 1 - 2.0 x 0.2
    (dict(vol=0.0005, equity=50.0), 0.1),                      # 95% drawdown: floored at 0.1
    (dict(vol=0.0005, losses=3), 0.88),                        # 1 - 0.2 x 3 / 5
    (dict(vol=0.0005, losses=50), 0.1),                        # a long streak: floored at 0.1
    (dict(vol=0.01, equity=500.0, losses=5), 0.044),           # 0.55 x 0.1 x 0.8
    (dict(vol=0.0005, peak=None), 1.0),                        # no peak: no drawdown scaling
    (dict(vol=10.0, equity=1.0, losses=500), 0.01),            # the overall floor
])
def test_rule_scale_values(kwargs, expected):
    assert scale(**kwargs) == pytest.approx(expected, abs=1e-9)
