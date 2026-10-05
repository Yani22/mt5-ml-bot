"""K6: no entry while the spread is a large fraction of the decision bar's ATR (rollover hours, news). The cap is
`risk.max_spread_atr` (spread / ATR; 0 turns it off) and can be set per symbol."""
import math
from types import SimpleNamespace as NS

import pytest

from src.config import Cfg
from test_trade_gate import FakeClient, decide, make_cfg, make_rm, make_sp   # ATR in these tests is 0.0010


class TickClient(FakeClient):
    def __init__(self, ask, bid):
        super().__init__()
        self.tick = NS(ask=ask, bid=bid) if ask is not None else None

    def symbol_info_tick(self, symbol):
        return self.tick


def run(ask, bid, cfg=None):
    sp = make_sp(make_rm(cfg or make_cfg(), TickClient(ask, bid)))
    decide(sp)
    return sp.execution.calls


def test_a_normal_spread_still_trades():
    assert len(run(1.1001, 1.1000)) == 1                  # spread 0.0001 = 0.1 ATR


def test_a_rollover_wide_spread_blocks_the_order():
    assert run(1.1013, 1.1000) == []                      # spread 0.0013 = 1.3 ATR


def test_the_cap_is_inclusive_of_a_spread_exactly_at_it():
    assert len(run(1.1010, 1.1000)) == 1                  # 1.0 ATR is allowed, above it is not


@pytest.mark.parametrize("ask, bid", [(1.0990, 1.1000), (math.nan, 1.1000), (1.1001, math.nan), (None, None)])
def test_a_crossed_missing_or_non_finite_quote_blocks_the_order(ask, bid):
    assert run(ask, bid) == []


def test_zero_turns_the_check_off():
    assert len(run(1.1013, 1.1000, make_cfg(max_spread_atr=0.0))) == 1


def test_a_symbol_override_applies():
    cfg = make_cfg()
    cfg.symbol_overrides = {"EURUSD#": {"max_spread_atr": 0.05}}
    assert run(1.1001, 1.1000, cfg) == []                 # 0.1 ATR is over this symbol's 0.05 cap


def test_the_default_cap_is_one_atr():
    assert Cfg().risk.max_spread_atr == 1.0


def test_a_risk_block_with_the_key_loads_and_keeps_its_other_values(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("risk:\n  risk_per_trade: 0.01\n  max_spread_atr: 1.5\n")
    risk = Cfg.from_yaml(str(path)).risk
    assert (risk.risk_per_trade, risk.max_spread_atr) == (0.01, 1.5)
