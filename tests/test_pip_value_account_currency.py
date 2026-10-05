"""B13: the money value of a price move must be in ACCOUNT currency (tick value), not quote currency
(point * contract size). For JPY, CHF and CAD quoted pairs the two differ by the exchange rate."""
import threading
from types import SimpleNamespace as NS

import pytest

from src.config import Cfg
from src.risk import RiskManager

EQUITY = 10_000.0


def make_rm(info):
    cfg = Cfg()
    cfg.risk.risk_per_trade = 0.01          # 1% of equity = $100
    cfg.risk.dynamic_risk = {"enabled": False}
    cfg.risk.max_portfolio_risk = 0.05

    class Client:
        def symbol_info(self, symbol):
            return info

    return RiskManager(cfg, Client(), threading.Lock())


def info(point, tick_size, tick_value, contract=100_000, **extra):
    base = dict(point=point, trade_contract_size=contract, digits=3, trade_stops_level=0,
                volume_min=0.01, volume_step=0.01, volume_max=100)
    base.update(dict(trade_tick_size=tick_size, trade_tick_value=tick_value), **extra)
    return NS(**base)


def size(rm, sl_distance):
    return rm.position_size(equity=EQUITY, atr=0.0, auc_score=0.6, total_open_risk=0.0, symbol="USDJPY#",
                            sl_distance=sl_distance)


def test_jpy_pair_is_sized_with_the_tick_value_in_account_currency():
    # USDJPY on a USD account: 1 tick (0.001) of one lot is worth about $0.667, not 100 JPY.
    rm = make_rm(info(point=0.001, tick_size=0.001, tick_value=2 / 3))
    lots, _ = size(rm, sl_distance=0.150)    # 150 points x $0.6667 = $100 per lot -> $100 budget buys 1 lot
    assert lots == pytest.approx(1.0)


def test_value_per_point_scales_when_the_tick_is_larger_than_the_point():
    rm = make_rm(info(point=0.01, tick_size=0.1, tick_value=1.0, contract=100))
    assert rm.get_pip_value("US30#") == pytest.approx(0.1)     # $1 per 0.1 tick = $0.1 per 0.01 point
    lots, _ = size(rm, sl_distance=10.0)                        # 1000 points x $0.1 = $100 per lot
    assert lots == pytest.approx(1.0)


@pytest.mark.parametrize("tick_value, tick_size", [(0.0, 0.001), (-1.0, 0.001), (0.6667, 0.0), (None, 0.001)])
def test_unusable_tick_data_gives_no_trade_instead_of_the_quote_currency_value(tick_value, tick_size):
    # A zero tick value (symbol not in Market Watch, cross rate not quoted) must not fall back to the 150x wrong value.
    rm = make_rm(info(point=0.001, tick_size=tick_size, tick_value=tick_value))
    assert rm.get_pip_value("USDJPY#") == 0.0
    assert size(rm, sl_distance=0.150) == (0.0, 0.0)


def test_a_usd_quoted_pair_sizes_exactly_as_before():
    rm = make_rm(info(point=1e-5, tick_size=1e-5, tick_value=1.0))   # EURUSD on a USD account
    assert rm.get_pip_value("EURUSD#") == pytest.approx(1.0)         # = point * contract size (1e-5 * 100000)
    lots, risk = size(rm, sl_distance=0.0010)
    assert lots == pytest.approx(1.0) and risk == pytest.approx(0.01)


def test_symbol_info_without_tick_fields_keeps_the_old_value():
    legacy = NS(point=1e-5, trade_contract_size=100_000, digits=5, trade_stops_level=0,
                volume_min=0.01, volume_step=0.01, volume_max=100)
    assert make_rm(legacy).get_pip_value("EURUSD#") == pytest.approx(1.0)


# ---- the backtester books money through one helper, in account currency ---------------------------------------

def test_money_of_a_price_move_is_in_account_currency_for_a_jpy_pair():
    rm = make_rm(info(point=0.001, tick_size=0.001, tick_value=2 / 3))
    # 0.5 lots, price up 1.500 JPY = 1500 points x $0.667 x 0.5 = $500; the quote-currency formula gave 75,000 JPY
    assert rm.move_value("USDJPY#", 1.500, 0.5) == pytest.approx(500.0)
    assert rm.move_value("USDJPY#", -0.150, 1.0) == pytest.approx(-100.0)


def test_money_of_a_price_move_matches_contract_size_for_a_usd_quoted_pair():
    rm = make_rm(info(point=1e-5, tick_size=1e-5, tick_value=1.0))
    assert rm.move_value("EURUSD#", 0.0010, 2.0) == pytest.approx(0.0010 * 2.0 * 100_000)


def test_money_of_a_price_move_is_zero_when_the_tick_value_is_unusable():
    rm = make_rm(info(point=0.001, tick_size=0.001, tick_value=0.0))
    assert rm.move_value("USDJPY#", 1.5, 0.5) == 0.0
