"""AUDIT C1: the volatility factor of the stop reads ATR as a fraction of price, so it does not depend on the symbol's price scale.
`vol_threshold` is a fraction of price (0.0005 = 0.05%). `context["vol"]` stays the ATR in price units; `context["price"]` is the close of the
same bar. Without a usable price the volatility factor is skipped (1.0), never computed on a price-unit ATR."""
import numpy as np
import pytest

from src.config import Cfg
from src.risk_controller import RiskController

EUR, JPY, GOLD = "EURUSD#", "USDJPY#", "GOLD#"


def make_rc(overrides=None):
    cfg = Cfg()
    cfg.symbols = [EUR, JPY, GOLD]
    cfg.thompson_sampling.bandit_reset_enabled = False
    if overrides:
        cfg.symbol_overrides = overrides
    return RiskController(cfg)


def vol_factor(rc, sym, atr, price):
    ctx = {"vol": atr, "price": price, "equity": 1000.0, "peak_equity": 1000.0}
    return rc._calculate_rule_scale(sym, ctx)


def test_the_same_relative_volatility_gives_the_same_factor_on_any_price_scale():
    rc = make_rc()
    eur = vol_factor(rc, EUR, 0.0005, 1.08)       # 0.046% of price
    jpy = vol_factor(rc, JPY, 0.057, 150.0)       # 0.038% of price: the old code gave 0.509 here
    assert eur == pytest.approx(1.0)
    assert jpy == pytest.approx(1.0)


def test_high_relative_volatility_shrinks_the_same_amount_on_any_price_scale():
    rc = make_rc()
    # 1% of price: 0.0005 / 0.01 + 0.5 = 0.55
    assert vol_factor(rc, EUR, 0.0108, 1.08) == pytest.approx(0.55)
    assert vol_factor(rc, JPY, 1.5, 150.0) == pytest.approx(0.55)
    assert vol_factor(rc, GOLD, 40.0, 4000.0) == pytest.approx(0.55)


def test_a_per_symbol_vol_threshold_reaches_the_rule_scale():
    rc = make_rc({GOLD: {"vol_threshold": 0.005}})
    # 0.5% of price: global threshold gives 0.0005 / 0.005 + 0.5 = 0.6, the override 0.005 / 0.005 + 0.5 -> capped 1.0
    assert vol_factor(rc, GOLD, 20.0, 4000.0) == pytest.approx(1.0)
    assert vol_factor(rc, EUR, 0.0054, 1.08) == pytest.approx(0.6)


@pytest.mark.parametrize("price", [None, 0.0, -1.0, float("nan")])
def test_without_a_usable_price_the_volatility_factor_is_skipped(price):
    rc = make_rc()
    ctx = {"vol": 0.057, "equity": 1000.0, "peak_equity": 1000.0}
    if price is not None:
        ctx["price"] = price
    assert rc._calculate_rule_scale(JPY, ctx) == pytest.approx(1.0)


def test_the_contextual_bandit_gets_the_relative_volatility(monkeypatch):
    rc = make_rc()
    rc.cfg.thompson_sampling.contextual_enabled = True
    rc = RiskController(rc.cfg)
    seen = {}

    def fake_sample(x):
        seen.setdefault("x", []).append(np.array(x, dtype=float))
        return 0

    for sym in (EUR, JPY):
        st = rc.symbol_states[sym]
        monkeypatch.setattr(st.contextual_bandit, "sample_arm", fake_sample)
    rc.get_params(EUR, {"vol": 0.0005, "price": 1.08, "ensemble_auc": 0.6})
    rc.get_params(JPY, {"vol": 0.057, "price": 150.0, "ensemble_auc": 0.6})
    eur_x0, jpy_x0 = seen["x"][0][0], seen["x"][1][0]
    assert eur_x0 == pytest.approx(0.0005 / 1.08 / 0.0005)
    assert jpy_x0 == pytest.approx(0.057 / 150.0 / 0.0005)
    assert abs(eur_x0 - jpy_x0) < 0.2


def test_the_contextual_bandit_gets_zero_volatility_without_a_price(monkeypatch):
    rc = make_rc()
    rc.cfg.thompson_sampling.contextual_enabled = True
    rc = RiskController(rc.cfg)
    seen = {}
    monkeypatch.setattr(rc.symbol_states[JPY].contextual_bandit, "sample_arm", lambda x: seen.setdefault("x", np.array(x)) is None or 0)
    rc.get_params(JPY, {"vol": 0.057, "ensemble_auc": 0.6})
    assert seen["x"][0] == 0.0


def test_the_symbol_processor_passes_the_decision_bar_close_as_price():
    import inspect
    import src.symbol_processor as sp
    import backtester
    assert '"price": float(last_closed_price)' in inspect.getsource(sp.SymbolProcessor._make_trade_decision)
    assert '"price": float(current_row["close"])' in inspect.getsource(backtester.HybridBacktester._process_bar)


def test_a_missing_price_warns_once_per_symbol():
    from loguru import logger
    rc = make_rc()
    lines = []
    sink = logger.add(lambda m: lines.append(str(m)), format="{level} {message}", level="WARNING")
    try:
        for _ in range(3):
            rc._calculate_rule_scale(JPY, {"vol": 0.057, "equity": 1000.0, "peak_equity": 1000.0})
        rc._calculate_rule_scale(EUR, {"vol": 0.0005, "equity": 1000.0, "peak_equity": 1000.0})
    finally:
        logger.remove(sink)
    jpy = [x for x in lines if JPY in x and "No usable price" in x]
    eur = [x for x in lines if EUR in x and "No usable price" in x]
    assert len(jpy) == 1 and len(eur) == 1


# ---- T6: macd_diff is in price units; the contextual input is divided by price ---------------------------------

def _x_for(rc, sym, ctx, monkeypatch):
    seen = {}
    monkeypatch.setattr(rc.symbol_states[sym].contextual_bandit, "sample_arm", lambda x: seen.setdefault("x", np.array(x, dtype=float)) is None or 0)
    rc.get_params(sym, {"ensemble_auc": 0.6, **ctx})
    return seen["x"]


def _ctx_rc():
    rc = make_rc()
    rc.cfg.thompson_sampling.contextual_enabled = True
    return RiskController(rc.cfg)


def test_the_macd_input_is_the_same_on_any_price_scale(monkeypatch):
    rc = _ctx_rc()
    eur = _x_for(rc, EUR, {"vol": 0.0005, "price": 1.08, "macd_diff": 1e-4 * 1.08}, monkeypatch)[6]
    jpy = _x_for(rc, JPY, {"vol": 0.057, "price": 150.0, "macd_diff": 1e-4 * 150.0}, monkeypatch)[6]
    assert eur == pytest.approx(0.1) and jpy == pytest.approx(0.1)


def test_the_macd_input_is_zero_without_a_price(monkeypatch):
    rc = _ctx_rc()
    assert _x_for(rc, JPY, {"vol": 0.057, "macd_diff": 0.02}, monkeypatch)[6] == 0.0
