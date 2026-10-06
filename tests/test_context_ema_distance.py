"""The risk context's `dist_from_ema_200` is real: the live decision and the backtester both compute it from the close series
(it is not a model column, so no saved model changes), they agree on the same bar, and a later bar never changes it."""
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

import tests.test_backtest_walkforward as twf
from src.features import ema_distance, ema_distance_at
from src.symbol_processor import SymbolProcessor

N = 400


def closes(n=N):
    idx = pd.date_range("2026-01-05", periods=n, freq="5min", tz="UTC")
    return pd.Series(150.0 + np.linspace(0, 8, n) + np.sin(np.arange(n) / 7.0), index=idx)


class Stop(Exception):
    pass


def live_context(close):
    """The context `_make_trade_decision` hands to the risk controller, for a frame ending at `close`'s last bar."""
    idx = close.index
    data = pd.DataFrame({"open": close, "high": close + 0.02, "low": close - 0.02, "close": close, "spread": 5}, index=idx)
    X = pd.DataFrame({"atr_14": 0.1, "adx": 20.0, "macd_diff": 0.01, "volatility_10": 0.001}, index=idx)
    seen = {}

    def get_params(symbol, context):
        seen.update(context)
        raise Stop

    sp = object.__new__(SymbolProcessor)
    sp.symbol = "USDJPY#"
    sp.ens_long = sp.ens_short = NS(ensemble_cv_auc_=0.56, predict_proba=lambda f: pd.Series([0.5]))
    sp.monitor = NS(current_equity=1000.0, peak_equity=1000.0)
    sp.risk_controller = NS(get_params=get_params)
    with pytest.raises(Stop):
        sp._make_trade_decision(data, X)
    return seen


def backtest_contexts(monkeypatch, tmp_path, close):
    monkeypatch.setattr(twf, "N", len(close))
    bt = twf.make_bt(monkeypatch, tmp_path)
    seen = []

    class Spy(twf.Controller):
        def get_params(self, sym, context):
            seen.append(dict(context))
            return super().get_params(sym, context)

    bt.risk_controller = Spy()
    frame, models, _ = twf.build()
    frame.bars["close"] = close.to_numpy()
    twf.run(bt, frame, models)
    return seen, frame.X.index


def test_the_live_context_carries_the_distance_from_the_ema(monkeypatch):
    close = closes()
    got = live_context(close)["dist_from_ema_200"]
    assert got != 0.0
    assert got == pytest.approx(ema_distance(close).iloc[-1])


def test_the_backtester_context_carries_it_too(monkeypatch, tmp_path):
    close = closes()
    seen, index = backtest_contexts(monkeypatch, tmp_path, close)
    by_time = {c["bar_time"]: c["dist_from_ema_200"] for c in seen}
    assert any(v != 0.0 for v in by_time.values())
    when = max(by_time)
    assert by_time[when] == pytest.approx(ema_distance(close).loc[when])


def test_live_and_the_backtester_give_the_same_value_for_the_same_bar(monkeypatch, tmp_path):
    close = closes()
    seen, _ = backtest_contexts(monkeypatch, tmp_path, close)
    c = seen[-1]
    live = live_context(close[:close.index.get_loc(c["bar_time"]) + 1])["dist_from_ema_200"]
    assert live == pytest.approx(c["dist_from_ema_200"])


def test_a_bar_appended_later_does_not_change_the_value_at_an_earlier_bar():
    close = closes()
    when = close.index[299]
    assert ema_distance_at(close, when) == pytest.approx(ema_distance_at(close[:300], when))


def test_before_200_bars_the_value_is_zero_not_a_made_up_number():
    close = closes(150)
    assert ema_distance_at(close, close.index[-1]) == 0.0


def test_a_repeated_timestamp_does_not_crash():
    close = closes()
    doubled = pd.concat([close, close.iloc[[-1]]])
    assert np.isfinite(ema_distance_at(doubled, close.index[-1]))
