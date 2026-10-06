"""When an enabled context (the H1 join or inter-market data) had no history, `load_cached` flipped `cfg.context_features.*.enabled`
to False for the rest of the process and trained on the primary features alone, and `get_training_data` carried on with no context.
The live bot builds the context columns, so a model trained without them does not match what it is asked to predict on (and the
retrain child's model can be promoted into the running processor). Training now refuses instead, and never edits the config."""
from types import SimpleNamespace as NS

import pandas as pd
import pytest

import src.utils as utils
from src import retraining
from src.data_manager import DataManager

SYM = "USDJPY#"


def frame(n, freq="5min"):
    idx = pd.date_range("2026-01-05", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 1}, index=idx)


def cfg(mta=True, inter_market=False):
    return NS(timeframe="M5", data_source="csv", history_bars=100, prediction_horizon=3,
              context_features=NS(mta=NS(enabled=mta, timeframe="H1"), inter_market=NS(enabled=inter_market, symbol="USDX")))


def manager(c, h1=None, usdx=None, loaded=None):
    """`load_cached`'s data manager with canned history per (symbol, timeframe); `h1` / `usdx` None means no history."""
    d = object.__new__(DataManager)
    d.cfg = c
    loaded = [] if loaded is None else loaded

    def load_local_history(symbol, timeframe, count=None):
        loaded.append((symbol, timeframe))
        if symbol == "USDX":
            return usdx if usdx is not None else pd.DataFrame()
        return frame(50) if timeframe == "M5" else (h1 if h1 is not None else pd.DataFrame())

    d.load_local_history = load_local_history
    d._build_features_and_labels = lambda df, *a, **k: (df[["close"]], pd.Series(1, index=df.index))
    return d


# ---- load_cached --------------------------------------------------------------------------------------------------

def test_load_cached_refuses_when_the_enabled_h1_context_has_no_history_and_leaves_the_config_alone():
    c = cfg(mta=True)
    data, X, y = manager(c).load_cached(SYM, None)
    assert data.empty and X.empty and y.empty
    assert c.context_features.mta.enabled is True


def test_load_cached_refuses_when_the_enabled_inter_market_context_has_no_history_and_leaves_the_config_alone():
    c = cfg(mta=False, inter_market=True)
    data, X, y = manager(c).load_cached(SYM, None)
    assert data.empty and X.empty and y.empty
    assert c.context_features.inter_market.enabled is True


def test_load_cached_with_the_context_present_returns_data():
    data, X, y = manager(cfg(mta=True), h1=frame(20, "1h")).load_cached(SYM, None)
    assert len(data) == 50 and len(X) == 50


def test_load_cached_with_the_context_off_never_asks_for_it():
    loaded = []
    data, _, _ = manager(cfg(mta=False), loaded=loaded).load_cached(SYM, None)
    assert len(data) == 50 and all(tf == "M5" for _, tf in loaded)


# ---- the retrain child --------------------------------------------------------------------------------------------

def test_the_child_trains_nothing_when_load_cached_comes_back_empty(monkeypatch):
    empty = pd.DataFrame({"close": pd.Series(dtype=float)})   # what an empty cache looks like to the child
    monkeypatch.setattr(retraining, "DataManager", lambda c: NS(load_cached=lambda *a, **k: (empty, empty, empty)))
    monkeypatch.setattr(retraining, "generate_long_short_labels", lambda *a: (pd.Series(dtype=int), pd.Series(dtype=int)))
    monkeypatch.setattr(retraining, "load_ensemble", lambda *a, **k: NS(ensemble_cv_auc_=0.6))
    monkeypatch.setattr(retraining, "discard_staged_ensemble", lambda *a: None)
    calls = []
    monkeypatch.setattr(retraining, "safe_retrain_ensemble", lambda *a, **k: calls.append(k))
    c = NS(prediction_horizon=12, retraining_window_bars=100, risk=NS(min_auc_improvement=0.005),
           get_symbol_value=lambda sym, key, default=None: default)

    retraining.run_retraining_in_background(c, SYM, NS(min_pct_change=0.0002), False, None, {SYM: None})

    assert calls == []


# ---- get_training_data --------------------------------------------------------------------------------------------

def test_get_training_data_refuses_when_the_enabled_h1_context_has_no_history(monkeypatch):
    monkeypatch.setattr(utils.data_manager, "DataManager", lambda c: manager(c))
    with pytest.raises(RuntimeError, match="H1"):
        utils.get_training_data(cfg(mta=True), SYM, None, source="csv")


def test_get_training_data_refuses_when_the_enabled_inter_market_context_has_no_history(monkeypatch):
    monkeypatch.setattr(utils.data_manager, "DataManager", lambda c: manager(c))
    with pytest.raises(RuntimeError, match="USDX"):
        utils.get_training_data(cfg(mta=False, inter_market=True), SYM, None, source="csv")
