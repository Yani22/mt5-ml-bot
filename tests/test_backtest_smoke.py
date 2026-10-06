"""End to end on a synthetic series with a REAL Ensemble (the only test that fits one): the walk-forward loop finishes, trades only
after the first training window, keeps equity finite and writes nothing under models/."""
import os

import numpy as np
import pandas as pd

from src.backtest_data import BacktestFrame
from src.backtest_models import WalkForwardModels, make_fit_fn
from tests.test_backtest_walkforward import Controller, make_bt

N, TRAIN, EVERY, HORIZON = 1700, 1100, 300, 6   # a fit needs at least MIN_SAMPLES_FOR_FIT (1000) rows


class Eager(Controller):
    def get_params(self, sym, context):
        params = super().get_params(sym, context)
        params.update(min_prob_long=0.5, min_prob_short=0.5)
        return params


def test_a_real_ensemble_walk_forward_run_trades_only_after_the_first_window(monkeypatch, tmp_path):
    bt = make_bt(monkeypatch, tmp_path)
    bt.risk_controller = Eager()
    cfg = bt.cfg
    cfg.models = [{"name": "lgbm", "defaults": {"n_estimators": 20, "max_depth": 3}}]
    cfg.cv_samples_per_split = 200
    cfg.symbol_overrides = {"USDJPY#": {"min_ensemble_auc": 0.0}}
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-05", periods=N, freq="5min", tz="UTC")
    close = 150 + np.cumsum(0.02 * np.sin(np.arange(N) / 9.0) + rng.normal(0, 0.01, N))
    bars = pd.DataFrame({"open": close, "high": close + 0.03, "low": close - 0.03, "close": close, "spread": 2.0}, index=idx)
    X = pd.DataFrame({"atr_14": 0.10, "f1": np.sin(np.arange(N) / 9.0), "f2": rng.normal(size=N)}, index=idx)
    future = pd.Series(close, index=idx).shift(-HORIZON) - pd.Series(close, index=idx)
    y_long, y_short = (future > 0).astype(int), (future < 0).astype(int)
    keep = future.notna()
    frame = BacktestFrame(bars=bars[keep], X=X[keep], y_long=y_long[keep], y_short=y_short[keep])
    models = WalkForwardModels(frame.X, frame.y_long, frame.y_short, frame.bars["close"], TRAIN, EVERY, HORIZON,
                               make_fit_fn(cfg, "USDJPY#", None, 0.0))
    bt._process_bar("USDJPY#", frame, models)
    bt._force_close_open_positions("USDJPY#", frame.bars)

    assert bt.positions and all(pd.Timestamp(p.entry_time) > idx[TRAIN] for p in bt.positions)
    assert np.isfinite(bt.equity) and len(models.fits) >= 2
    assert not os.path.exists("models")                      # dry_run: nothing is saved
