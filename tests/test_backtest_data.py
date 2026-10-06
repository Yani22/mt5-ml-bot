import numpy as np
import pandas as pd
import pytest

from src import backtest_data
from src.config import Cfg, FeatureCfg


def write(tmp_path, n=50, spread=True):
    idx = pd.date_range("2026-01-05", periods=n, freq="5min", tz="UTC")
    m5 = pd.DataFrame({"open": np.arange(n) + 100.0, "high": 0, "low": 0, "close": 0, "volume": 1}, index=idx)
    m5.to_csv(tmp_path / "USDJPY_M5.csv")
    if spread:
        pd.DataFrame({"spread": np.arange(n) + 1.0}, index=idx).to_csv(tmp_path / "USDJPY_M5_spread.csv")
    return idx


def fake_training(seen, idx):
    def get_training_data(cfg, sym, **kw):
        seen["data_source"] = cfg.data_source
        seen["load_all_data"] = kw.get("load_all_data")
        data = pd.DataFrame({"close": 1.0, "high": 1.0, "low": 1.0}, index=idx[5:])
        return data, data[[]].assign(f=1.0), pd.Series(1, index=data.index), pd.Series(0, index=data.index)
    return get_training_data


def test_bars_carry_the_csv_open_and_the_spread_in_points(monkeypatch, tmp_path):
    idx, seen = write(tmp_path), {}
    monkeypatch.setattr(backtest_data, "get_training_data", fake_training(seen, idx))
    cfg = Cfg()
    cfg.history_bars = 45000
    frame = backtest_data.load_backtest_frame(cfg, "USDJPY#", FeatureCfg(), 12, 0.0002, raw_dir=str(tmp_path))
    assert list(frame.bars.index) == list(frame.X.index) == list(idx[5:])
    assert frame.bars["open"].iloc[0] == 105.0
    assert frame.bars["spread"].iloc[0] == 6.0
    assert seen["load_all_data"] is True and cfg.history_bars == 45000   # every bar is loaded; the shared config is untouched


def test_a_missing_spread_file_gives_nan_spreads(monkeypatch, tmp_path):
    idx, seen = write(tmp_path, spread=False), {}
    monkeypatch.setattr(backtest_data, "get_training_data", fake_training(seen, idx))
    frame = backtest_data.load_backtest_frame(Cfg(), "USDJPY#", FeatureCfg(), 12, 0.0002, raw_dir=str(tmp_path))
    assert frame.bars["spread"].isna().all()


def test_the_frame_is_always_read_from_csv_even_when_the_config_says_mt5(monkeypatch, tmp_path):
    idx, seen = write(tmp_path), {}
    monkeypatch.setattr(backtest_data, "get_training_data", fake_training(seen, idx))
    cfg = Cfg()
    cfg.data_source = "mt5"
    backtest_data.load_backtest_frame(cfg, "USDJPY#", FeatureCfg(), 12, 0.0002, raw_dir=str(tmp_path))
    assert seen["data_source"] == "csv" and cfg.data_source == "mt5"      # get_training_data reads cfg.data_source, not `source`


def test_a_bar_without_an_open_is_an_error_not_a_nan_fill(monkeypatch, tmp_path):
    idx, seen = write(tmp_path, n=50), {}
    pd.read_csv(tmp_path / "USDJPY_M5.csv", index_col=0).iloc[:30].to_csv(tmp_path / "USDJPY_M5.csv")   # the csv stops early
    monkeypatch.setattr(backtest_data, "get_training_data", fake_training(seen, idx))
    with pytest.raises(ValueError, match="open"):
        backtest_data.load_backtest_frame(Cfg(), "USDJPY#", FeatureCfg(), 12, 0.0002, raw_dir=str(tmp_path))


def fake_training_with_warmup(idx, warmup=5):
    """The real pipeline: X and the labels cover every bar (leading rows have NaN features), `data` is dropna'd and starts later."""
    def get_training_data(cfg, sym, **kw):
        data = pd.DataFrame({"close": 1.0, "high": 1.0, "low": 1.0}, index=idx[warmup:])
        X = pd.DataFrame({"f": 1.0}, index=idx)
        X.iloc[:warmup] = float("nan")
        return data, X, pd.Series(1, index=idx), pd.Series(0, index=idx)
    return get_training_data


def test_bars_x_and_labels_share_one_index_when_the_features_have_a_warmup(monkeypatch, tmp_path):
    """The backtester reads bars and X by the same position. On USDJPY# the bars started 600 rows after X (the feature warm-up), so
    every fill and exit came from bars 50 hours after the decision bar, and the loop ran past the end of the bars."""
    idx = write(tmp_path)
    monkeypatch.setattr(backtest_data, "get_training_data", fake_training_with_warmup(idx))
    frame = backtest_data.load_backtest_frame(Cfg(), "USDJPY#", FeatureCfg(), 12, 0.0002, raw_dir=str(tmp_path))
    assert frame.bars.index.equals(frame.X.index) and frame.X.index.equals(frame.y_long.index) and frame.X.index.equals(frame.y_short.index)
    assert frame.bars.index[0] == frame.X.index[0] and len(frame.bars) == len(frame.X)
    assert not frame.X.isna().any().any()           # the warm-up rows with NaN features are not traded on


def test_a_misaligned_pipeline_result_is_an_error_not_a_shifted_backtest(monkeypatch, tmp_path):
    idx = write(tmp_path)

    def broken(cfg, sym, **kw):
        data = pd.DataFrame({"close": 1.0, "high": 1.0, "low": 1.0}, index=idx[5:])
        return data, pd.DataFrame({"f": 1.0}, index=idx[10:]), pd.Series(1, index=idx[:-3]), pd.Series(0, index=idx)
    monkeypatch.setattr(backtest_data, "get_training_data", broken)
    with pytest.raises(ValueError, match="index"):
        backtest_data.load_backtest_frame(Cfg(), "USDJPY#", FeatureCfg(), 12, 0.0002, raw_dir=str(tmp_path))
