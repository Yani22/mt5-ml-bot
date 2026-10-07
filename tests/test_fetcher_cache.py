"""fetcher.py must add to the cache, not replace it: a run used to overwrite `<SYM>_<TF>.csv` with the last `count` bars (destroying a
longer history) and kept the bar that was still forming."""
import numpy as np
import pandas as pd

import fetcher

DTYPE = [("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"), ("close", "f8"), ("tick_volume", "i8")]


def rates(first, n):
    base = int(pd.Timestamp(first, tz="UTC").timestamp())
    return np.array([(base + 300 * i, 1.0 + i, 1.5 + i, 0.5 + i, 1.2 + i, 10 + i) for i in range(n)], dtype=DTYPE)


def write_old(tmp_path, first, n):
    idx = pd.date_range(first, periods=n, freq="5min", tz="UTC", name="time")
    old = pd.DataFrame({"open": 9.0, "high": 9.5, "low": 8.5, "close": 9.2, "volume": 5}, index=idx)
    old.to_csv(tmp_path / "EURUSD_M5.csv")
    return old


def test_older_bars_in_the_cache_survive_a_fetch(tmp_path, monkeypatch):
    old = write_old(tmp_path, "2024-01-01 00:00", 10)
    monkeypatch.setattr(fetcher.mt5, "copy_rates_from_pos", lambda *a: rates("2024-02-01 00:00", 4), raising=False)
    fetcher.fetch_and_save_bars("EURUSD#", "M5", 4, str(tmp_path))
    saved = pd.read_csv(tmp_path / "EURUSD_M5.csv", index_col=0, parse_dates=True)
    assert len(saved) == len(old) + 3                     # the 10 old bars and 3 closed new ones
    assert saved.index[0] == old.index[0]


def test_the_forming_bar_is_not_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(fetcher.mt5, "copy_rates_from_pos", lambda *a: rates("2024-02-01 00:00", 4), raising=False)
    fetcher.fetch_and_save_bars("EURUSD#", "M5", 4, str(tmp_path))
    saved = pd.read_csv(tmp_path / "EURUSD_M5.csv", index_col=0, parse_dates=True)
    assert len(saved) == 3 and saved.index[-1] == pd.Timestamp("2024-02-01 00:10", tz="UTC")


def test_a_bar_fetched_again_replaces_the_cached_one(tmp_path, monkeypatch):
    write_old(tmp_path, "2024-02-01 00:00", 3)
    monkeypatch.setattr(fetcher.mt5, "copy_rates_from_pos", lambda *a: rates("2024-02-01 00:00", 4), raising=False)
    fetcher.fetch_and_save_bars("EURUSD#", "M5", 4, str(tmp_path))
    saved = pd.read_csv(tmp_path / "EURUSD_M5.csv", index_col=0, parse_dates=True)
    assert len(saved) == 3 and saved["open"].iloc[0] == 1.0     # the fresh values, no duplicate stamps
