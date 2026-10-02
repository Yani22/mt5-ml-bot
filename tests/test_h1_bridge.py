"""File-bridge behaviour: merging EA bars, signal format, and one full polling cycle with a stub strategy."""
import json

import numpy as np
import pandas as pd
import pytest

import h1_bot
from src import h1_bridge as br
from src.h1_strategy import Decision


def _bars(start, n, base=150.0):
    idx = pd.date_range(start, periods=n, freq="1h")
    c = base + np.arange(n) * 0.01
    return pd.DataFrame({"open": c, "high": c + 0.05, "low": c - 0.05, "close": c, "volume": 100, "spread": 8}, index=idx)


def test_merge_appends_only_newer_bars_and_keeps_origin():
    cache = _bars("2024-01-02", 100).drop(columns="spread")
    new = _bars("2024-01-02", 105)  # overlaps the cache, 5 new bars
    out = br.merge_bars(cache, new)
    assert len(out) == 105 and out.index[0] == cache.index[0]
    assert list(out.columns) == list(cache.columns)
    assert br.merge_bars(out, new).equals(out)  # idempotent


def test_merge_rejects_mismatched_feed_and_gaps():
    cache = _bars("2024-01-02", 100).drop(columns="spread")
    wrong = _bars("2024-01-02", 105, base=1.1)  # e.g. a different symbol
    with pytest.raises(ValueError, match="disagree"):
        br.merge_bars(cache, wrong)
    late = _bars("2024-02-01", 5, base=150.0)
    with pytest.raises(ValueError, match="gap"):
        br.merge_bars(cache, late)


def test_signal_json_format():
    t = pd.Timestamp("2026-10-05 10:00:00")
    buy = json.loads(br.signal_json(Decision(t, 1, 0.62, 0.40, 0.21, 0.42), 0.01))
    assert buy["side"] == "BUY" and buy["id"] == "2026-10-05 10:00:00" and buy["bar_close"] == "2026-10-05 11:00:00"
    assert buy["stop_distance"] == 0.42 and buy["hold_bars"] == 24 and buy["lot"] == 0.01
    none = json.loads(br.signal_json(Decision(t, 0, 0.5, 0.5, 0.21, 0.42), 0.01))
    assert none["side"] == "NONE" and none["stop_distance"] == 0.0


class _Stub:
    def __init__(self):
        self.calls = []

    def decide(self, h1):
        self.calls.append(h1.index[-1])
        return Decision(h1.index[-1], -1, 0.3, 0.7, 0.2, 0.4)


def test_process_cycle_writes_signal_once_per_new_bar(tmp_path, monkeypatch):
    base = tmp_path / "hist"
    base.mkdir()
    monkeypatch.setattr(h1_bot, "ORIGIN", str(base / "{base}_H1.csv"))
    monkeypatch.setattr(h1_bot, "LIVE_CACHE", str(tmp_path / "live" / "{base}_H1.csv"))
    _bars("2024-01-02", 200).drop(columns="spread").to_csv(base / "USDJPY_H1.csv")
    files = tmp_path / "files"
    files.mkdir()
    state, dec_csv, stub = str(tmp_path / "state.json"), str(tmp_path / "dec.csv"), _Stub()
    # no EA file yet: nothing happens
    assert h1_bot.process(str(files), "USDJPY#", stub, state, dec_csv) is False
    _bars("2024-01-02", 203).to_csv(files / "h1_bars_USDJPY#.csv", index_label="time")
    assert h1_bot.process(str(files), "USDJPY#", stub, state, dec_csv) is True
    sig = json.loads((files / "h1_signal_USDJPY#.json").read_text())
    assert sig["side"] == "SELL" and sig["id"] == "2024-01-10 10:00:00"
    # same bars again: no duplicate decision
    assert h1_bot.process(str(files), "USDJPY#", stub, state, dec_csv) is False
    assert len(stub.calls) == 1
    # one more closed bar arrives -> exactly one more decision
    _bars("2024-01-02", 204).to_csv(files / "h1_bars_USDJPY#.csv", index_label="time")
    assert h1_bot.process(str(files), "USDJPY#", stub, state, dec_csv) is True and len(stub.calls) == 2
    assert len(pd.read_csv(dec_csv)) == 2
    # the stub saw the full fixed-origin history (starts at the cache origin, not a sliding window)
    assert pd.read_csv(tmp_path / "live" / "USDJPY_H1.csv", index_col=0).index[0].startswith("2024-01-02")
