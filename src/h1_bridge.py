"""File bridge between the H1 strategy (Python) and the MQL5 EA (H1Bridge.mq5 in mql5/).

EA -> Python: MQL5/Files/h1_bars_<SYMBOL>.csv with the last closed H1 bars (time,open,high,low,close,volume,spread;
server time, the forming bar is excluded). Python -> EA: MQL5/Files/h1_signal_<SYMBOL>.json, written atomically.
"""
import json
import os
import tempfile
from typing import Optional

import pandas as pd

from src.h1_strategy import HORIZON, Decision

MAX_OVERLAP_DIFF = 0.05  # price units; larger means the EA file is a different symbol/feed than the cache
MAX_GAP = pd.Timedelta("5D")  # a longer hole in the cache means missed bars (bot was off); refuse to trade on it


def load_bars_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"], index_col="time").sort_index()
    return df[~df.index.duplicated(keep="last")]


def merge_bars(cache: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Append bars newer than the cache end. Fixed origin: the cache start never moves."""
    overlap = cache.index.intersection(new.index)
    if len(overlap):
        diff = (cache.loc[overlap, "close"] - new.loc[overlap, "close"]).abs().max()
        if diff > MAX_OVERLAP_DIFF:
            raise ValueError(f"EA bars disagree with the cache on {len(overlap)} overlapping bars (max diff {diff})")
    fresh = new[new.index > cache.index[-1]]
    if fresh.empty:
        return cache
    if fresh.index[0] - cache.index[-1] > MAX_GAP:
        raise ValueError(f"gap between cache end {cache.index[-1]} and first new bar {fresh.index[0]}")
    return pd.concat([cache, fresh[cache.columns]])


def atomic_write(path: str, text: str) -> None:
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def signal_json(dec: Decision, lot: float) -> str:
    side = {1: "BUY", -1: "SELL", 0: "NONE"}[dec.side]
    return json.dumps({
        "id": dec.bar_time.strftime("%Y-%m-%d %H:%M:%S"),  # open time of the signal bar; strictly increasing
        "bar_close": (dec.bar_time + pd.Timedelta("1h")).strftime("%Y-%m-%d %H:%M:%S"),
        "side": side,
        "stop_distance": round(dec.stop_distance, 5) if dec.side else 0.0,
        "hold_bars": HORIZON,
        "lot": lot,
        "p_long": round(dec.p_long, 4),
        "p_short": round(dec.p_short, 4),
    })


def last_processed(state_path: str) -> Optional[pd.Timestamp]:
    if not os.path.exists(state_path):
        return None
    with open(state_path) as f:
        return pd.Timestamp(json.load(f)["last_bar"])


def save_processed(state_path: str, bar_time: pd.Timestamp) -> None:
    atomic_write(state_path, json.dumps({"last_bar": str(bar_time)}))
