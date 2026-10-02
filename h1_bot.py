"""USDJPY H1 24-bar bot (signal side). Reads closed bars exported by the MT5 EA, decides once per new bar, writes the
signal file the EA acts on. It never talks to the broker itself.

    python h1_bot.py --files-dir "<MT5 data folder>/MQL5/Files" --symbol USDJPY#

The EA (mql5/H1Bridge.mq5) owns orders, the broker-side stop, the 24-bar exit and the kill switch.
"""
import argparse
import os
import shutil
import time

import pandas as pd
from loguru import logger  # type: ignore

from src import h1_bridge as br
from src.h1_strategy import H1Strategy

ORIGIN = "data/historical_data/{base}_H1.csv"  # fixed-origin history the models were validated on
LIVE_CACHE = "data/live/{base}_H1.csv"
LOT = 0.01


def process(files_dir: str, symbol: str, strat: H1Strategy, state_path: str, decisions_csv: str) -> bool:
    """One polling step. Returns True if a new bar was decided."""
    base = symbol.replace("#", "")
    cache_path = LIVE_CACHE.format(base=base)
    if not os.path.exists(cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        shutil.copy(ORIGIN.format(base=base), cache_path)
    bars_path = os.path.join(files_dir, f"h1_bars_{symbol}.csv")
    if not os.path.exists(bars_path):
        return False
    cache = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    cache.index = cache.index.tz_localize(None) if cache.index.tz is not None else cache.index
    merged = br.merge_bars(cache, br.load_bars_file(bars_path))
    last = br.last_processed(state_path)
    if merged.index[-1] == cache.index[-1] and last == merged.index[-1]:
        return False
    if len(merged) != len(cache):
        br.atomic_write(cache_path, merged.to_csv())
    if last is not None and merged.index[-1] <= last:
        return False
    dec = strat.decide(merged)
    br.atomic_write(os.path.join(files_dir, f"h1_signal_{symbol}.json"), br.signal_json(dec, LOT))
    br.save_processed(state_path, dec.bar_time)
    os.makedirs(os.path.dirname(decisions_csv), exist_ok=True)
    row = pd.DataFrame([{"bar_time": dec.bar_time, "side": dec.side, "p_long": dec.p_long, "p_short": dec.p_short,
                         "atr": dec.atr, "stop_distance": dec.stop_distance, "reason": dec.reason}])
    row.to_csv(decisions_csv, mode="a", header=not os.path.exists(decisions_csv), index=False)
    logger.info(f"{symbol} bar {dec.bar_time}: side={dec.side} pl={dec.p_long:.3f} ps={dec.p_short:.3f} {dec.reason}")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--files-dir", required=True)
    ap.add_argument("--symbol", default="USDJPY#")
    ap.add_argument("--poll", type=int, default=15)
    a = ap.parse_args()
    strat = H1Strategy()
    state = f"data/live/{a.symbol.replace('#', '')}_state.json"
    decisions = f"results/h1_decisions_{a.symbol.replace('#', '')}.csv"
    logger.info(f"H1 bot watching {a.files_dir} for {a.symbol} (signals only, no orders from Python)")
    while True:
        try:
            process(a.files_dir, a.symbol, strat, state, decisions)
        except Exception as e:  # keep running; the EA ignores stale signals and the stop lives at the broker
            logger.exception(f"cycle failed: {e}")
        time.sleep(a.poll)
