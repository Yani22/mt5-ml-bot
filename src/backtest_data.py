"""The backtester's bars: the live feature pipeline over EVERY bar on disk (`get_training_data` caps at `history_bars`), plus the
`open` and per-bar spread that pipeline does not return."""
import copy
import os
from dataclasses import dataclass

import pandas as pd
from loguru import logger

from src.utils import get_training_data


@dataclass
class BacktestFrame:
    bars: pd.DataFrame
    X: pd.DataFrame
    y_long: pd.Series
    y_short: pd.Series


def _read(path):
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame


def load_backtest_frame(cfg, symbol, feature_cfg, horizon, min_pct_change, raw_dir=None):
    """`bars["spread"]` is in POINTS, as the spread file has it (the backtester converts with the symbol's point). A missing spread
    file gives NaN spreads, which the fills replace with `trading_costs`. The feature bars always come from the csv files (the config's `data_source` is ignored here); `get_training_data` loads the H1
    and inter-market context itself."""
    local = copy.copy(cfg)               # `get_training_data` reads cfg.data_source (its `source` argument is ignored): force the csv files
    local.data_source = "csv"            # the `open` and spread below come from them; the shared config is untouched
    data, X, y_long, y_short = get_training_data(
        local, symbol, feature_cfg=feature_cfg, source="csv", load_all_data=True, min_pct_change=min_pct_change,
        return_long_short_labels=True, prediction_horizon=horizon)
    # `data` is the pipeline's rows with no NaN; X and the labels still hold the warm-up rows before them (600 on USDJPY# with a 50-bar H1
    # EMA). The backtester reads bars and X by the same position, so all four must share one index, or every fill comes from another time.
    for name, part in (("X", X), ("y_long", y_long), ("y_short", y_short)):
        if not data.index.isin(part.index).all():
            raise ValueError(f"[{symbol}] the bars' index is not inside the index of {name}: the backtest would read prices of another time")
    X, y_long, y_short = X.loc[data.index], y_long.loc[data.index], y_short.loc[data.index]
    raw_dir = raw_dir or cfg.fetch.raw_data_dir
    stem = symbol.rstrip("#")
    m5 = _read(os.path.join(raw_dir, f"{stem}_M5.csv"))
    bars = data[["high", "low", "close"]].copy()
    bars.insert(0, "open", m5["open"].reindex(bars.index))
    missing = int(bars["open"].isna().sum())
    if missing:
        raise ValueError(f"[{symbol}] {missing} of {len(bars)} bars have no open in {stem}_M5.csv: the csv and the feature bars differ")
    spread_path = os.path.join(raw_dir, f"{stem}_M5_spread.csv")
    if os.path.exists(spread_path):
        bars["spread"] = _read(spread_path)["spread"].reindex(bars.index)
    else:
        logger.warning(f"[{symbol}] No {stem}_M5_spread.csv: every bar uses the trading_costs spread.")
        bars["spread"] = float("nan")
    return BacktestFrame(bars=bars, X=X, y_long=y_long, y_short=y_short)
