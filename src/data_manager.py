# src/data_manager.py
from __future__ import annotations
import math
import os
import tempfile
import pandas as pd  # type: ignore
from loguru import logger  # type: ignore
from typing import Optional, Tuple

from src.config import Cfg
from src.features import FeatureCfg, build_features
from src.labels import generate_labels

import time
from src.mt5_client import MT5Client


def ensure_dir(path: str):
    if path is None:
        return
    os.makedirs(path, exist_ok=True)


class DataManager:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.raw_data_dir = cfg.fetch.raw_data_dir
        ensure_dir(self.raw_data_dir)

        # Initialize MT5Client and add retry logic for connection
        self.mt5_client = MT5Client(
            os.getenv("MT5_LOGIN"),
            os.getenv("MT5_PASSWORD"),
            os.getenv("MT5_SERVER"),
            os.getenv("MT5_PATH"),
        )
        max_retries = 3
        for i in range(max_retries):
            if self.mt5_client.connect():
                logger.info("MT5Client connected successfully in DataManager.")
                break
            else:
                logger.warning(f"MT5Client connection attempt {i + 1}/{max_retries} failed in DataManager. Retrying in 5 seconds...")
                time.sleep(5)
        if not self.mt5_client.is_connected():
            logger.error("Failed to connect MT5Client in DataManager after multiple retries.")
            # Optionally, raise an exception or handle this failure more gracefully

    def _local_csv_path(self, symbol: str, timeframe: str) -> str:
        fname = f"{symbol.replace('#', '')}_{timeframe}.csv"
        return os.path.join(self.raw_data_dir, fname)

    def _atomic_write_df(self, df: pd.DataFrame, path: str, fmt: str = "csv"):
        ensure_dir(os.path.dirname(path))
        fd, tmp = tempfile.mkstemp(prefix="tmp_", dir=os.path.dirname(path))
        os.close(fd)
        try:
            if fmt == "csv":
                df.to_csv(tmp, index=True)
            else:
                df.to_parquet(tmp, index=True)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass

    def load_local_history(self, symbol: str, timeframe: str, count: Optional[int] = None) -> pd.DataFrame:
        path = self._local_csv_path(symbol, timeframe)
        if not os.path.exists(path):
            return pd.DataFrame()
        try:
            df = pd.read_csv(path, index_col=0)
        except (pd.errors.ParserError, UnicodeDecodeError):
            logger.warning(f"Pandas C engine failed to parse {path}. Retrying with Python engine.")
            df = pd.read_csv(path, index_col=0, engine='python')
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
        if count is not None and len(df) > count:
            df = df.tail(count)
        return df

    def append_new_bars(self, symbol: str, new_bars: pd.DataFrame, timeframe: Optional[str] = None):
        if not isinstance(new_bars, pd.DataFrame) or new_bars.empty:
            logger.debug(f"[{symbol}] No new bars to append.")
            return
        path = self._local_csv_path(symbol, timeframe if timeframe else self.cfg.timeframe)
        nb = new_bars.copy()
        try:
            nb.index = pd.to_datetime(nb.index)
        except Exception:
            nb.index = pd.to_datetime(nb.index.astype(str))

        if os.path.exists(path):
            existing = pd.read_csv(path, index_col=0)
            try:
                existing.index = pd.to_datetime(existing.index)
            except Exception:
                pass
            combined = pd.concat([existing, nb])
            combined = combined[~combined.index.duplicated(keep='last')].sort_index()
        else:
            combined = nb.sort_index()

        self._atomic_write_df(combined, path, fmt="csv")

    def _fetch_bars_from_mt5_chunked(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        import MetaTrader5 as mt5  # type: ignore
        import datetime  # NEW
        from src.time_utils import timeframe_to_seconds  # NEW

        TF_MAP = {
            "M1": getattr(mt5, "TIMEFRAME_M1", None),
            "M5": getattr(mt5, "TIMEFRAME_M5", None),
            "M15": getattr(mt5, "TIMEFRAME_M15", None),
            "M30": getattr(mt5, "TIMEFRAME_M30", None),
            "H1": getattr(mt5, "TIMEFRAME_H1", None),
            "H4": getattr(mt5, "TIMEFRAME_H4", None),
            "D1": getattr(mt5, "TIMEFRAME_D1", None),
        }
        tf = TF_MAP.get(str(timeframe).upper())
        if tf is None:
            logger.error(f"[{symbol}] Unsupported timeframe: {timeframe}")
            return pd.DataFrame()

        if count is None:
            count = 36000
        try:
            # Use copy_rates_from_pos to get the latest 'count' bars
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, int(count))
            if rates is None or len(rates) == 0:
                try:
                    last_error = mt5.last_error()
                except Exception:
                    last_error = "unavailable"
                logger.warning(f"[{symbol}] MT5 returned no {timeframe} bars (terminal error: {last_error}).")
                return pd.DataFrame()
            if len(rates) < int(count):
                # Raw length, before the forming bar is dropped below. The terminal only serves bars within its chart history.
                logger.info(f"[{symbol}] MT5 returned {len(rates)} {timeframe} bars, {int(count)} were asked for "
                            f"(first {pd.to_datetime(rates[0]['time'], unit='s', utc=True)}, last {pd.to_datetime(rates[-1]['time'], unit='s', utc=True)}): the terminal's 'Max bars in chart' "
                            f"or the broker's history is limiting it.")

            df = pd.DataFrame(rates)
            if "time" not in df.columns:
                logger.warning(f"[{symbol}] fetched data missing 'time' column — returning empty DataFrame")
                return pd.DataFrame()
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.set_index("time").sort_index()
            df.index = df.index.as_unit("us")   # the unit the cached CSV frames load with; mixing units breaks merges under pandas 3
            df = df.rename(columns={"tick_volume": "volume"})

            # Drop the last bar if it's the currently forming one
            # This ensures all data used is from fully closed bars
            if not df.empty:
                df = df.iloc[:-1]

            return df[["open", "high", "low", "close", "volume"]].copy()
        except Exception as e:
            logger.exception(f"[{symbol}] Error fetching bars: {e}")
            return pd.DataFrame()

    def bootstrap_history(self, symbol: str, initial_bars: int, timeframe: Optional[str] = None):
        target_timeframe = timeframe if timeframe else self.cfg.timeframe
        path = self._local_csv_path(symbol, target_timeframe)

        # 1. Load existing full local history
        current_local_history = self.load_local_history(symbol, target_timeframe)

        # 2. Fetch a small chunk of the absolute latest data from MT5 and append it
        # This ensures the local history is fresh before checking its length.
        logger.info(f"[{symbol}] Bootstrapping: Fetching latest 200 bars from MT5 to refresh local history for {target_timeframe}.")
        latest_mt5_bars = self._fetch_bars_from_mt5_chunked(symbol, target_timeframe, 200)
        if not latest_mt5_bars.empty:
            self.append_new_bars(symbol, latest_mt5_bars, timeframe=target_timeframe)
            # Reload the history after appending to get the most up-to-date view
            current_local_history = self.load_local_history(symbol, target_timeframe)
        else:
            logger.warning(f"[{symbol}] Bootstrapping: No latest bars fetched from MT5 for {target_timeframe}. Local history might be outdated.")

        # 3. Check if the total length meets initial_bars. If not, fetch more.
        if len(current_local_history) < initial_bars:
            bars_to_fetch_more = initial_bars - len(current_local_history)
            logger.info(f"[{symbol}] Bootstrapping: Local history for {target_timeframe} still short. Have={len(current_local_history)}, Need={initial_bars}, Fetching={bars_to_fetch_more} more bars.")

            # Fetch the remaining required bars. Start from the end of current_local_history if possible.
            # For simplicity, we'll fetch the total initial_bars again, and append_new_bars will handle duplicates.
            # A more optimized approach would be to fetch from a specific date/time.
            df_more = self._fetch_bars_from_mt5_chunked(symbol, target_timeframe, initial_bars)
            if not df_more.empty:
                self.append_new_bars(symbol, df_more, timeframe=target_timeframe)
                logger.info(f"[{symbol}] Bootstrapped local history to {path} (total {len(self.load_local_history(symbol, target_timeframe))} rows).")
            else:
                logger.warning(f"[{symbol}] Bootstrap failed to fetch additional bars for {target_timeframe}. Local history might still be insufficient.")
        else:
            logger.debug(f"[{symbol}] Local history for {target_timeframe} OK ({len(current_local_history)} rows).")

        # 4. Warn once if the history is still below what this timeframe needs (K13). The terminal only serves bars within its
        # "Max bars in chart" history, so a request for `initial_bars` can silently come back far shorter.
        need = self._bars_needed(initial_bars, target_timeframe)
        have = len(self.load_local_history(symbol, target_timeframe))
        if have < need:
            logger.warning(f"[{symbol}] Local {target_timeframe} history has {have} bars, below the {need} needed "
                           f"({initial_bars} {self.cfg.timeframe} bars span that window). The terminal's 'Max bars in chart' "
                           f"(Tools > Options > Charts) or the broker's history may be capping it: raise it to at least {need} "
                           f"and restart the terminal, or features and training will use a shorter window than configured.")

    def _bars_needed(self, initial_bars: int, timeframe: str) -> int:
        """Bars of `timeframe` that span the same period as `initial_bars` bars of the primary timeframe, never more than the
        `initial_bars` that are asked for (a finer context timeframe cannot be judged against more than was requested)."""
        from src.time_utils import timeframe_to_seconds
        try:
            primary, target = timeframe_to_seconds(self.cfg.timeframe), timeframe_to_seconds(timeframe)
            return max(1, min(int(initial_bars), math.ceil(initial_bars * primary / target)))
        except Exception:
            return int(initial_bars)

    @staticmethod
    def _context_bar_needed(data: pd.DataFrame, timeframe: str):
        """Open time of the newest higher-timeframe bar the context join gives the decision bar (the last row of `data`).

        `add_contextual_features` shifts each context bar to its close and joins backward on the primary bar's OPEN time, so the
        decision bar at 10:55 uses the 09:00 H1 bar and the bar at 11:00 uses the 10:00 one. The last primary bar before the current
        context bar opens is used (not "one context bar earlier"), so a weekend or session gap needs the last bar before the gap.
        None when there is no earlier primary bar to judge by."""
        from src.time_utils import timeframe_to_seconds
        seconds = timeframe_to_seconds(timeframe)
        floor = data.index[-1].floor(f"{seconds}s")
        earlier = data.index[data.index < floor]
        return earlier[-1].floor(f"{seconds}s") if len(earlier) else None

    def fetch_live(self, symbol: str, feature_cfg: FeatureCfg) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # 1. Load the bulk of the history from the local cache first.
        data = self.load_local_history(symbol, self.cfg.timeframe, count=self.cfg.history_bars)

        # 2. Fetch only a small number of recent bars to get the absolute latest data.
        # This is more efficient than fetching the entire history every time.
        recent_data = self._fetch_bars_from_mt5_chunked(symbol, self.cfg.timeframe, 200)  # Fetch last 200 bars

        # 3. Combine and de-duplicate.
        if recent_data.empty:
            # Deciding on the cached history would pair an old bar with a live tick: skip this bar instead.
            last_cached = data.index[-1] if not data.empty else "none"
            logger.warning(f"[{symbol}] No recent bars from MT5; the cache ends at {last_cached} and would be stale. "
                           f"Skipping this bar.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        data = pd.concat([data, recent_data])
        data = data[~data.index.duplicated(keep='last')].sort_index()

        # Save the newly fetched recent data to local history if enabled
        if self.cfg.fetch.save_raw_data_locally:
            # Append only the new recent_data to avoid re-writing entire history
            # The append_new_bars method handles merging with existing data
            self.append_new_bars(symbol, recent_data)

        if data.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # 4. Fetch context features, if enabled
        mta_df = None
        if self.cfg.context_features.mta.enabled:
            # Fetch only a small number of recent bars for MTA data
            mta_recent_data = self._fetch_bars_from_mt5_chunked(symbol, self.cfg.context_features.mta.timeframe, 200)
            if mta_recent_data.empty:
                # The processor rebuilds X from the cached context, so "disabling it for this tick" never held: skip the bar.
                logger.warning(f"[{symbol}] No live MTA data loaded for timeframe {self.cfg.context_features.mta.timeframe}; "
                               f"the cached context may be stale. Skipping this bar.")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            else:
                # Append the newly fetched recent MTA data to local history
                if self.cfg.fetch.save_raw_data_locally:
                    self.append_new_bars(symbol, mta_recent_data, timeframe=self.cfg.context_features.mta.timeframe)
                # Load the full (updated) local history for feature building, and add the bars just fetched: with
                # `save_raw_data_locally` off (or a failed append) the cache alone would stay at its start-up state.
                mta_tf = self.cfg.context_features.mta.timeframe
                mta_df = self.load_local_history(symbol, mta_tf, count=self.cfg.history_bars)
                mta_df = pd.concat([mta_df, mta_recent_data])
                mta_df = mta_df[~mta_df.index.duplicated(keep='last')].sort_index()
                needed = self._context_bar_needed(data, mta_tf)
                if needed is not None and (mta_df.empty or mta_df.index[-1] < needed):
                    have = mta_df.index[-1] if not mta_df.empty else "none"
                    logger.warning(f"[{symbol}] The {mta_tf} context ends at {have} but the decision bar needs the bar opened at {needed}; "
                                   f"the context is stale. Skipping this bar.")
                    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        inter_market_df = None
        if self.cfg.context_features.inter_market.enabled:
            im_sym = self.cfg.context_features.inter_market.symbol
            # Fetch only a small number of recent bars for Inter-Market data
            im_recent_data = self._fetch_bars_from_mt5_chunked(im_sym, self.cfg.timeframe, 200)
            if im_recent_data.empty:
                logger.warning(f"[{symbol}] No live Inter-Market data loaded for symbol {im_sym}; "
                               f"the cached context may be stale. Skipping this bar.")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            else:
                # Append the newly fetched recent Inter-Market data to local history
                if self.cfg.fetch.save_raw_data_locally:
                    self.append_new_bars(im_sym, im_recent_data, timeframe=self.cfg.timeframe)
                # Load the full (updated) local history for feature building
                inter_market_df = self.load_local_history(im_sym, self.cfg.timeframe, count=self.cfg.history_bars)

        # 3. Build features. For live data, labels are not needed.
        X = build_features(data.copy(), feature_cfg, self.cfg, symbol=symbol, mta_df=mta_df, inter_market_df=inter_market_df)

        # Create an empty dataframe for y to match function signature, it's not used in live trading.
        y = pd.DataFrame()

        # 4. Align data with X. This handles any rows dropped from the start by feature engineering.
        # This ensures that we don't truncate the most recent data needed for live decisions.
        common_idx = data.index.intersection(X.index)
        X = X.loc[common_idx]
        data = data.loc[common_idx]

        # The decision bar is the last row; if the feature build dropped the newest bar, the decision would be on an old one.
        if data.empty or data.index[-1] != recent_data.index[-1]:
            logger.warning(f"[{symbol}] Features do not reach the newest bar ({recent_data.index[-1]}); skipping this bar.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        return data, X, y

    def load_cached(self, symbol: str, feature_cfg: FeatureCfg, count: Optional[int] = None, min_pct_change: float = 0.0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        data = self.load_local_history(symbol, self.cfg.timeframe, count=count)
        if data.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Load MTA data if enabled
        mta_df = None
        if self.cfg.context_features.mta.enabled:
            mta_df = self.load_local_history(symbol, self.cfg.context_features.mta.timeframe, count=count)
            if mta_df.empty:
                # Training without the enabled context would give a model whose columns do not match what the live bot builds,
                # and the old "disable it in the shared config" edit was never undone: refuse instead.
                logger.error(f"[{symbol}] No MTA data for timeframe {self.cfg.context_features.mta.timeframe}, but it is enabled; "
                             f"not building training data without it.")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            else:
                logger.info(f"[{symbol}] Successfully loaded MTA data for timeframe {self.cfg.context_features.mta.timeframe}.")

        # Load Inter-Market data if enabled
        inter_market_df = None
        if self.cfg.context_features.inter_market.enabled:
            im_sym = self.cfg.context_features.inter_market.symbol
            inter_market_df = self.load_local_history(im_sym, self.cfg.timeframe, count=count)
            if inter_market_df.empty:
                logger.error(f"[{symbol}] No Inter-Market data for symbol {im_sym}, but it is enabled; "
                             f"not building training data without it.")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            else:
                logger.info(f"[{symbol}] Successfully loaded Inter-Market data for symbol {im_sym}.")

        X, y = self._build_features_and_labels(data, feature_cfg, symbol, min_pct_change, mta_df=mta_df, inter_market_df=inter_market_df)

        # Align X, y, and data by index
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        data = data.loc[common_idx]  # Align data here

        logger.debug(f"[{symbol}] load_cached returning X with shape: {X.shape}")

        return data, X, y

    def _build_features_and_labels(self, df: pd.DataFrame, feature_cfg: FeatureCfg, symbol: str, min_pct_change: float, mta_df: pd.DataFrame | None = None, inter_market_df: pd.DataFrame | None = None) -> Tuple[pd.DataFrame, pd.Series]:
        # Build features and labels
        X = build_features(df.copy(), feature_cfg, self.cfg, symbol=symbol, mta_df=mta_df, inter_market_df=inter_market_df)
        y = generate_labels(df, self.cfg.prediction_horizon, min_pct_change)

        # Align X and y by index
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        logger.debug(f"[{symbol}] _build_features_and_labels returning X with shape: {X.shape}")

        return X, y

    def get_latest_bar_close(self, symbol: str) -> Optional[float]:
        """
        Retrieves the close price of the most recent bar for a given symbol from local history.
        Used for dry-run trade closure simulation.
        """
        df = self.load_local_history(symbol, self.cfg.timeframe, count=1)
        if not df.empty:
            return df["close"].iloc[-1]
        return None
