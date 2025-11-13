#src/data_manager.py
from __future__ import annotations
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
                logger.warning(f"MT5Client connection attempt {i+1}/{max_retries} failed in DataManager. Retrying in 5 seconds...")
                time.sleep(5)
        if not self.mt5_client.is_connected():
            logger.error("Failed to connect MT5Client in DataManager after multiple retries.")
            # Optionally, raise an exception or handle this failure more gracefully


    def _local_csv_path(self, symbol: str, timeframe: str) -> str:
        fname = f"{symbol.replace('#','')}_{timeframe}.csv"
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
        import datetime # NEW
        from src.time_utils import timeframe_to_seconds # NEW

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
                logger.warning(f"[{symbol}] MT5 returned no bars.")
                return pd.DataFrame()
            
            df = pd.DataFrame(rates)
            if "time" not in df.columns:
                logger.warning(f"[{symbol}] fetched data missing 'time' column — returning empty DataFrame")
                return pd.DataFrame()
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.set_index("time").sort_index()
            df = df.rename(columns={"tick_volume": "volume"})
            
            # Drop the last bar if it's the currently forming one
            # This ensures all data used is from fully closed bars
            if not df.empty:
                df = df.iloc[:-1]

            return df[["open","high","low","close","volume"]].copy()
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

    def fetch_live(self, symbol: str, feature_cfg: FeatureCfg) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # 1. Load the bulk of the history from the local cache first.
        data = self.load_local_history(symbol, self.cfg.timeframe, count=self.cfg.history_bars)

        # 2. Fetch only a small number of recent bars to get the absolute latest data.
        # This is more efficient than fetching the entire history every time.
        recent_data = self._fetch_bars_from_mt5_chunked(symbol, self.cfg.timeframe, 200) # Fetch last 200 bars

        # 3. Combine and de-duplicate.
        if not recent_data.empty:
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
                logger.warning(f"[{symbol}] No live MTA data loaded for timeframe {self.cfg.context_features.mta.timeframe}. Disabling MTA for this tick.")
                mta_df = None
            else:
                # Append the newly fetched recent MTA data to local history
                if self.cfg.fetch.save_raw_data_locally:
                    self.append_new_bars(symbol, mta_recent_data, timeframe=self.cfg.context_features.mta.timeframe)
                # Load the full (updated) local history for feature building
                mta_df = self.load_local_history(symbol, self.cfg.context_features.mta.timeframe, count=self.cfg.history_bars)


        inter_market_df = None
        if self.cfg.context_features.inter_market.enabled:
            im_sym = self.cfg.context_features.inter_market.symbol
            # Fetch only a small number of recent bars for Inter-Market data
            im_recent_data = self._fetch_bars_from_mt5_chunked(im_sym, self.cfg.timeframe, 200)
            if im_recent_data.empty:
                logger.warning(f"[{symbol}] No live Inter-Market data loaded for symbol {im_sym}. Disabling for this tick.")
                inter_market_df = None
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

        # Drop the last row to ensure we only use closed bars
        if not data.empty:
            pass # Removed redundant iloc[:-1] calls to fix data misalignment

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
                logger.warning(f"[{symbol}] No MTA data loaded for timeframe {self.cfg.context_features.mta.timeframe}. Disabling MTA features.")
                self.cfg.context_features.mta.enabled = False # Temporarily disable to prevent errors
                mta_df = None
            else:
                logger.info(f"[{symbol}] Successfully loaded MTA data for timeframe {self.cfg.context_features.mta.timeframe}.")

        # Load Inter-Market data if enabled
        inter_market_df = None
        if self.cfg.context_features.inter_market.enabled:
            im_sym = self.cfg.context_features.inter_market.symbol
            inter_market_df = self.load_local_history(im_sym, self.cfg.timeframe, count=count)
            if inter_market_df.empty:
                logger.warning(f"[{symbol}] No Inter-Market data loaded for symbol {im_sym}. Disabling Inter-Market features.")
                self.cfg.context_features.inter_market.enabled = False # Temporarily disable to prevent errors
                inter_market_df = None
            else:
                logger.info(f"[{symbol}] Successfully loaded Inter-Market data for symbol {im_sym}.")

        X, y = self._build_features_and_labels(data, feature_cfg, symbol, min_pct_change, mta_df=mta_df, inter_market_df=inter_market_df)

        # Align X, y, and data by index
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        data = data.loc[common_idx] # Align data here

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