#src/data_manager.py
from __future__ import annotations
import os
import tempfile
import pandas as pd  # type: ignore
from loguru import logger  # type: ignore
from typing import Optional, Tuple, Dict
from enum import Enum
import datetime

from src.config import Cfg
from src.features import FeatureCfg, build_features
from src.labels import generate_labels

class DataStatus(Enum):
    OK = "OK"
    STALE = "STALE"
    GAP = "GAP"
    EMPTY = "EMPTY"
    ERROR = "ERROR"

def ensure_dir(path: str):
    if path is None:
        return
    os.makedirs(path, exist_ok=True)

class DataManager:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.raw_data_dir = cfg.fetch.raw_data_dir
        ensure_dir(self.raw_data_dir)
        self.last_processed_bar_time: Dict[str, pd.Timestamp | None] = {sym: None for sym in cfg.symbols}
        self.consecutive_valid_fetches: Dict[str, int] = {sym: 0 for sym in cfg.symbols}

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

    def _validate_fetched_data(self, symbol: str, fetched_data: pd.DataFrame, timeframe_seconds: int, notifier) -> Tuple[DataStatus, str]:
        if fetched_data.empty:
            return DataStatus.EMPTY, f"[{symbol}] Fetched data is empty."

        latest_bar_time = fetched_data.index[-1]
        last_processed_time = self.last_processed_bar_time.get(symbol)

        # If this is the very first bar for the symbol, just accept it
        if last_processed_time is None:
            self.last_processed_bar_time[symbol] = latest_bar_time
            self.consecutive_valid_fetches[symbol] += 1
            return DataStatus.OK, f"[{symbol}] First bar processed: {latest_bar_time}"

        # 1. Stale Data Check
        if latest_bar_time <= last_processed_time:
            msg = f"[{symbol}] Stale data detected. Latest bar time: {latest_bar_time}, Last processed: {last_processed_time}"
            logger.warning(msg)
            if notifier: notifier.send_message(f"<b>WARNING:</b> {msg}", level="WARNING")
            self.consecutive_valid_fetches[symbol] = 0 # Reset consecutive valid fetches
            return DataStatus.STALE, msg

        # 2. Gap Detection
        expected_next_bar_time = last_processed_time + pd.Timedelta(seconds=timeframe_seconds)
        if latest_bar_time > expected_next_bar_time:
            # Allow for slight clock drift, but if it's more than 1.5x the timeframe, it's a gap
            if (latest_bar_time - expected_next_bar_time).total_seconds() > (timeframe_seconds * 1.5):
                msg = f"[{symbol}] Data gap detected. Expected next bar around: {expected_next_bar_time}, Received: {latest_bar_time}"
                logger.warning(msg)
                if notifier: notifier.send_message(f"<b>WARNING:</b> {msg}", level="WARNING")
                self.consecutive_valid_fetches[symbol] = 0 # Reset consecutive valid fetches
                return DataStatus.GAP, msg

        # If all checks pass, update last_processed_bar_time and increment consecutive valid fetches
        self.last_processed_bar_time[symbol] = latest_bar_time
        self.consecutive_valid_fetches[symbol] += 1
        return DataStatus.OK, f"[{symbol}] Data OK. Latest bar: {latest_bar_time}."

    def load_local_history(self, symbol: str, timeframe: str, count: Optional[int] = None) -> pd.DataFrame:
        path = self._local_csv_path(symbol, timeframe)
        if not os.path.exists(path):
            return pd.DataFrame()
        df = pd.read_csv(path, index_col=0)
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
        if count is not None and len(df) > count:
            df = df.tail(count)
        return df

    def append_new_bars(self, symbol: str, new_bars: pd.DataFrame):
        if not isinstance(new_bars, pd.DataFrame) or new_bars.empty:
            logger.debug(f"[{symbol}] No new bars to append.")
            return
        path = self._local_csv_path(symbol, self.cfg.timeframe)
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
        logger.debug(f"[{symbol}] Appended {len(nb)} bars to {path} (total {len(combined)})")

    def _fetch_bars_from_mt5_chunked(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        import MetaTrader5 as mt5  # type: ignore
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
            logger.info(f"[{symbol}] Fetching {count} bars from MT5 for timeframe {timeframe}...") # New log
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, int(count))
            if rates is None or len(rates) == 0:
                logger.warning(f"[{symbol}] MT5 returned no bars.")
                return pd.DataFrame()
            
            logger.info(f"[{symbol}] Fetched {len(rates)} bars from MT5.") # New log

            df = pd.DataFrame(rates)
            if "time" not in df.columns:
                logger.warning(f"[{symbol}] fetched data missing 'time' column — returning empty DataFrame")
                return pd.DataFrame()
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.set_index("time").sort_index()
            df = df.rename(columns={"tick_volume": "volume"})
            return df[["open","high","low","close","volume"]].copy()
        except Exception as e:
            logger.exception(f"[{symbol}] Error fetching bars: {e}")
            return pd.DataFrame()

    def bootstrap_history(self, symbol: str, initial_bars: int):
        path = self._local_csv_path(symbol, self.cfg.timeframe)
        current = self.load_local_history(symbol, self.cfg.timeframe)
        if current.empty or len(current) < initial_bars:
            logger.info(f"[{symbol}] Bootstrapping local history: have={len(current)}, need={initial_bars}")
            df = self._fetch_bars_from_mt5_chunked(symbol, self.cfg.timeframe, initial_bars)
            if df.empty:
                logger.warning(f"[{symbol}] Bootstrap failed: MT5 returned no data.")
                return
            self._atomic_write_df(df, path, fmt="csv")
            logger.info(f"[{symbol}] Bootstrapped local history to {path} ({len(df)} rows).")
        else:
            logger.debug(f"[{symbol}] Local history OK ({len(current)} rows).")

    def fetch_live(self, symbol: str, feature_cfg: "FeatureCfg", min_pct_change: float, notifier) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, DataStatus, str]:
        # 1. Load the bulk of the history from the local cache first.
        data = self.load_local_history(symbol, self.cfg.timeframe, count=self.cfg.history_bars)

        # 2. Fetch only a small number of recent bars to get the absolute latest data.
        # This is more efficient than fetching the entire history every time.
        recent_data = self._fetch_bars_from_mt5_chunked(symbol, self.cfg.timeframe, 200) # Fetch last 200 bars

        # 3. Validate the fetched recent data
        timeframe_seconds = self.cfg.timeframe_seconds()
        if timeframe_seconds is None:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), DataStatus.ERROR, f"[{symbol}] Invalid timeframe configuration."

        validation_status, validation_msg = self._validate_fetched_data(symbol, recent_data, timeframe_seconds, notifier)

        if validation_status != DataStatus.OK:
            # If data is not OK, return empty dataframes and the status/message
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), validation_status, validation_msg

        # 4. Combine and de-duplicate.
        if not recent_data.empty:
            data = pd.concat([data, recent_data])
            data = data[~data.index.duplicated(keep='last')].sort_index()

        if data.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), DataStatus.EMPTY, f"[{symbol}] Combined data is empty after fetch."

        # 5. Fetch context features, if enabled
        mta_df = None
        if self.cfg.context_features.mta.enabled:
            logger.debug(f"[{symbol}] Fetching live MTA data...")
            # Fetching the full history for MTA is okay as it's a different timeframe and less frequent.
            mta_df = self._fetch_bars_from_mt5_chunked(symbol, self.cfg.context_features.mta.timeframe, self.cfg.history_bars)
            if mta_df.empty:
                logger.warning(f"[{symbol}] No live MTA data loaded for timeframe {self.cfg.context_features.mta.timeframe}. Disabling MTA for this tick.")
                mta_df = None

        inter_market_df = None
        if self.cfg.context_features.inter_market.enabled:
            im_sym = self.cfg.context_features.inter_market.symbol
            logger.debug(f"[{symbol}] Fetching live Inter-Market data for {im_sym}...")
            inter_market_df = self._fetch_bars_from_mt5_chunked(im_sym, self.cfg.timeframe, self.cfg.history_bars)
            if inter_market_df.empty:
                logger.warning(f"[{symbol}] No live Inter-Market data loaded for symbol {im_sym}. Disabling for this tick.")
                inter_market_df = None

        # 6. Build features and labels with all data
        X, y = self._build_features_and_labels(data, feature_cfg, symbol, min_pct_change, mta_df=mta_df, inter_market_df=inter_market_df)

        # 7. Align all dataframes by index
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        data = data.loc[common_idx]

        return data, X, y, DataStatus.OK, validation_msg

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