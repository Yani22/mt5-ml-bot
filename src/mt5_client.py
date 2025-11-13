# src/mt5_client.py
from __future__ import annotations
import time
import datetime
from typing import Optional
import MetaTrader5 as mt5  # type: ignore
from loguru import logger
from src.time_utils import timeframe_to_seconds

class MT5Client:
    """ Safe wrapper around MetaTrader5 initialization and login. """

    def __init__(
        self,
        login: Optional[str] | Optional[int],
        password: Optional[str],
        server: Optional[str],
        path: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        self._raw_login = login
        self.password = password
        self.server = server
        self.path = path
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._connected = False

        # attempt coercion to int but keep original if not possible
        self.login = None
        if login is not None and str(login).strip() != "":
            try:
                self.login = int(login)
            except Exception:
                # could be string-based login – keep the raw value for mt5.login
                self.login = login

    def connect(self) -> bool:
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"MT5Client: initialize() attempt {attempt}/{self.max_retries} (path={self.path})")
                ok = mt5.initialize(path=self.path) if self.path else mt5.initialize()
                if not ok:
                    last_err = mt5.last_error()
                    logger.error(f"MT5 initialize() failed: {last_err}")
                    mt5.shutdown()
                    time.sleep(self.retry_delay)
                    continue

                if self.login is not None and self.password and self.server:
                    logger.debug("MT5Client: attempting explicit mt5.login()")
                    authorized = mt5.login(login=self.login, password=self.password, server=self.server)
                    if not authorized:
                        last_err = mt5.last_error()
                        logger.error(f"MT5 login failed: {last_err}")
                        mt5.shutdown()
                        time.sleep(self.retry_delay)
                        continue
                    logger.debug("MT5 login OK")
                else:
                    # No creds: validate terminal login
                    acct = mt5.account_info()
                    if acct is None:
                        last_err = mt5.last_error()
                        logger.error("MT5 terminal not logged in and no credentials were provided.")
                        mt5.shutdown()
                        time.sleep(self.retry_delay)
                        continue
                    logger.info(f"MT5 terminal already logged in (account={acct.login})")

                # verify account_info now
                account_info = mt5.account_info()
                if account_info is None:
                    last_err = mt5.last_error()
                    logger.error("MT5 connected but account_info() returned None.")
                    mt5.shutdown()
                    time.sleep(self.retry_delay)
                    continue

                logger.info(f"MT5 connected successfully (account={account_info.login})")
                self._connected = True
                return True

            except Exception as exc:
                last_err = exc
                logger.exception(f"MT5Client: unexpected error on connect: {exc}")
                try:
                    mt5.shutdown()
                except Exception:
                    pass
                time.sleep(self.retry_delay)

        logger.critical(f"MT5Client: failed to connect after {self.max_retries} attempts. Last error: {last_err}")
        return False

    def is_connected(self) -> bool:
        return bool(self._connected)

    def shutdown(self) -> None:
        try:
            if self._connected:
                logger.info("MT5Client: shutting down connection.")
            else:
                logger.info("MT5Client: shutdown() called but client not connected.")
            mt5.shutdown()
        except Exception as e:
            logger.warning(f"MT5Client: exception during shutdown: {e}")
        finally:
            self._connected = False

    def account_info(self):
        if not self._connected: return None
        try:
            return mt5.account_info()
        except Exception:
            return None

    def now_utc(self):
        """Returns the current UTC time from the MetaTrader 5 terminal."""
        if not self._connected:
            return datetime.datetime.now(datetime.timezone.utc)

        try:
            # First, try to get server time from a common symbol like EURUSDm
            common_symbol = "EURUSDm#"
            tick = mt5.symbol_info_tick(common_symbol)
            if tick and tick.time > 0:
                return datetime.datetime.fromtimestamp(tick.time, tz=datetime.timezone.utc)
            else:
                logger.warning(f"MT5Client: Could not get server time from {common_symbol}. Trying other symbols.")

            # If that fails, iterate through all symbols to find one with a valid tick
            symbols = mt5.symbols_get()
            if symbols:
                # Filter out the common symbol since it has already been checked
                other_symbols = [s for s in symbols if s.name != common_symbol]
                for symbol_info in other_symbols:
                    tick = mt5.symbol_info_tick(symbol_info.name)
                    if tick and tick.time > 0:
                        return datetime.datetime.fromtimestamp(tick.time, tz=datetime.timezone.utc)
        except Exception as e:
            logger.warning(f"MT5Client: exception getting server time from tick: {e}")
            # Fallback to system time if we can't get server time from any tick
            pass
        
        return datetime.datetime.now(datetime.timezone.utc)

    def get_timezone_offset(self) -> Optional[float]:
        """Calculates the timezone offset of the broker's server from UTC in hours."""
        if not self._connected:
            logger.warning("get_timezone_offset: Not connected to MT5.")
            return None

        server_time_utc = self.now_utc()
        system_time_utc = datetime.datetime.now(datetime.timezone.utc)
        
        offset_seconds = (server_time_utc - system_time_utc).total_seconds()
        return offset_seconds / 3600

    def symbol_info_tick(self, symbol: str):
        """Wrapper for mt5.symbol_info_tick()"""
        if not self._connected: return None
        try:
            return mt5.symbol_info_tick(symbol)
        except Exception:
            return None

    def symbol_info(self, symbol: str):
        """Wrapper for mt5.symbol_info()"""
        if not self._connected: return None
        try:
            return mt5.symbol_info(symbol)
        except Exception:
            return None

    def history_deals_get(self, *args, **kwargs):
        """Wrapper for mt5.history_deals_get()"""
        if not self._connected: return None
        try:
            return mt5.history_deals_get(*args, **kwargs)
        except Exception:
            return None

    def order_send(self, request: dict):
        """Wrapper for mt5.order_send()"""
        if not self._connected: return None
        try:
            return mt5.order_send(request)
        except Exception:
            return None
        
    def positions_get(self, *args, **kwargs):
        """Wrapper for mt5.positions_get()"""
        if not self._connected: return None
        try:
            return mt5.positions_get(*args, **kwargs)
        except Exception:
            return None

    def get_rates(self, symbol: str, timeframe: int, count: int):
        """Wrapper for mt5.copy_rates_from_pos()"""
        if not self._connected: return None
        try:
            return mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        except Exception:
            return None

    def wait_for_new_bar(self, symbol: str, timeframe: int = mt5.TIMEFRAME_M1, timeout_multiplier: float = 1.5):
        """Waits for a new bar to appear for a given symbol and timeframe."""
        if not self._connected:
            logger.warning("wait_for_new_bar: Not connected to MT5.")
            return False

        timeframe_seconds = timeframe_to_seconds(timeframe) # Convert MT5 timeframe to seconds
        dynamic_timeout = int(timeframe_seconds * timeout_multiplier) # Calculate dynamic timeout
        if dynamic_timeout < 60: # Ensure a minimum timeout of 60 seconds
            dynamic_timeout = 60

        last_bar = self.get_rates(symbol, timeframe, 1)
        if last_bar is None or len(last_bar) == 0:
            logger.warning(f"wait_for_new_bar: Could not get last bar for {symbol}.")
            return False

        last_bar_time = last_bar[0][0]
        logger.info(f"wait_for_new_bar: Waiting for new bar for {symbol}. Last bar time: {datetime.datetime.fromtimestamp(last_bar_time, tz=datetime.timezone.utc)}. Timeout: {dynamic_timeout}s")
        start_time = time.time()

        while time.time() - start_time < dynamic_timeout:
            new_bar = self.get_rates(symbol, timeframe, 1)
            if new_bar is not None and len(new_bar) > 0 and new_bar[0][0] > last_bar_time:
                return True
            time.sleep(1)

        logger.warning(f"wait_for_new_bar: Timeout waiting for new bar for {symbol} after {dynamic_timeout}s.")
        return False

    # --- MT5 Constants ---
    ORDER_TYPE_BUY = mt5.ORDER_TYPE_BUY
    ORDER_TYPE_SELL = mt5.ORDER_TYPE_SELL
    TRADE_ACTION_DEAL = mt5.TRADE_ACTION_DEAL
    ORDER_TIME_GTC = mt5.ORDER_TIME_GTC
    ORDER_FILLING_IOC = mt5.ORDER_FILLING_IOC
    TRADE_RETCODE_DONE = mt5.TRADE_RETCODE_DONE