# src/mt5_client.py
from __future__ import annotations
import time
import datetime
from typing import Optional
import MetaTrader5 as mt5  # type: ignore
from loguru import logger

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
                logger.info(f"MT5Client: initialize() attempt {attempt}/{self.max_retries} (path={self.path})")
                ok = mt5.initialize(path=self.path) if self.path else mt5.initialize()
                if not ok:
                    last_err = mt5.last_error()
                    logger.error(f"MT5 initialize() failed: {last_err}")
                    mt5.shutdown()
                    time.sleep(self.retry_delay)
                    continue

                if self.login is not None and self.password and self.server:
                    logger.info("MT5Client: attempting explicit mt5.login()")
                    authorized = mt5.login(self.login, password=self.password, server=self.server)
                    if not authorized:
                        last_err = mt5.last_error()
                        logger.error(f"MT5 login failed: {last_err}")
                        mt5.shutdown()
                        time.sleep(self.retry_delay)
                        continue
                    logger.info("MT5 login OK")
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
        try:
            return mt5.account_info()
        except Exception:
            return None

    def now_utc(self):
        """Returns the current UTC time from the MetaTrader 5 terminal."""
        if not self._connected:
            return datetime.datetime.now(datetime.timezone.utc)

        try:
            # We need a symbol to get the server time. Let's use the first one from the market watch.
            symbols = mt5.symbols_get()
            if symbols:
                tick = mt5.symbol_info_tick(symbols[0].name)
                if tick and tick.time > 0:
                    return datetime.datetime.fromtimestamp(tick.time, tz=datetime.timezone.utc)
        except Exception:
            # Fallback to system time if we can't get server time from a tick
            pass
        
        return datetime.datetime.now(datetime.timezone.utc)

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

    # --- MT5 Constants ---
    ORDER_TYPE_BUY = mt5.ORDER_TYPE_BUY
    ORDER_TYPE_SELL = mt5.ORDER_TYPE_SELL
    TRADE_ACTION_DEAL = mt5.TRADE_ACTION_DEAL
    ORDER_TIME_GTC = mt5.ORDER_TIME_GTC
    ORDER_FILLING_IOC = mt5.ORDER_FILLING_IOC
    TRADE_RETCODE_DONE = mt5.TRADE_RETCODE_DONE