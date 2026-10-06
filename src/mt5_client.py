# src/mt5_client.py
from __future__ import annotations
import threading
import time
import datetime
from typing import Optional
import MetaTrader5 as mt5  # type: ignore
from loguru import logger
from src.time_utils import timeframe_to_seconds


# The MetaTrader5 package keeps ONE connection per process. The bot makes many MT5Client objects (main, one per symbol, one
# per DataManager), so they share it: the first connect() initialises and logs in, later ones reuse it while it is healthy,
# a client's shutdown() only releases that client, and teardown_connection() (called by main on a reconnect and at exit)
# closes the real connection and makes every client attached before it read as disconnected.
_state_lock = threading.Lock()
_generation = 0         # bumped by teardown_connection(); a client attached in an older generation is disconnected
_initialized = False    # this process has initialised the terminal connection (read and written under _state_lock)


def teardown_connection() -> None:
    """Close the real terminal connection and disconnect every client attached so far."""
    global _generation, _initialized
    with _state_lock:
        try:
            mt5.shutdown()
        except Exception as e:
            logger.warning(f"MT5Client: exception during teardown: {e}")
        _initialized = False
        _generation += 1


def _reset_shared_state() -> None:
    """Tests only: forget the process-wide connection state."""
    global _generation, _initialized
    with _state_lock:
        _generation, _initialized = 0, False


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
        time_symbols: Optional[list] = None,
    ):
        self._raw_login = login
        self.password = password
        self.server = server
        self.path = path
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._attached_generation: Optional[int] = None  # see the `_connected` property
        self.time_symbols = list(time_symbols or [])  # symbols asked for the server time (the ones the bot trades)
        self._time_symbol: Optional[str] = None       # the symbol that last gave a tick
        self._time_warned = False                     # the "no tick from the configured symbols" warning is logged once

        # attempt coercion to int but keep original if not possible
        self.login = None
        if login is not None and str(login).strip() != "":
            try:
                self.login = int(login)
            except Exception:
                # could be string-based login – keep the raw value for mt5.login
                self.login = login

    @property
    def _connected(self) -> bool:
        """True while this client is attached to the process's connection and no teardown has happened since."""
        return self._attached_generation is not None and self._attached_generation == _generation

    @_connected.setter
    def _connected(self, value: bool) -> None:
        self._attached_generation = _generation if value else None

    @staticmethod
    def _shared_connection_is_healthy() -> bool:
        try:
            return mt5.terminal_info() is not None and mt5.account_info() is not None
        except Exception:
            return False

    def connect(self) -> bool:
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            with _state_lock:   # one attempt at a time; the retry sleep below is outside the lock
                ok, last_err = self._attempt(attempt)
            if ok:
                return True
            time.sleep(self.retry_delay)

        logger.critical(f"MT5Client: failed to connect after {self.max_retries} attempts. Last error: {last_err}")
        return False

    def _attempt(self, attempt: int):
        """One connect attempt, called with _state_lock held. Returns (connected, last_error)."""
        global _initialized
        if _initialized and self._shared_connection_is_healthy():
            logger.debug("MT5Client: reusing the process's existing MT5 connection.")
            self._connected = True
            return True, None
        # Nothing healthy to share: (re)initialise. No mt5.shutdown() first: a false "unhealthy" reading must not cut off
        # threads that still have a working connection.
        last_err = None
        try:
            logger.debug(f"MT5Client: initialize() attempt {attempt}/{self.max_retries} (path={self.path})")
            ok = mt5.initialize(path=self.path) if self.path else mt5.initialize()
            if not ok:
                last_err = mt5.last_error()
                logger.error(f"MT5 initialize() failed: {last_err}")
                _initialized = False
                mt5.shutdown()
                return False, last_err

            if self.login is not None and self.password and self.server:
                logger.debug("MT5Client: attempting explicit mt5.login()")
                authorized = mt5.login(login=self.login, password=self.password, server=self.server)
                if not authorized:
                    last_err = mt5.last_error()
                    logger.error(f"MT5 login failed: {last_err}")
                    _initialized = False
                    mt5.shutdown()
                    return False, last_err
                logger.debug("MT5 login OK")
            else:
                # No creds: validate terminal login
                acct = mt5.account_info()
                if acct is None:
                    last_err = mt5.last_error()
                    logger.error("MT5 terminal not logged in and no credentials were provided.")
                    _initialized = False
                    mt5.shutdown()
                    return False, last_err
                logger.info(f"MT5 terminal already logged in (account={acct.login})")

            # verify account_info now
            account_info = mt5.account_info()
            if account_info is None:
                last_err = mt5.last_error()
                logger.error("MT5 connected but account_info() returned None.")
                _initialized = False
                mt5.shutdown()
                return False, last_err

            logger.info(f"MT5 connected successfully (account={account_info.login})")
            _initialized = True
            self._connected = True
            return True, None

        except Exception as exc:
            logger.exception(f"MT5Client: unexpected error on connect: {exc}")
            _initialized = False
            try:
                mt5.shutdown()
            except Exception:
                pass
            return False, exc

    def is_connected(self) -> bool:
        return bool(self._connected)

    def shutdown(self) -> None:
        """Release this client. The shared MT5 connection stays open for the other clients; teardown_connection() closes it."""
        if self._connected:
            logger.info("MT5Client: releasing this client (the shared MT5 connection stays open).")
        else:
            logger.info("MT5Client: shutdown() called but client not connected.")
        self._connected = False

    def account_info(self):
        if not self._connected:
            return None
        try:
            return mt5.account_info()
        except Exception:
            return None

    def now_utc(self):
        """Returns the current time from the MetaTrader 5 terminal (the server's clock, labelled UTC by this code base).

        Asks the symbol that last worked, then `time_symbols`; only if none answers does it look at the Market Watch symbols and
        keep the one with the newest tick (a closed market's last tick can be hours old). Falls back to the system clock."""
        if not self._connected:
            return datetime.datetime.now(datetime.timezone.utc)

        try:
            tried = []
            for name in [self._time_symbol, *self.time_symbols]:
                if not name or name in tried:
                    continue
                tried.append(name)
                tick = mt5.symbol_info_tick(name)
                if tick and tick.time > 0:
                    self._time_symbol = name
                    return datetime.datetime.fromtimestamp(tick.time, tz=datetime.timezone.utc)

            if not self._time_warned:
                self._time_warned = True
                logger.warning(f"MT5Client: no tick from {tried or 'any configured symbol'} for the server time; "
                               f"using the freshest Market Watch tick instead.")
            best = None
            for info in mt5.symbols_get() or []:
                if not getattr(info, "select", False) or info.name in tried:
                    continue
                tick = mt5.symbol_info_tick(info.name)
                if tick and tick.time > 0 and (best is None or tick.time > best[0]):
                    best = (tick.time, info.name)
            if best:
                self._time_symbol = best[1]
                return datetime.datetime.fromtimestamp(best[0], tz=datetime.timezone.utc)
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
        if not self._connected:
            return None
        try:
            return mt5.symbol_info_tick(symbol)
        except Exception:
            return None

    def symbol_info(self, symbol: str):
        """Wrapper for mt5.symbol_info()"""
        if not self._connected:
            return None
        try:
            return mt5.symbol_info(symbol)
        except Exception:
            return None

    def history_deals_get(self, *args, **kwargs):
        """Wrapper for mt5.history_deals_get()"""
        if not self._connected:
            return None
        try:
            return mt5.history_deals_get(*args, **kwargs)
        except Exception:
            return None

    def order_send(self, request: dict):
        """Wrapper for mt5.order_send()"""
        if not self._connected:
            return None
        try:
            return mt5.order_send(request)
        except Exception:
            return None

    def positions_get(self, *args, **kwargs):
        """Wrapper for mt5.positions_get()"""
        if not self._connected:
            return None
        try:
            return mt5.positions_get(*args, **kwargs)
        except Exception:
            return None

    def get_rates(self, symbol: str, timeframe: int, count: int):
        """Wrapper for mt5.copy_rates_from_pos()"""
        if not self._connected:
            return None
        try:
            return mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        except Exception:
            return None

    def wait_for_new_bar(self, symbol: str, timeframe: int = mt5.TIMEFRAME_M1, timeout_multiplier: float = 1.5, stop_event=None):
        """Waits for a new bar to appear for a given symbol and timeframe. Returns False early once `stop_event` is set."""
        if not self._connected:
            logger.warning("wait_for_new_bar: Not connected to MT5.")
            return False

        timeframe_seconds = timeframe_to_seconds(timeframe)  # Convert MT5 timeframe to seconds
        dynamic_timeout = int(timeframe_seconds * timeout_multiplier)  # Calculate dynamic timeout
        if dynamic_timeout < 60:  # Ensure a minimum timeout of 60 seconds
            dynamic_timeout = 60

        last_bar = self.get_rates(symbol, timeframe, 1)
        if last_bar is None or len(last_bar) == 0:
            logger.warning(f"wait_for_new_bar: Could not get last bar for {symbol}.")
            return False

        last_bar_time = last_bar[0][0]
        logger.info(f"wait_for_new_bar: Waiting for new bar for {symbol}. Last bar time: {datetime.datetime.fromtimestamp(last_bar_time, tz=datetime.timezone.utc)}. Timeout: {dynamic_timeout}s")
        start_time = time.time()

        while time.time() - start_time < dynamic_timeout:
            if stop_event is not None and stop_event.is_set():
                return False
            new_bar = self.get_rates(symbol, timeframe, 1)
            if new_bar is not None and len(new_bar) > 0 and new_bar[0][0] > last_bar_time:
                return True
            if stop_event is not None:
                stop_event.wait(1)
            else:
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
