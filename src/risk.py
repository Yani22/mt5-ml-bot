from loguru import logger  # type: ignore
try:
    import MetaTrader5 as mt5  # type: ignore
except ImportError:  # not available on Linux; only needed on the live (Windows) path
    mt5 = None
from .balance_flows import BalanceFlowCursor, shift_peak
from .config import Cfg
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import copy
import datetime
import math
from datetime import timezone, timedelta
from typing import List, Optional  # Import Optional
from .trade import SimPosition  # Import SimPosition
from .notifier import TelegramNotifier  # NEW import
import threading


def trailing_losses(profits) -> int:
    """Losing results at the end of `profits` (oldest first, net of costs): the watchdog's count, shared by live and the backtester."""
    count = 0
    for p in reversed(list(profits)):
        if p < 0:
            count += 1
        else:
            break
    return count


class RiskManager:
    """
    RiskManager handles dynamic position sizing, SL/TP, portfolio exposure caps,
    open-position bookkeeping, and watchdog/cooldown behavior.

    Callers: pass `cfg` (the Cfg object) to constructor so both risk and watchdog settings are available.
    """

    def __init__(self, cfg: Cfg, mt5_client, lock: threading.Lock, notifier: Optional[TelegramNotifier] = None):
        self.cfg = cfg
        self.risk_cfg = cfg.risk
        self.watchdog_cfg = cfg.watchdog
        self.equity_peak: float | None = None
        self._balance_flows = BalanceFlowCursor()  # deposits and withdrawals the peak already accounts for (B7)
        self._pending_flow = 0.0  # polled but not yet applied to the peak (the equity read failed in between)
        self.open_positions_cache: dict[str, dict] = {}
        self.cooldown_until: datetime.datetime | None = None
        self.recently_closed_trades: List[SimPosition] = []  # New: To store closed trades for monitoring
        self.notifier = notifier  # NEW
        self.mt5_client = mt5_client
        self.cache_lock = lock
        # One entry at a time: the open-risk read, the `max_positions` check, sizing and the order must not interleave
        # between symbols. Always taken before `cache_lock`, never while holding it.
        self.entry_lock = threading.Lock()
        self.entry_lock_timeout = 60.0

    def get_contract_size(self, symbol: str) -> float:
        symbol_info = self.mt5_client.symbol_info(symbol)
        if not symbol_info:
            logger.warning(f"[{symbol}] Could not get symbol info. Returning default contract size.")
            return 100000.0
        return symbol_info.trade_contract_size

    def get_pip_size(self, symbol: str) -> float:
        symbol_info = self.mt5_client.symbol_info(symbol)
        if not symbol_info:
            logger.warning(f"[{symbol}] Could not get symbol info. Returning default pip size.")
            return 0.0001
        return symbol_info.point

    def get_pip_value(self, symbol: str) -> float:
        symbol_info = self.mt5_client.symbol_info(symbol)
        if not symbol_info:
            logger.warning(f"[{symbol}] Could not get symbol info. Returning default pip value.")
            return 1.0
        if not (hasattr(symbol_info, "trade_tick_value") and hasattr(symbol_info, "trade_tick_size")):
            # Older stand-ins without tick data: point * contract size is in the quote currency (right for USD-quoted pairs only).
            return symbol_info.point * symbol_info.trade_contract_size
        # Money per point per lot in ACCOUNT currency. A missing or zero tick value (symbol not in Market Watch, cross rate
        # not quoted) returns 0 so sizing skips the trade; falling back to the quote-currency value would be ~150x off for JPY.
        try:
            tick_value = float(symbol_info.trade_tick_value)
            tick_size = float(symbol_info.trade_tick_size)
        except (TypeError, ValueError):
            return 0.0
        if tick_value <= 0 or tick_size <= 0:
            logger.warning(f"[{symbol}] Tick value ({tick_value}) or tick size ({tick_size}) unusable; no pip value.")
            return 0.0
        return symbol_info.point * tick_value / tick_size

    def pips_to_money(self, symbol: str, pips: float, lots: float) -> float:
        """Account-currency value of `pips` pips on `lots`. A pip is 10 points on 3 and 5 digit symbols and 1 point otherwise
        (so for gold and indices a "pip" is one point). 0 when the tick value is unusable."""
        info = self.mt5_client.symbol_info(symbol)
        pip_value = self.get_pip_value(symbol)
        if not info or pip_value <= 0:
            return 0.0
        points_per_pip = 10 if getattr(info, "digits", None) in (3, 5) else 1
        return float(pips) * points_per_pip * pip_value * float(lots)

    def move_value(self, symbol: str, price_diff: float, lots: float) -> float:
        """Money (account currency) gained by a price move of price_diff (signed) on `lots`; 0 when the tick value is unusable."""
        pip_size = self.get_pip_size(symbol)
        pip_value = self.get_pip_value(symbol)
        if pip_size <= 0 or pip_value <= 0:
            return 0.0
        return float(price_diff) / pip_size * pip_value * float(lots)

    # ---------- Dynamic value helpers ----------
    def _get_dynamic_value(self, dynamic_cfg: dict | None, auc_score: float, default_val: float) -> float:
        if not dynamic_cfg or not dynamic_cfg.get("enabled"):
            return float(default_val)
        auc_floor = float(dynamic_cfg.get("auc_floor", 0.55))
        auc_ceiling = float(dynamic_cfg.get("auc_ceiling", 0.65))
        base_val = float(dynamic_cfg.get("base_risk", dynamic_cfg.get("base_tp_mult", default_val)))
        max_val = float(dynamic_cfg.get("max_risk", dynamic_cfg.get("max_tp_mult", default_val)))
        clamped = float(np.clip(auc_score, auc_floor, auc_ceiling))
        denom = max(auc_ceiling - auc_floor, 1e-6)
        val = base_val + (clamped - auc_floor) * (max_val - base_val) / denom
        logger.debug(f"Dynamic value calc: AUC={auc_score:.4f}, Clamped={clamped:.4f}, Value={val:.4f}")
        return float(val)

    # ---------- Position sizing ----------
    def position_size(self, equity: float, atr: float, auc_score: float, total_open_risk: float = 0.0, symbol: str | None = None, exploration_mult: float = 1.0, ac_multiplier: float = 1.0, sl_distance: float | None = None, risk_scale: float = 1.0) -> tuple[float, float]:
        """
        Calculates position size from the stop distance and the configured risk.
        total_open_risk is the MONEY (account currency) currently at risk in open positions; the portfolio cap
        max_portfolio_risk is a fraction of equity. sl_distance is the stop distance in price units that will really
        be placed; without it the configured ATR multiple is used.
        """
        with self.cache_lock:
            # CRITICAL SAFETY CHECK: Do not open a new position if one already exists for this symbol.
            if any(pos.get('symbol') == symbol for pos in self.open_positions_cache.values()):
                logger.warning(f"[{symbol}] Blocking new trade: A position is already open for this symbol.")
                return 0.0, 0.0

        symbol_info = self.mt5_client.symbol_info(symbol)
        if not symbol_info:
            logger.warning(f"[{symbol}] Could not get symbol info. Cannot calculate position size.")
            return 0.0, 0.0

        pip_size = symbol_info.point
        pip_value = self.get_pip_value(symbol)  # Use helper to get value of 1 pip move per lot

        # Get symbol-specific or default risk per trade
        risk_per_trade_base = self.cfg.get_symbol_value(symbol, 'risk_per_trade', 0.005)
        risk_per_trade = self._get_dynamic_value(self.cfg.get_symbol_value(symbol, 'dynamic_risk'), auc_score, risk_per_trade_base)

        open_risk_fraction = float(total_open_risk) / float(equity) if equity > 0 else 1.0
        max_risk_allowed = max(0.0, float(self.risk_cfg.max_portfolio_risk) - open_risk_fraction)
        # Streak, exploration and drawdown (risk_scale) multipliers apply BEFORE the cap, so the cap is a hard limit.
        effective_risk = min(risk_per_trade * exploration_mult * ac_multiplier * risk_scale, max_risk_allowed)

        risk_amt = float(equity) * float(effective_risk)

        # Size on the stop that is really placed; without one, fall back to the configured ATR multiple.
        if sl_distance is None:
            atr_mult_sl = self.cfg.get_symbol_value(symbol, 'atr_multiplier_sl', 1.0)
            sl_distance = float(atr_mult_sl) * float(atr)
        sl_distance = float(sl_distance)

        if sl_distance <= 0 or pip_value <= 0 or pip_size <= 0:
            logger.warning(f"Invalid SL distance ({sl_distance}), pip_value ({pip_value}), or pip_size ({pip_size})")
            return 0.0, 0.0

        # The value at risk per lot is purely the stop distance in currency terms.
        # This matches the simpler, successful backtest model.
        sl_in_pips = sl_distance / pip_size
        risk_per_lot = sl_in_pips * pip_value

        if risk_per_lot <= 0:
            logger.warning(f"Calculated risk per lot is not positive: {risk_per_lot}")
            return 0.0, 0.0

        units = risk_amt / risk_per_lot

        volume_min = symbol_info.volume_min
        volume_step = symbol_info.volume_step
        volume_max = symbol_info.volume_max

        if not volume_step or volume_step <= 0:
            logger.warning(f"[{symbol}] Invalid volume step ({volume_step}). Skipping trade.")
            return 0.0, 0.0

        # Floor to a whole number of steps (epsilon absorbs float noise on exact multiples). Never round up:
        # that would risk more than the budget, and the minimum lot is a floor, not something to clip up to.
        steps = min(math.floor(units / volume_step + 1e-9), math.floor(volume_max / volume_step + 1e-9))
        lots = round(steps * volume_step, 8)
        if lots < volume_min - 1e-12:
            logger.info(f"Lot size {units:.4f} floors to {lots} (broker minimum {volume_min}). Skipping trade.")
            return 0.0, 0.0

        logger.info(f"Position sizing (backtest logic): equity={equity:.2f}, ATR={atr:.6f}, lots={lots:.4f}, effective_risk={effective_risk:.6f}")
        return lots, effective_risk

    # ---------- SL / TP ----------
    def stop_targets(self, price: float, atr: float, direction: str, auc_score: float, symbol: str, sl_mult: float | None = None, tp_mult: float | float | None = None):
        symbol_info = self.mt5_client.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"[{symbol}] Could not get symbol info for stop_targets.")
            return float(0.0), float(0.0)

        pip_size = symbol_info.point

        _sl_mult = sl_mult if sl_mult is not None else self.cfg.get_symbol_value(symbol, 'atr_multiplier_sl', 1.5)

        min_rr = self.cfg.get_symbol_value(symbol, "min_risk_reward_ratio", 1.2)
        required_tp_mult = _sl_mult * min_rr

        default_tp_mult = self.cfg.get_symbol_value(symbol, 'atr_multiplier_tp', 2.5)
        dynamic_tp_cfg = self.cfg.get_symbol_value(symbol, 'dynamic_tp')
        base_tp_mult = tp_mult if tp_mult is not None else self._get_dynamic_value(dynamic_tp_cfg, auc_score, default_tp_mult)

        _tp_mult = max(base_tp_mult, required_tp_mult)

        # --- CRITICAL: Ensure multipliers are positive ---
        # The TS bandit can sample negative values, which must be clamped.
        _sl_mult = max(0.1, _sl_mult)
        _tp_mult = max(0.1, _tp_mult)

        price = float(price)
        atr = float(atr)
        if direction == "long":
            sl = price - _sl_mult * atr
            tp = price + _tp_mult * atr
        else:
            sl = price + _sl_mult * atr
            tp = price - _tp_mult * atr

        price_digits = symbol_info.digits
        stops_level = symbol_info.trade_stops_level
        pip_size = symbol_info.point

        # Adjust for stops_level
        if direction == "long":
            sl = min(sl, price - stops_level * pip_size)
            tp = max(tp, price + stops_level * pip_size)
        else:  # short
            sl = max(sl, price + stops_level * pip_size)
            tp = min(tp, price - stops_level * pip_size)

        sl = round(sl, price_digits)
        tp = round(tp, price_digits)

        logger.debug(f"Stop targets (rounded): dir={direction}, price={price:.{price_digits}f}, SL={sl:.{price_digits}f}, TP={tp:.{price_digits}f}, sl_mult={_sl_mult:.2f}, tp_mult={_tp_mult:.2f}")
        return float(sl), float(tp)

    # ---------- Watchdog / cooldown helpers ----------
    def _update_equity_peak(self, equity_value: float):
        if self.equity_peak is None or equity_value > self.equity_peak:
            self.equity_peak = equity_value
            logger.debug(f"Equity peak updated: {self.equity_peak:.2f}")

    def _drawdown_exceeded(self, equity_value: float) -> bool:
        if self.equity_peak is None:
            return False
        dd = 1.0 - (equity_value / self.equity_peak) if self.equity_peak else 0.0
        # Note: block_on_drawdown is global, not per-symbol
        if dd >= getattr(self.risk_cfg, "block_on_drawdown", 0.10):
            logger.warning(f"Drawdown threshold exceeded: equity={equity_value:.2f}, peak={self.equity_peak:.2f}, drawdown={dd:.4f} >= {self.risk_cfg.block_on_drawdown}")
            return True
        return False

    def _count_consecutive_losses(self, now: datetime.datetime, lookback_hours: int = 48) -> int:
        if self.cfg.data_source != "mt5":
            return 0  # Not applicable for CSV backtesting
        """
        Query MT5 deal history in the last `lookback_hours` and compute the number
        of most recent consecutive losing closed trades (profit < 0).
        """
        try:
            since = now - timedelta(hours=lookback_hours)
            # fetch recent deals
            deals = self.mt5_client.history_deals_get(since, now)
            if not deals:
                return 0
            # Convert to list sorted by time ascending
            recs = sorted(list(deals), key=lambda d: getattr(d, "time", 0))
            # only this bot's closing deals: not deposits/withdrawals, not other systems' trades
            my_magic = self.cfg.magic_number
            closing = mt5.DEAL_ENTRY_OUT if mt5 is not None else 1
            recs = [d for d in recs if getattr(d, "magic", None) == my_magic and getattr(d, "entry", None) == closing]
            # get only deals with non-zero profit (closed)
            profits = []
            for d in recs:
                # net of commission, swap and fee, so a breakeven exit that only paid costs counts as a loss
                p = sum(float(getattr(d, f, 0.0) or 0.0) for f in ("profit", "commission", "swap", "fee"))
                # skip 0-profit deals (e.g., internal adjustments)
                if abs(p) > 1e-9:
                    profits.append(p)
            count = trailing_losses(profits)
            logger.debug(f"Consecutive losing closed trades in last {lookback_hours}h: {count}")
            return count
        except Exception as e:
            logger.exception(f"_count_consecutive_losses failed: {e}")
            return 0

    def _trigger_cooldown(self, now: datetime.datetime | None = None):
        hours = float(getattr(self.watchdog_cfg, "cooldown_hours", 1.0))
        current_time = now if now is not None else datetime.datetime.now(timezone.utc)
        self.cooldown_until = current_time + timedelta(hours=hours)
        message = f"<b>RISK ALERT:</b> Watchdog triggered cooldown until {self.cooldown_until.isoformat()}"
        logger.warning(message)
        if self.notifier:
            self.notifier.send_message(message, level="WARNING")

    def cooldown_active(self, now: datetime.datetime | None = None) -> bool:
        if self.cooldown_until is None:
            return False

        current_time = now if now is not None else datetime.datetime.now(timezone.utc)

        if current_time < self.cooldown_until:
            return True

        # cooldown finished
        self.cooldown_until = None
        return False

    def total_open_risk(self) -> float:
        """Money at risk in all tracked open positions. Takes `cache_lock`: the caller must not hold it (it is not reentrant)."""
        with self.cache_lock:
            return float(sum(p["risk"] for p in self.open_positions_cache.values()))

    def cache_snapshot(self) -> dict:
        """A deep copy of the open-position cache, taken under `cache_lock` (the caller must not hold it), so it can be
        serialised or sent somewhere slow without holding the lock."""
        with self.cache_lock:
            return copy.deepcopy(self.open_positions_cache)

    # ---------- Exposed check for trading permission ----------
    def max_positions_reached(self) -> bool:
        """True when the open-position cache already holds `risk.max_positions` positions (all symbols)."""
        limit = int(getattr(self.risk_cfg, "max_positions", 0) or 0)
        with self.cache_lock:
            count = len(self.open_positions_cache)
        if count >= limit:
            logger.info(f"Trading blocked: {count} open positions, max_positions={limit}.")
            return True
        return False

    def session_allows(self, now_utc: datetime.datetime) -> bool:
        """False when `risk.session_filter` is set and `now_utc` (real UTC) is outside its same-day window, both ends inclusive; a window
        that wraps midnight blocks everything. An invalid setting allows trading with a warning. Shared by `should_trade` and the backtester."""
        sess = self.risk_cfg.session_filter
        if sess:
            try:
                start_t = pd.to_datetime(sess["start"]).time()
                end_t = pd.to_datetime(sess["end"]).time()
                allowed = start_t <= now_utc.time() <= end_t
                if not allowed:
                    logger.info(f"Trading blocked: outside session {start_t}-{end_t}, current={now_utc.time()}")
                    return False
            except Exception:
                logger.warning("Invalid session_filter in config; allowing trades by default.")
                return True
        return True

    def should_trade(self, now_local: datetime.datetime, drawdown: float) -> bool:
        """
        Returns True if trading is allowed.
        This function now enforces:
        - equity drawdown block (block_on_drawdown); with MT5 data it fails closed when the account cannot be read
        - watchdog consecutive losses/cooldown (if enabled)
        - session filter

        `now_local` must be timezone-aware (UTC): the cooldown is stored as an aware time.
        """

        # 1) Watchdog checks (if enabled)
        if self.watchdog_cfg.enabled:
            # Cooldown check (highest priority)
            if self.cooldown_active(now=now_local):
                logger.info(f"Trading blocked: watchdog cooldown active until {self.cooldown_until.isoformat()}")
                return False

            # Consecutive losses check
            max_losses = getattr(self.watchdog_cfg, "max_consecutive_losses", None)
            if max_losses is not None and max_losses > 0:
                lost = self._count_consecutive_losses(now=now_local)
                if lost >= max_losses:
                    message = f"<b>RISK ALERT:</b> Watchdog: consecutive losses {lost} >= threshold {max_losses}. Triggering cooldown."
                    logger.warning(message)
                    if self.notifier:
                        self.notifier.send_message(message, level="WARNING")
                    self._trigger_cooldown(now=now_local)
                    return False

        # 2) Drawdown check (based on cfg.block_on_drawdown)
        if self.cfg.data_source == "mt5":
            # Look at balance deals before reading the equity: a withdrawal that lands between the two then shows as a
            # short drawdown (blocks) instead of being missed (B7)
            self._pending_flow += self._balance_flows.poll(self.mt5_client, now_local)
            try:
                acct = self.mt5_client.account_info()
            except Exception as e:
                logger.warning(f"Trading blocked: account_info() failed ({e}); cannot check drawdown.")
                return False
            if not acct:
                logger.warning("Trading blocked: account_info() returned nothing; cannot check drawdown.")
                return False
            equity = float(getattr(acct, "equity", 0.0))
            if self._pending_flow and self.equity_peak is not None:
                # before the peak update: a deposit must not lift the peak twice
                self.equity_peak = shift_peak(self.equity_peak, equity, self._pending_flow)
                logger.info(f"Balance deals of {self._pending_flow:+.2f}: equity peak now {self.equity_peak:.2f}.")
            self._pending_flow = 0.0
            self._update_equity_peak(equity)
            if self._drawdown_exceeded(equity):
                # Trigger cooldown only if watchdog is also enabled
                if self.watchdog_cfg.enabled:
                    self._trigger_cooldown(now=now_local)
                return False
        else:
            # For CSV backtesting, rely on the passed drawdown parameter
            if drawdown >= getattr(self.risk_cfg, "block_on_drawdown", 0.10):
                message = f"<b>RISK ALERT:</b> Trading blocked: drawdown {drawdown:.3f} >= {self.risk_cfg.block_on_drawdown}"
                logger.info(message)
                if self.notifier:
                    self.notifier.send_message(message, level="INFO")
                return False

        # 3) Session filter
        if not self.session_allows(now_local):
            return False

        # 4) Block on drawdown parameter (if provided separately) - This is now handled in the else block above for CSV
        # if drawdown >= getattr(self.risk_cfg, "block_on_drawdown", 0.10):
        #     message = f"<b>RISK ALERT:</b> Trading blocked: drawdown {drawdown:.3f} >= {self.risk_cfg.block_on_drawdown}"
        #     logger.info(message)
        #     if self.notifier: self.notifier.send_message(message, level="INFO")
        #     return False

        # allowed by default
        return True

    # ---------- Manage open positions: breakeven at +1R, then trailing ----------
    def manage_open_positions(self, symbol: str, current_atr: float):
        """
        Moves the stop of this symbol's open positions: to entry once the trade is +1R, then trails it by
        `trailing_atr_mult` x ATR. 1R is the stop actually placed (`sl_atr_mult` x the entry ATR, stored at entry);
        entries without it get no breakeven. Trailing starts once the trade has reached +1R or the stop is already
        at entry, and the stop is only ever tightened. Simulated (dry-run) entries are never sent to the broker.
        """
        if self.cfg.data_source != "mt5":
            return  # Not applicable for CSV backtesting

        breakeven_enabled = self.cfg.get_symbol_value(symbol, 'breakeven_at_1R', True)
        trailing_mult = self.cfg.get_symbol_value(symbol, 'trailing_atr_mult', 0.0)

        if not (breakeven_enabled or trailing_mult > 0):
            return  # No trailing logic enabled for this symbol

        with self.cache_lock:
            # a position that already left the terminal's list and waits for its closing deal (reconcile sets `close_first_seen`) has no
            # stop left to move
            positions_to_manage = [(key, dict(p)) for key, p in self.open_positions_cache.items()
                                   if p.get("symbol") == symbol and not p.get("dry_run") and not p.get("close_first_seen")]

        if not positions_to_manage:
            return

        tick = self.mt5_client.symbol_info_tick(symbol)
        if not tick:
            logger.warning(f"[{symbol}] Could not get tick for trailing stop management.")
            return

        for cache_key, pos_details in positions_to_manage:
            ticket = pos_details.get('ticket', cache_key)
            direction = pos_details.get('direction')
            entry_price = pos_details.get('entry_price')
            current_sl = pos_details.get('sl', 0.0)
            pos_atr_at_entry = pos_details.get('atr', 0.0)

            if not all([ticket, direction, entry_price, pos_atr_at_entry]):
                logger.debug(f"[{symbol}] Skipping position {ticket} due to missing details in cache.")
                continue

            is_long = direction == "long"
            exit_price = tick.bid if is_long else tick.ask
            profit_move = (exit_price - entry_price) if is_long else (entry_price - exit_price)
            placed_mult = pos_details.get('sl_atr_mult')
            reached_1r = bool(placed_mult) and profit_move >= placed_mult * pos_atr_at_entry - 1e-9  # tolerance: float noise in prices
            stop_at_entry = (current_sl >= entry_price) if is_long else (0 < current_sl <= entry_price)

            new_sl = current_sl
            if breakeven_enabled and reached_1r and not stop_at_entry:
                new_sl = entry_price
                logger.info(f"[{symbol}] Condition met to move SL to breakeven for position {ticket} at {new_sl:.5f}")

            if trailing_mult > 0 and (reached_1r or stop_at_entry):
                trailing_atr_dist = current_atr * trailing_mult
                if is_long:
                    new_sl = max(new_sl, exit_price - trailing_atr_dist)
                else:
                    new_sl = min(new_sl, exit_price + trailing_atr_dist)

            tightened = new_sl > current_sl + 1e-9 if is_long else 0 < new_sl < current_sl - 1e-9
            if new_sl > 0 and tightened:
                # --- Dynamic Freeze Level Check based on Spread ---
                symbol_info = self.mt5_client.symbol_info(symbol)
                if not symbol_info:
                    logger.warning(f"[{symbol}] Could not get symbol info for dynamic freeze level check. Skipping SL modification.")
                    continue

                # Ensure new_sl is rounded to correct precision before checks
                price_digits = symbol_info.digits
                new_sl = round(new_sl, price_digits)
                still_tighter = round(current_sl, price_digits) < new_sl if is_long else new_sl < round(current_sl, price_digits)
                if not still_tighter:
                    continue  # rounds back to the current stop: nothing to send

                # Calculate current spread
                current_spread = abs(tick.ask - tick.bid)
                # Enforce a minimum distance of 1.5x the current spread as a safety margin
                # This is more robust than a fixed point value.
                MIN_SPREAD_MULTIPLIER = 1.5
                effective_min_distance = current_spread * MIN_SPREAD_MULTIPLIER

                # For a buy position, the SL is triggered by the Bid price. For a sell, by the Ask price.
                market_price_for_sl = tick.bid if direction == "long" else tick.ask

                # Calculate distance from market price to new SL
                sl_dist_from_market = 0.0
                if direction == "long":
                    sl_dist_from_market = market_price_for_sl - new_sl
                else:  # short
                    sl_dist_from_market = new_sl - market_price_for_sl

                # Check if the new SL is too close to the market price
                if sl_dist_from_market < effective_min_distance:
                    logger.info(f"[{symbol}] Skipping SL modification for ticket {ticket}. New SL {new_sl:.{price_digits}f} is too close to market price {market_price_for_sl:.{price_digits}f} (within dynamic min distance of {effective_min_distance:.{price_digits}f}).")
                    continue  # Skip to the next position

                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": symbol,
                    "position": ticket,
                    "sl": new_sl,
                    "tp": pos_details.get('tp', 0.0),
                }

                logger.info(f"[{symbol}] Attempting to modify SL for position {ticket} to {new_sl:.{price_digits}f}")
                result = self.mt5_client.order_send(request)

                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"[{symbol}] Successfully modified SL for position {ticket}.")
                    with self.cache_lock:
                        if cache_key in self.open_positions_cache:
                            self.open_positions_cache[cache_key]['sl'] = new_sl
                else:
                    retcode = result.retcode if result else 'N/A'
                    comment = result.comment if result else 'N/A'
                    logger.error(f"[{symbol}] Failed to modify SL for position {ticket}. Request: {request}. Code: {retcode}, Comment: {comment}")
