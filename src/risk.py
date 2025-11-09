# src/risk.py
from __future__ import annotations
import pandas as pd # type: ignore
import numpy as np # type: ignore
from loguru import logger # type: ignore
from .config import Cfg
import datetime
from datetime import timezone, timedelta
from typing import List, Optional # Import Optional
from .trade import SimPosition # Import SimPosition
from .notifier import TelegramNotifier # NEW import

class RiskManager:
    """
    RiskManager handles dynamic position sizing, SL/TP, portfolio exposure caps,
    open-position bookkeeping, and watchdog/cooldown behavior.

    Callers: pass `cfg` (the Cfg object) to constructor so both risk and watchdog settings are available.
    """

    def __init__(self, cfg: Cfg, mt5_client, notifier: Optional[TelegramNotifier] = None):
        self.cfg = cfg
        self.risk_cfg = cfg.risk
        self.watchdog_cfg = cfg.watchdog
        self.equity_peak: float | None = None
        self.open_positions_cache: dict[str, dict] = {}
        self.cooldown_until: datetime.datetime | None = None
        self.recently_closed_trades: List[SimPosition] = [] # New: To store closed trades for monitoring
        self.notifier = notifier # NEW
        self.mt5_client = mt5_client

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
        return symbol_info.point * symbol_info.trade_contract_size

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
    def position_size(self, equity: float, atr: float, auc_score: float, spread_value: float, total_open_risk: float = 0.0, symbol: str | None = None, exploration_mult: float = 1.0) -> tuple[float, float]:
        # CRITICAL SAFETY CHECK: Do not open a new position if one already exists for this symbol.
        if any(pos.get('symbol') == symbol for pos in self.open_positions_cache.values()):
            logger.warning(f"[{symbol}] Blocking new trade: A position is already open for this symbol.")
            return 0.0, 0.0

        # Fetch symbol info dynamically
        symbol_info = self.mt5_client.symbol_info(symbol)
        if not symbol_info:
            logger.warning(f"[{symbol}] Could not get symbol info. Cannot calculate position size.")
            return 0.0, 0.0

        pip_size = symbol_info.point
        contract_size = symbol_info.trade_contract_size
        pip_value = pip_size * contract_size
        logger.info(f"[{symbol}] position_size: pip_size={pip_size}, contract_size={contract_size}, calculated pip_value={pip_value}")

        # Get symbol-specific or default risk per trade
        risk_per_trade_base = self.cfg.get_symbol_value(symbol, 'risk_per_trade', 0.005)
        risk_per_trade = self._get_dynamic_value(self.cfg.get_symbol_value(symbol, 'dynamic_risk'), auc_score, risk_per_trade_base)
        
        max_risk_allowed = max(0.0, float(self.risk_cfg.max_portfolio_risk) - float(total_open_risk))
        effective_risk = min(risk_per_trade, max_risk_allowed)
        
        # Apply exploration multiplier
        effective_risk *= exploration_mult

        risk_amt = float(equity) * float(effective_risk)

        atr_mult_sl = self.cfg.get_symbol_value(symbol, 'atr_multiplier_sl', 1.0)
        sl_distance_atr = float(atr_mult_sl) * float(atr)
        sl_distance = sl_distance_atr + spread_value

        if sl_distance <= 0 or (pip_value is None) or pip_value <= 0:
            logger.warning("Invalid SL distance or pip_value when computing position size")
            return 0.0, 0.0

        if pip_size <= 0:
            logger.warning(f"Invalid pip_size for position sizing: {pip_size}")
            return 0.0, 0.0
        
        sl_in_pips = sl_distance / pip_size
        risk_per_lot = sl_in_pips * pip_value

        if risk_per_lot <= 0:
            logger.warning(f"Calculated risk per lot is not positive: {risk_per_lot}")
            return 0.0, 0.0

        units = risk_amt / risk_per_lot

        # --- Context-aware volume limit enforcement ---
        volume_min = symbol_info.volume_min
        volume_step = symbol_info.volume_step
        volume_max = symbol_info.volume_max

        if units < volume_min:
            logger.info(f"Live run: Calculated lot size {units:.4f} is below broker's minimum of {volume_min}. Skipping trade.") 
        # # If units is less than volume_min, but not zero, force it to volume_min
        # # This ensures we always trade the minimum if a signal is present and risk allows.
        # if 0 < units < volume_min:
        #     logger.info(f"Live run: Calculated lot size {units:.4f} is below broker's minimum of {volume_min}. Forcing to {volume_min}.")
        #     units = volume_min
        # elif units <= 0: # If units is zero or negative, skip the trade
        #     logger.info(f"Live run: Calculated lot size {units:.4f} is zero or negative. Skipping trade.")
            return 0.0, 0.0

        # Adjust lots to the nearest valid step and clip to bounds
        lots = round(units / volume_step) * volume_step
        lots = float(np.clip(lots, volume_min, volume_max))

        logger.info(f"Position sizing: equity={equity:.2f}, ATR={atr:.6f}, lots={lots:.4f}, effective_risk={effective_risk:.6f}")
        return round(lots, 2), effective_risk

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

        price = float(price)
        atr = float(atr)
        if direction == "long":
            sl = price - _sl_mult * atr
            tp = price + _tp_mult * atr
        else:
            sl = price + _sl_mult * atr
            tp = price - _tp_mult * atr

        price_digits = symbol_info.digits
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
            return 0 # Not applicable for CSV backtesting
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
            # get only deals with non-zero profit (closed)
            profits = []
            for d in recs:
                p = float(getattr(d, "profit", 0.0))
                # skip 0-profit deals (e.g., internal adjustments)
                if abs(p) > 1e-9:
                    profits.append(p)
            # count last consecutive negatives from end
            count = 0
            for p in reversed(profits):
                if p < 0:
                    count += 1
                else:
                    break
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
        if self.notifier: self.notifier.send_message(message, level="WARNING")

    def cooldown_active(self, now: datetime.datetime | None = None) -> bool:
        if self.cooldown_until is None:
            return False
        
        current_time = now if now is not None else datetime.datetime.now(timezone.utc)

        if current_time < self.cooldown_until:
            return True
        
        # cooldown finished
        self.cooldown_until = None
        return False

    # ---------- Exposed check for trading permission ----------
    def should_trade(self, now_local: pd.Timestamp, drawdown: float) -> bool:
        import MetaTrader5 as mt5  # type: ignore
        """
        Returns True if trading is allowed.
        This function now enforces:
        - equity drawdown block (block_on_drawdown)
        - watchdog consecutive losses/cooldown (if enabled)
        - session filter
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
                    if self.notifier: self.notifier.send_message(message, level="WARNING")
                    self._trigger_cooldown(now=now_local)
                    return False

        # 2) Drawdown check (based on cfg.block_on_drawdown)
        if self.cfg.data_source == "mt5":
            try:
                acct = self.mt5_client.account_info()
                if acct:
                    equity = float(getattr(acct, "equity", 0.0))
                    self._update_equity_peak(equity)
                    if self._drawdown_exceeded(equity):
                        # Trigger cooldown only if watchdog is also enabled
                        if self.watchdog_cfg.enabled:
                            self._trigger_cooldown()
                        return False
            except Exception:
                logger.debug("should_trade: account_info() unavailable for drawdown checks")
        else:
            # For CSV backtesting, rely on the passed drawdown parameter
            if drawdown >= getattr(self.risk_cfg, "block_on_drawdown", 0.10):
                message = f"<b>RISK ALERT:</b> Trading blocked: drawdown {drawdown:.3f} >= {self.risk_cfg.block_on_drawdown}"
                logger.info(message)
                if self.notifier: self.notifier.send_message(message, level="INFO")
                return False

        # 3) Session filter
        sess = self.risk_cfg.session_filter
        if sess:
            try:
                start_t = pd.to_datetime(sess["start"]).time()
                end_t = pd.to_datetime(sess["end"]).time()
                allowed = start_t <= now_local.time() <= end_t
                if not allowed:
                    logger.info(f"Trading blocked: outside session {start_t}-{end_t}, current={now_local.time()}")
                    return False
            except Exception:
                logger.warning("Invalid session_filter in config; allowing trades by default.")
                return True

        # 4) Block on drawdown parameter (if provided separately) - This is now handled in the else block above for CSV
        # if drawdown >= getattr(self.risk_cfg, "block_on_drawdown", 0.10):
        #     message = f"<b>RISK ALERT:</b> Trading blocked: drawdown {drawdown:.3f} >= {self.risk_cfg.block_on_drawdown}"
        #     logger.info(message)
        #     if self.notifier: self.notifier.send_message(message, level="INFO")
        #     return False

        # allowed by default
        return True

    # ---------- Manage open positions (unchanged mostly) ----------
    def manage_open_positions(self, symbol: str, current_atr: float):
        if self.cfg.data_source != "mt5":
            return # Not applicable for CSV backtesting
        """
        Manages trailing stops for open positions of a given symbol.
        Includes breakeven and ATR trailing logic, adapted for live trading.
        """

        breakeven_enabled = self.cfg.get_symbol_value(symbol, 'breakeven_at_1R', True)
        trailing_mult = self.cfg.get_symbol_value(symbol, 'trailing_atr_mult', 0.0)

        if not (breakeven_enabled or trailing_mult > 0):
            return # No trailing logic enabled for this symbol

        positions_to_manage = [p for p in self.open_positions_cache.values() if p.get("symbol") == symbol]

        if not positions_to_manage:
            return

        tick = self.mt5_client.symbol_info_tick(symbol)
        if not tick:
            logger.warning(f"[{symbol}] Could not get tick for trailing stop management.")
            return

        for pos_details in positions_to_manage:
            ticket = pos_details.get('ticket')
            direction = pos_details.get('direction')
            entry_price = pos_details.get('entry_price')
            current_sl = pos_details.get('sl', 0.0)
            pos_atr_at_entry = pos_details.get('atr', 0.0)

            if not all([ticket, direction, entry_price, pos_atr_at_entry]):
                logger.debug(f"[{symbol}] Skipping position {ticket} due to missing details in cache.")
                continue

            new_sl = current_sl
            exit_price = tick.bid if direction == "long" else tick.ask

            if breakeven_enabled:
                sl_mult = self.cfg.get_symbol_value(symbol, 'atr_multiplier_sl', 1.5)
                one_r_price_move = sl_mult * pos_atr_at_entry
                
                is_in_profit_for_be = (direction == "long" and exit_price >= entry_price + one_r_price_move) or \
                                     (direction == "short" and exit_price <= entry_price - one_r_price_move)
                
                is_sl_not_at_be = (direction == "long" and current_sl < entry_price) or \
                                  (direction == "short" and current_sl > entry_price)

                if is_in_profit_for_be and is_sl_not_at_be:
                    new_sl = entry_price
                    logger.info(f"[{symbol}] Condition met to move SL to breakeven for position {ticket} at {new_sl:.5f}")

            if trailing_mult > 0:
                trailing_atr_dist = current_atr * trailing_mult
                potential_new_sl = 0.0
                
                if direction == "long":
                    potential_new_sl = exit_price - trailing_atr_dist
                    if potential_new_sl > new_sl:
                        new_sl = potential_new_sl
                else: # Short position
                    potential_new_sl = exit_price + trailing_atr_dist
                    if (new_sl == 0.0) or (potential_new_sl < new_sl):
                        new_sl = potential_new_sl

            if new_sl > 0 and abs(new_sl - current_sl) > 1e-9:
                # --- Dynamic Freeze Level Check based on Spread ---
                symbol_info = self.mt5_client.symbol_info(symbol)
                if not symbol_info:
                    logger.warning(f"[{symbol}] Could not get symbol info for dynamic freeze level check. Skipping SL modification.")
                    continue

                # Ensure new_sl is rounded to correct precision before checks
                price_digits = symbol_info.digits
                new_sl = round(new_sl, price_digits)

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
                else: # short
                    sl_dist_from_market = new_sl - market_price_for_sl
                
                # Check if the new SL is too close to the market price
                if sl_dist_from_market < effective_min_distance:
                    logger.info(f"[{symbol}] Skipping SL modification for ticket {ticket}. New SL {new_sl:.{price_digits}f} is too close to market price {market_price_for_sl:.{price_digits}f} (within dynamic min distance of {effective_min_distance:.{price_digits}f}).")
                    continue # Skip to the next position

                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": ticket,
                    "sl": new_sl,
                    "tp": pos_details.get('tp', 0.0),
                }
                
                logger.info(f"[{symbol}] Attempting to modify SL for position {ticket} to {new_sl:.{price_digits}f}")
                result = self.mt5_client.order_send(request)
                
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"[{symbol}] Successfully modified SL for position {ticket}.")
                    self.open_positions_cache[ticket]['sl'] = new_sl
                else:
                    retcode = result.retcode if result else 'N/A'
                    comment = result.comment if result else 'N/A'
                    logger.error(f"[{symbol}] Failed to modify SL for position {ticket}. Request: {request}. Code: {retcode}, Comment: {comment}")
