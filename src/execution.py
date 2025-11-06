# src/execution.py
from __future__ import annotations
import os
import copy
import json
from dataclasses import dataclass
from loguru import logger # type: ignore
import time
from .ensemble import Ensemble
from .risk import RiskManager
import pandas as pd # type: ignore
from typing import List, Optional, Dict, Any # Import Dict
# from backtester import SimPosition # Import SimPosition - REMOVED as not used
import datetime
from .notifier import TelegramNotifier

@dataclass
class OrderResult:
    ok: bool
    ticket: int | None
    message: str

@dataclass
class ClosedTrade:
    """A comprehensive closed trade object for all post-trade processing."""
    ticket: int
    symbol: str
    direction: str
    lots: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    risk_fraction: float
    atr: float
    atr_idx: int
    min_prob_long_idx: int
    min_prob_short_idx: int
    entry_auc: float
    entry_equity: Optional[float] = None
    exit_equity: Optional[float] = None
    adx: float = 0.0
    macd_diff: float = 0.0
    volatility_10: float = 0.0
    dist_from_ema_200: float = 0.0

    def __repr__(self):
        return f"<ClosedTrade ticket={self.ticket}, pnl={self.pnl:.2f}>"

class Execution:
    """ Handles trade decision & order sending with retries + dry-run. """

    def __init__(self, ens_per_symbol_long: Dict[str, Ensemble], ens_per_symbol_short: Dict[str, Ensemble], risk_manager: RiskManager, mt5_client, data_manager, dry_run: bool = False, notifier: Optional[TelegramNotifier] = None):
        self.ens_per_symbol_long = ens_per_symbol_long
        self.ens_per_symbol_short = ens_per_symbol_short
        self.risk = risk_manager
        self.mt5_client = mt5_client # Store MT5 client
        self.data_manager = data_manager # Store DataManager instance
        self.dry_run = dry_run
        self.notifier = notifier
        self._open_tickets = {}   # ticket -> dict of trade details from risk.open_positions_cache
        self._seen_closed = set() # to avoid reporting the same trade twice
        self._last_deal_time = 0  # Timestamp of the last deal processed
        self.state_file = "results/open_positions_state.json" # File to persist open positions state
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True) # Ensure directory exists

    def _save_open_positions_state(self):
        """Saves the current open_positions_cache to a JSON file."""
        try:
            # Convert datetime objects to ISO format strings for JSON serialization
            serializable_cache = copy.deepcopy(self.risk.open_positions_cache)
            for pos_id, details in serializable_cache.items():
                if "entry_time" in details and isinstance(details["entry_time"], datetime.datetime):
                    details["entry_time"] = details["entry_time"].isoformat()
            
            with open(self.state_file, 'w') as f:
                json.dump(serializable_cache, f, indent=4)
            logger.info(f"Open positions state saved to {self.state_file}")
        except Exception as e:
            logger.exception(f"Failed to save open positions state: {e}")

    def _load_open_positions_state(self) -> Dict[int, Dict[str, Any]]:
        """Loads the open_positions_cache from a JSON file."""
        if not os.path.exists(self.state_file):
            return {}
        try:
            with open(self.state_file, 'r') as f:
                loaded_state = json.load(f)
            
            # Convert ISO format strings back to datetime objects
            for pos_id, details in loaded_state.items():
                if "entry_time" in details and isinstance(details["entry_time"], str):
                    details["entry_time"] = datetime.datetime.fromisoformat(details["entry_time"])
            logger.info(f"Open positions state loaded from {self.state_file}")
            return loaded_state
        except Exception as e:
            logger.exception(f"Failed to load open positions state: {e}")
            return {}


    def reconcile_open_positions_with_mt5(self):
        """
        Enhances the internal position cache by adding any new positions
        that were opened on the broker (e.g., manually) while the bot was offline.
        This function no longer handles the detection of closed trades.
        """
        logger.info("Checking for new externally-opened positions on MT5...")
        if self.dry_run:
            logger.info("[DRY-RUN] Skipping reconciliation of new external positions.")
            return

        try:
            # 1. Get all open positions from MT5 (ground truth)
            mt5_open_positions = self.mt5_client.positions_get() or []
            
            # 2. Identify positions that are on the broker but NOT in our internal cache
            for pos in mt5_open_positions:
                if pos.ticket not in self.risk.open_positions_cache:
                    # This is a new position that the bot was not tracking.
                    direction = "long" if pos.type == mt5.POSITION_TYPE_BUY else "short"
                    entry_time_dt = datetime.datetime.fromtimestamp(pos.time, tz=datetime.timezone.utc)
                    
                    # Add it to the cache with placeholder data.
                    self.risk.open_positions_cache[pos.ticket] = {
                        "risk": 0.0,
                        "ticket": pos.ticket,
                        "symbol": pos.symbol,
                        "entry_price": pos.price_open,
                        "direction": direction,
                        "lots": pos.volume,
                        "entry_time": entry_time_dt,
                        "atr": 0.0,
                        "entry_auc": 0.5,
                        "risk_fraction": 0.0,
                        "entry_equity": 0.0,
                        "sl": pos.sl,
                        "tp": pos.tp,
                        "atr_idx": -1,
                        "min_prob_long_idx": -1,
                        "min_prob_short_idx": -1,
                        "adx": 0.0,
                        "macd_diff": 0.0,
                        "volatility_10": 0.0,
                        "dist_from_ema_200": 0.0,
                    }
                    logger.info(f'Discovered new external position: Ticket={pos.ticket}, Symbol={pos.symbol}, Direction={direction}, Lots={pos.volume}')
            
            logger.info("Reconciliation of new positions complete.")

        except Exception as e:
            logger.exception(f"Failed to reconcile new external positions with MT5: {e}")
            if self.notifier: self.notifier.send_message(f"<b>ERROR:</b> Failed to reconcile new external positions with MT5: {e}", level="ERROR")

    def _send_order_with_retry(self, request: dict, retries: int = -1, delay: float = 1.0):
        num_retries = self.risk.cfg.trading_costs.defaults.retry_order_send if retries == -1 else retries
        last = None
        for attempt in range(1, num_retries + 1):
            try:
                result = self.mt5_client.order_send(request)
                last = result
                if result is not None and getattr(result, "retcode", None) == 10009:
                    return result
                logger.warning(f"Order send failed attempt {attempt}/{retries}: {result}")
            except Exception as e:
                logger.exception(f"Order send exception attempt {attempt}: {e}")
            time.sleep(delay)
        return last

    def check_closed_trades(self, latest_prices: Dict[str, float], now_utc: datetime.datetime) -> List[ClosedTrade]:
        """
        Reconciles the internal cache of open positions with the broker's state (live)
        or simulates closures based on price action (dry-run).
        Returns a list of newly detected closed trades.
        """
        closed_trades_list = []

        if self.dry_run:
            # Simulate closures in dry-run mode
            positions_to_check = list(self.risk.open_positions_cache.items()) # Iterate over a copy
            for pid, trade_details in positions_to_check:
                symbol = trade_details.get("symbol")
                direction = trade_details.get("direction")
                entry_price = trade_details.get("entry_price")
                sl = trade_details.get("sl")
                tp = trade_details.get("tp")
                lots = trade_details.get("lots")
                pip_size = trade_details.get("pip_size")
                pip_value = trade_details.get("pip_value")
                entry_time = trade_details.get("entry_time")
                risk_fraction = trade_details.get("risk_fraction")
                atr = trade_details.get("atr")
                atr_idx = trade_details.get("atr_idx")
                min_prob_long_idx = trade_details.get("min_prob_long_idx")
                min_prob_short_idx = trade_details.get("min_prob_short_idx")
                entry_auc = trade_details.get("entry_auc")
                entry_equity = trade_details.get("entry_equity")

                # Get current price from the passed latest_prices
                current_price = latest_prices.get(symbol)
                if current_price is None:
                    logger.debug(f"[{symbol}] No current price in latest_prices for dry-run closure check. Skipping {pid}.")
                    continue

                closed = False
                exit_price = 0.0
                pnl = 0.0
                closure_reason = ""

                if direction == "long":
                    if current_price <= sl:
                        closed = True
                        exit_price = sl
                        closure_reason = "SL hit"
                    elif current_price >= tp:
                        closed = True
                        exit_price = tp
                        closure_reason = "TP hit"
                elif direction == "short":
                    if current_price >= sl:
                        closed = True
                        exit_price = sl
                        closure_reason = "SL hit"
                    elif current_price <= tp:
                        closed = True
                        exit_price = tp
                        closure_reason = "TP hit"
                
                if closed:
                    # Defensive check for malformed cache data from old versions
                    if not all([pip_size, pip_value]) or pip_size <= 0 or pip_value <= 0:
                        logger.warning(f"[DRY-RUN] Cannot simulate closure for trade {pid} due to missing or invalid pip_size/pip_value in cache. Removing position.")
                        self.risk.open_positions_cache.pop(pid, None)
                        continue

                    # Calculate PnL for the simulated trade
                    if direction == "long":
                        gross_pnl = (exit_price - entry_price) / pip_size * pip_value * lots
                    else: # short
                        gross_pnl = (entry_price - exit_price) / pip_size * pip_value * lots
                    
                    # Apply transaction costs (spread and commission)
                    spread_pips = getattr(self.risk.cfg.trading_costs.defaults, 'spread_pips', 0.0)
                    commission_per_lot = getattr(self.risk.cfg.trading_costs.defaults, 'commission_per_trade', 0.0)
                    
                    transaction_cost = (spread_pips * pip_value * lots) + (commission_per_lot * lots)
                    pnl = gross_pnl - transaction_cost

                    # Get current equity for the ClosedTrade object (simulated)
                    # This is a simplification; in live, it would be actual equity.
                    # For dry-run, we use the current simulated equity from LivePerformanceMonitor
                    # which is updated in main.py after this call.
                    # For now, we'll use entry_equity + pnl as a proxy for exit_equity for this trade.
                    simulated_exit_equity = entry_equity + pnl if entry_equity is not None else pnl

                    closed_trade = ClosedTrade(
                        ticket=pid,
                        symbol=symbol,
                        direction=direction,
                        lots=lots,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        entry_time=entry_time,
                        exit_time=now_utc, # Use current time as simulated exit time
                        pnl=pnl,
                        risk_fraction=risk_fraction,
                        atr=atr,
                        atr_idx=atr_idx,
                        min_prob_long_idx=min_prob_long_idx,
                        min_prob_short_idx=min_prob_short_idx,
                        entry_auc=entry_auc,
                        entry_equity=entry_equity,
                        exit_equity=simulated_exit_equity,
                        adx=trade_details.get("adx", 0.0),
                        macd_diff=trade_details.get("macd_diff", 0.0),
                        volatility_10=trade_details.get("volatility_10", 0.0),
                        dist_from_ema_200=trade_details.get("dist_from_ema_200", 0.0)
                    )
                    closed_trades_list.append(closed_trade)
                    logger.info(f"[DRY-RUN] Simulated closed trade: {closed_trade} ({closure_reason})")

                    # Remove the now-closed position from our internal cache
                    self.risk.open_positions_cache.pop(pid, None)
            
            return closed_trades_list

        # --- LIVE MODE LOGIC (existing code) ---
        try:
            # Get the ground truth of open positions from the broker
            open_positions_on_broker = self.mt5_client.positions_get() or []
            open_position_ids_on_broker = {pos.ticket for pos in open_positions_on_broker}

            # Get the list of positions we are tracking internally
            tracked_position_ids = list(self.risk.open_positions_cache.keys())

            # Find positions that are in our cache but not in the broker's list of open positions
            closed_pids = [pid for pid in tracked_position_ids if pid not in open_position_ids_on_broker]

            for pid in closed_pids:
                trade_details = self.risk.open_positions_cache.get(pid)
                if not trade_details:
                    continue

                # Fetch the deal history for this specific closed position to find the PnL
                deals = self.mt5_client.history_deals_get(position=pid)
                if not deals:
                    logger.warning(f"Position {pid} is closed but no deal history found. Removing from cache.")
                    self.risk.open_positions_cache.pop(pid, None)
                    continue

                # Find the closing deal to get the final profit and exit details
                final_profit = 0.0
                last_exit_time = None
                last_exit_price = None
                for deal in sorted(deals, key=lambda d: d.time):
                    if deal.entry == mt5.DEAL_ENTRY_OUT:
                        # Account for commission and swap explicitly
                        final_profit += (deal.profit + deal.commission + deal.swap)
                        last_exit_time = deal.time
                        last_exit_price = deal.price

                if last_exit_time is None:
                    logger.warning(f"Position {pid} is closed but no 'out' deal found. Removing from cache.")
                    self.risk.open_positions_cache.pop(pid, None)
                    continue

                # Get current equity for the ClosedTrade object
                account_info = self.mt5_client.account_info()
                actual_equity = getattr(account_info, "equity", 0.0) if account_info else 0.0
                exit_time_dt = datetime.datetime.fromtimestamp(last_exit_time, tz=datetime.timezone.utc)

                # Create the comprehensive ClosedTrade object
                closed_trade = ClosedTrade(
                    ticket=pid,
                    symbol=trade_details.get("symbol", "UNKNOWN"),
                    direction=trade_details.get("direction", ""),
                    lots=trade_details.get("lots", 0.0),
                    entry_price=trade_details.get("entry_price", 0.0),
                    exit_price=last_exit_price or 0.0,
                    entry_time=trade_details.get("entry_time"),
                    exit_time=exit_time_dt,
                    pnl=final_profit,
                    risk_fraction=trade_details.get("risk_fraction", 0.0),
                    atr=trade_details.get("atr", 0.0),
                    atr_idx=trade_details.get("atr_idx", -1),
                    min_prob_long_idx=trade_details.get("min_prob_long_idx", -1),
                    min_prob_short_idx=trade_details.get("min_prob_short_idx", -1),
                    entry_auc=trade_details.get("entry_auc", 0.5),
                    entry_equity=trade_details.get("entry_equity", 0.0),
                    exit_equity=actual_equity,
                    adx=trade_details.get("adx", 0.0),
                    macd_diff=trade_details.get("macd_diff", 0.0),
                    volatility_10=trade_details.get("volatility_10", 0.0),
                    dist_from_ema_200=trade_details.get("dist_from_ema_200", 0.0)
                )
                closed_trades_list.append(closed_trade)
                logger.info(f"Detected closed trade via reconciliation: {closed_trade}")

                # Remove the now-closed position from our internal cache
                self.risk.open_positions_cache.pop(pid, None)

        except Exception as e:
            logger.exception(f"Failed to check/reconcile closed trades: {e}")
        
        # Sort closed trades by exit_time before returning
        closed_trades_list.sort(key=lambda trade: trade.exit_time)
        return closed_trades_list

    def trade(self, symbol: str, direction: str, lots: float, price: float, sl: float, tp: float, equity: float, pip_size: float, pip_value: float, now_utc: datetime.datetime, X: pd.DataFrame | None = None, atr: float | None = None, auc_score: float | None = 0.5, total_open_risk: float = 0.0, atr_idx: int = -1, min_prob_long_idx: int = -1, min_prob_short_idx: int = -1) -> OrderResult:
        type_map = {"long": self.mt5_client.ORDER_TYPE_BUY, "short": self.mt5_client.ORDER_TYPE_SELL}
        tick = self.mt5_client.symbol_info_tick(symbol)
        deviation_ticks = (float(tick.ask) - float(tick.bid)) if hasattr(tick, "ask") and hasattr(tick, "bid") else 0.0
        deviation = max(10, int(2 * (deviation_ticks) / (pip_size or 1e-6)))

        request = {
            "action": self.mt5_client.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lots),
            "type": type_map[direction],
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": deviation,
            "magic": self.risk.cfg.magic_number,
            "comment": "ml-bot",
            "type_time": self.mt5_client.ORDER_TIME_GTC,
            "type_filling": self.mt5_client.ORDER_FILLING_IOC,
        }

        if self.dry_run:
            simulated_ticket = int(time.time() * 1000000) # Unique enough for simulation
            logger.info(f"[DRY-RUN][{symbol}][{now_utc.strftime('%Y-%m-%d %H:%M:%S%z')}] Opened {direction} position at {price:.5f}. Lots: {lots:.2f}, SL: {sl:.5f}, TP: {tp:.5f}, AUC: {auc_score:.4f}")
            if self.notifier: self.notifier.send_message(f"[DRY-RUN] Prepared {direction} for {symbol}: lots={lots}, SL={sl}, TP={tp}", level="INFO")

            # Store comprehensive details for later SimPosition reconstruction in dry-run
            self.risk.open_positions_cache[simulated_ticket] = {
                "risk": float(equity * self.risk._get_dynamic_value(self.risk.risk_cfg.dynamic_risk, auc_score, getattr(self.risk.risk_cfg, "risk_per_trade", 0.005))), # Store the dollar amount at risk
                "entry_time": now_utc,
                "atr": atr,
                "entry_auc": auc_score,
                "risk_fraction": self.risk._get_dynamic_value(self.risk.risk_cfg.dynamic_risk, auc_score, getattr(self.risk.risk_cfg, "risk_per_trade", 0.005)),
                "entry_equity": equity,
                "sl": sl,
                "tp": tp,
                "pip_size": pip_size,
                "pip_value": pip_value,
                "atr_idx": atr_idx,
                "min_prob_long_idx": min_prob_long_idx,
                "min_prob_short_idx": min_prob_short_idx,
                "adx": float(X["adx"].iloc[-1]) if X is not None and "adx" in X.columns else 0.0,
                "macd_diff": float(X["macd_diff"].iloc[-1]) if X is not None and "macd_diff" in X.columns else 0.0,
                "volatility_10": float(X["volatility_10"].iloc[-1]) if X is not None and "volatility_10" in X.columns else 0.0,
                "dist_from_ema_200": float(X["dist_from_ema_200"].iloc[-1]) if X is not None and "dist_from_ema_200" in X.columns else 0.0,
            }
            return OrderResult(True, simulated_ticket, "Dry-run prepared")

        logger.debug(f"[{symbol}] Sending order request: {request}")
        res = self._send_order_with_retry(request)
        if res is None or getattr(res, "retcode", None) != self.mt5_client.TRADE_RETCODE_DONE:
            error_msg = f"<b>CRITICAL:</b> Order failed for {symbol} after retries: {res}"
            logger.error(error_msg)
            if self.notifier: self.notifier.send_message(error_msg, level="CRITICAL")
            return OrderResult(False, getattr(res, "order", None) if res else None, f"Order failed: {res}")

        deal_ticket = getattr(res, "deal", None)
        if not deal_ticket:
            logger.error(f"Order for {symbol} succeeded but no deal ticket returned. Cannot track position.")
            return OrderResult(False, None, "Order sent but no deal ticket.")

        # Fetch the deal to get the position_id, which is the reliable key
        deals = self.mt5_client.history_deals_get(ticket=deal_ticket)
        if not deals:
            logger.error(f"Could not fetch deal info for deal {deal_ticket}. Cannot track position.")
            return OrderResult(False, None, "Failed to fetch deal info.")
        
        position_id = deals[0].position_id

        logger.info(f"[{symbol}][{now_utc.strftime('%Y-%m-%d %H:%M:%S%z')}] Opened {direction} position at {price:.5f}. Lots: {lots:.2f}, SL: {sl:.5f}, TP: {tp:.5f}, AUC: {auc_score:.4f}")
        if self.notifier: self.notifier.send_message(f"<b>TRADE EXECUTED:</b> {direction} {lots} lots of {symbol} at {price:.5f}. SL:{sl:.5f} TP:{tp:.5f}", level="INFO")

        # compute effective risk and store in cache keyed by the reliable position_id
        try:
            risk_per_trade = self.risk._get_dynamic_value(self.risk.risk_cfg.dynamic_risk, auc_score, getattr(self.risk.risk_cfg, "risk_per_trade", 0.005))
            risk_amt = equity * risk_per_trade
            sl_distance = max(1e-6, self.risk.risk_cfg.atr_multiplier_sl * atr)
            effective_lots = (risk_amt / (sl_distance * pip_value)) if pip_value and sl_distance else 0.0
            
            # Store comprehensive details for later SimPosition reconstruction
            self.risk.open_positions_cache[position_id] = { # Use position_id as key
                "risk": float(risk_amt), # Store the dollar amount at risk
                "ticket": position_id, # Store position_id for consistency
                "symbol": symbol, # Store symbol
                "entry_price": price,
                "direction": direction,
                "lots": float(lots),
                "entry_time": now_utc, # Use current UTC time
                "atr": atr, # ATR at the time of entry
                "entry_auc": auc_score, # AUC at the time of entry
                "risk_fraction": risk_per_trade, # Store the risk_per_trade as risk_fraction
                "entry_equity": equity, # Store equity at the time of entry
                "sl": sl, # SL at entry
                "tp": tp, # TP at entry
                "pip_size": pip_size, # <-- ADD THIS
                "pip_value": pip_value, # <-- ADD THIS
                "atr_idx": atr_idx,
                "min_prob_long_idx": min_prob_long_idx,
                "min_prob_short_idx": min_prob_short_idx,
                "adx": float(X["adx"].iloc[-1]) if "adx" in X.columns else 0.0,
                "macd_diff": float(X["macd_diff"].iloc[-1]) if "macd_diff" in X.columns else 0.0,
                "volatility_10": float(X["volatility_10"].iloc[-1]) if "volatility_10" in X.columns else 0.0,
                "dist_from_ema_200": float(X["dist_from_ema_200"].iloc[-1]) if "dist_from_ema_200" in X.columns else 0.0,
                # Add inter_market_feature and mta_feature to open_positions_cache
                "inter_market_feature": float(X["inter_market_feature"].iloc[-1]) if "inter_market_feature" in X.columns else 0.0,
                "mta_feature": float(X["mta_feature"].iloc[-1]) if "mta_feature" in X.columns else 0.0,
            }
        except Exception as e:
            logger.warning(f"Could not record open position in cache: {e}")
            if self.notifier: self.notifier.send_message(f"<b>WARNING:</b> Could not record open position in cache for {symbol}: {e}", level="WARNING")

        return OrderResult(True, position_id, "OK")
