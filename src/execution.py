# src/execution.py
from __future__ import annotations
import os
import copy
import json
from dataclasses import dataclass
import MetaTrader5 as mt5  # type: ignore
from loguru import logger
import time
from .ensemble import Ensemble
from .risk import RiskManager
import pandas as pd
import numpy as np
from typing import List, Optional, Dict # Import Dict
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
    min_prob_idx: int
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

    def __init__(self, ens_per_symbol_long: Dict[str, Ensemble], ens_per_symbol_short: Dict[str, Ensemble], risk_manager: RiskManager, mt5_client, dry_run: bool = False, notifier: Optional[TelegramNotifier] = None):
        self.ens_per_symbol_long = ens_per_symbol_long
        self.ens_per_symbol_short = ens_per_symbol_short
        self.risk = risk_manager
        self.mt5_client = mt5_client # Store MT5 client
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
        Queries MT5 for all currently open positions and updates the internal
        open_positions_cache to reflect the ground truth from the broker.
        This is crucial for maintaining state across bot restarts.
        It attempts to restore detailed information from a saved state.
        """
        logger.info("Reconciling open positions with MT5...")
        try:
            # 1. Load previously saved detailed state for open positions
            loaded_state = self._load_open_positions_state()
            
            # 2. Get all open positions from MT5 (ground truth)
            mt5_open_positions = mt5.positions_get() or []
            
            # 3. Clear the existing internal cache
            self.risk.open_positions_cache.clear()

            # 4. Populate the internal cache with positions from MT5, prioritizing loaded_state details
            for pos in mt5_open_positions:
                direction = "long" if pos.type == mt5.POSITION_TYPE_BUY else "short"
                entry_time_dt = datetime.datetime.fromtimestamp(pos.time, tz=datetime.timezone.utc)

                # Check if this position was in our previously saved state
                if pos.ticket in loaded_state:
                    # Restore full details from loaded state
                    self.risk.open_positions_cache[pos.ticket] = loaded_state[pos.ticket]
                    # Ensure entry_time is a datetime object (might be string from JSON)
                    if isinstance(self.risk.open_positions_cache[pos.ticket]["entry_time"], str):
                        self.risk.open_positions_cache[pos.ticket]["entry_time"] = datetime.datetime.fromisoformat(self.risk.open_positions_cache[pos.ticket]["entry_time"])
                    logger.info(f"Reconciled open position (restored from state): Ticket={pos.ticket}, Symbol={pos.symbol}, Direction={direction}, Lots={pos.volume}")
                else:
                    # New position or position opened while bot was down, use placeholders
                    self.risk.open_positions_cache[pos.ticket] = {
                        "risk": 0.0, # Cannot infer from MT5 position directly, will be updated on next trade decision
                        "ticket": pos.ticket,
                        "symbol": pos.symbol,
                        "entry_price": pos.price_open,
                        "direction": direction,
                        "lots": pos.volume,
                        "entry_time": entry_time_dt,
                        "atr": 0.0, # Placeholder, will be updated on next trade decision
                        "entry_auc": 0.5, # Placeholder, will be updated on next trade decision
                        "risk_fraction": 0.0, # Placeholder, will be updated on next trade decision
                        "entry_equity": 0.0, # Placeholder, will be updated on next trade decision
                        "sl": pos.sl,
                        "tp": pos.tp,
                        "atr_idx": -1, # Placeholder
                        "min_prob_idx": -1, # Placeholder
                        "adx": 0.0, # Placeholder
                        "macd_diff": 0.0, # Placeholder
                        "volatility_10": 0.0, # Placeholder
                        "dist_from_ema_200": 0.0, # Placeholder
                        "inter_market_feature": 0.0, # Placeholder
                        "mta_feature": 0.0, # Placeholder
                    }
                    logger.info(f"Reconciled open position (new/placeholder): Ticket={pos.ticket}, Symbol={pos.symbol}, Direction={direction}, Lots={pos.volume}")
            
            logger.info(f"Reconciliation complete. {len(self.risk.open_positions_cache)} open positions tracked.")

        except Exception as e:
            logger.exception(f"Failed to reconcile open positions with MT5: {e}")
            if self.notifier: self.notifier.send_message(f"<b>ERROR:</b> Failed to reconcile open positions with MT5: {e}", level="ERROR")

    def _send_order_with_retry(self, request: dict, retries: int = -1, delay: float = 1.0):
        num_retries = self.risk.cfg.trading_costs.defaults.retry_order_send if retries == -1 else retries
        last = None
        for attempt in range(1, num_retries + 1):
            try:
                result = mt5.order_send(request)
                last = result
                if result is not None and getattr(result, "retcode", None) == getattr(mt5, "TRADE_RETCODE_DONE", 10009):
                    return result
                logger.warning(f"Order send failed attempt {attempt}/{retries}: {result}")
            except Exception as e:
                logger.exception(f"Order send exception attempt {attempt}: {e}")
            time.sleep(delay)
        return last

    def check_closed_trades(self) -> List[ClosedTrade]:
        """
        Reconciles the internal cache of open positions with the broker's state.
        Returns a list of newly detected closed trades.
        """
        closed_trades_list = []
        try:
            # Get the ground truth of open positions from the broker
            open_positions_on_broker = mt5.positions_get() or []
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
                deals = mt5.history_deals_get(position=pid)
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
                account_info = mt5.account_info()
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
                    min_prob_idx=trade_details.get("min_prob_idx", -1),
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

    def trade(self, symbol: str, direction: str, lots: float, price: float, sl: float, tp: float, X: pd.DataFrame | None = None, atr: float | None = None, auc_score: float | None = 0.5, total_open_risk: float = 0.0, atr_idx: int = -1, min_prob_idx: int = -1) -> OrderResult:


        type_map = {"long": mt5.ORDER_TYPE_BUY, "short": mt5.ORDER_TYPE_SELL}
        deviation_ticks = (float(tick.ask) - float(tick.bid)) if hasattr(tick, "ask") and hasattr(tick, "bid") else 0.0
        deviation = max(10, int(2 * (deviation_ticks) / (pip_size or 1e-6)))

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lots),
            "type": type_map[direction],
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": deviation,
            "magic": self.risk.cfg.magic_number,
            "comment": "ml-bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if self.dry_run:
            logger.info(f"[DRY-RUN] Prepared {direction} for {symbol}: lots={lots}, SL={sl}, TP={tp}")
            if self.notifier: self.notifier.send_message(f"[DRY-RUN] Prepared {direction} for {symbol}: lots={lots}, SL={sl}, TP={tp}", level="INFO")
            return OrderResult(True, None, "Dry-run prepared")

        logger.debug(f"[{symbol}] Sending order request: {request}")
        res = self._send_order_with_retry(request)
        if res is None or getattr(res, "retcode", None) != getattr(mt5, "TRADE_RETCODE_DONE", 10009):
            error_msg = f"<b>CRITICAL:</b> Order failed for {symbol} after retries: {res}"
            logger.error(error_msg)
            if self.notifier: self.notifier.send_message(error_msg, level="CRITICAL")
            return OrderResult(False, getattr(res, "order", None) if res else None, f"Order failed: {res}")

        deal_ticket = getattr(res, "deal", None)
        if not deal_ticket:
            logger.error(f"Order for {symbol} succeeded but no deal ticket returned. Cannot track position.")
            return OrderResult(False, None, "Order sent but no deal ticket.")

        # Fetch the deal to get the position_id, which is the reliable key
        deals = mt5.history_deals_get(ticket=deal_ticket)
        if not deals:
            logger.error(f"Could not fetch deal info for deal {deal_ticket}. Cannot track position.")
            return OrderResult(False, None, "Failed to fetch deal info.")
        
        position_id = deals[0].position_id

        logger.info(f"Order executed: ticket={getattr(res, 'order', None)}, position_id={position_id}, dir={direction}, lots={lots}, SL={sl}, TP={tp}")
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
                "entry_time": datetime.datetime.now(datetime.timezone.utc), # Use current UTC time
                "atr": atr, # ATR at the time of entry
                "entry_auc": auc_score, # AUC at the time of entry
                "risk_fraction": risk_per_trade, # Store the risk_per_trade as risk_fraction
                "entry_equity": equity, # Store equity at the time of entry
                "sl": sl, # SL at entry
                "tp": tp, # TP at entry
                "atr_idx": atr_idx,
                "min_prob_idx": min_prob_idx,
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
