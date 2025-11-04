# src/execution.py
from __future__ import annotations
import os
import copy
import json
from dataclasses import dataclass
import MetaTrader5 as mt5  # type: ignore
from loguru import logger # type: ignore
import time
from .ensemble import Ensemble
from .risk import RiskManager
import pandas as pd # type: ignore
from typing import List, Optional, Dict, Any # Import Dict
# from backtester import SimPosition # Import SimPosition - REMOVED as not used
import datetime
from src.alert_manager import AlertManager # NEW
from .slippage_model import calculate_dynamic_slippage # NEW

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
    combined_arm_idx: int = -1 # NEW: For contextual bandit
    dynamic_risk_base: float = 0.0 # NEW
    dynamic_risk_max: float = 0.0 # NEW
    dynamic_risk_auc_floor: float = 0.0 # NEW
    dynamic_risk_auc_ceiling: float = 0.0 # NEW
    dynamic_tp_base_mult: float = 0.0 # NEW
    dynamic_tp_max_mult: float = 0.0 # NEW
    dynamic_tp_auc_floor: float = 0.0 # NEW
    dynamic_tp_auc_ceiling: float = 0.0 # NEW

    def __repr__(self):
        return f"<ClosedTrade ticket={self.ticket}, pnl={self.pnl:.2f}>"

class Execution:
    """ Handles trade decision & order sending with retries + dry-run. """

    def __init__(self, ens_per_symbol_long: Dict[str, Ensemble], ens_per_symbol_short: Dict[str, Ensemble], risk_manager: RiskManager, mt5_client, data_manager, dry_run: bool = False, alert_manager: Optional[AlertManager] = None):
        self.ens_per_symbol_long = ens_per_symbol_long
        self.ens_per_symbol_short = ens_per_symbol_short
        self.risk = risk_manager
        self.mt5_client = mt5_client # Store MT5 client
        self.data_manager = data_manager # Store DataManager instance
        self.dry_run = dry_run
        self.alert_manager = alert_manager
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
            
            if self.dry_run:
                logger.info("[DRY-RUN] Reconciling open positions from saved state only.")
                self.risk.open_positions_cache.clear()
                for pos_id, details in loaded_state.items():
                    # Ensure entry_time is a datetime object (might be string from JSON)
                    if isinstance(details["entry_time"], str):
                        details["entry_time"] = datetime.datetime.fromisoformat(details["entry_time"])
                    self.risk.open_positions_cache[int(pos_id)] = details # Ensure key is int
                    logger.info(f'[DRY-RUN] Reconciled open position from state: Ticket={pos_id}, Symbol={details.get("symbol")}, Direction={details.get("direction")}, Lots={details.get("lots")}')
                logger.info(f"[DRY-RUN] Reconciliation complete. {len(self.risk.open_positions_cache)} open positions tracked.")
                return

            # --- LIVE MODE LOGIC ---
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
                    logger.info(f'Reconciled open position (restored from state): Ticket={pos.ticket}, Symbol={pos.symbol}, Direction={direction}, Lots={pos.volume}')
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
                    logger.info(f'Reconciled open position (new/placeholder): Ticket={pos.ticket}, Symbol={pos.symbol}, Direction={direction}, Lots={pos.volume}')
            
            logger.info(f"Reconciliation complete. {len(self.risk.open_positions_cache)} open positions tracked.")

        except Exception as e:
            logger.exception(f"Failed to reconcile open positions with MT5: {e}")
            if self.alert_manager: self.alert_manager.send_alert(f"Failed to reconcile open positions with MT5: {e}", level="ERROR", category="MT5_RECONCILIATION")

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

    def check_closed_trades(self, latest_prices: Dict[str, float]) -> List[ClosedTrade]:
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
                combined_arm_idx = trade_details.get("combined_arm_idx", -1)
                dynamic_risk_base = trade_details.get("dynamic_risk_base", 0.0)
                dynamic_risk_max = trade_details.get("dynamic_risk_max", 0.0)
                dynamic_risk_auc_floor = trade_details.get("dynamic_risk_auc_floor", 0.0)
                dynamic_risk_auc_ceiling = trade_details.get("dynamic_risk_auc_ceiling", 0.0)
                dynamic_tp_base_mult = trade_details.get("dynamic_tp_base_mult", 0.0)
                dynamic_tp_max_mult = trade_details.get("dynamic_tp_max_mult", 0.0)
                dynamic_tp_auc_floor = trade_details.get("dynamic_tp_auc_floor", 0.0)
                dynamic_tp_auc_ceiling = trade_details.get("dynamic_tp_auc_ceiling", 0.0)
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
                    # Calculate PnL for the simulated trade
                    if direction == "long":
                        gross_pnl = (exit_price - entry_price) / pip_size * pip_value * lots
                    else: # short
                        gross_pnl = (entry_price - exit_price) / pip_size * pip_value * lots
                    
                    # Apply transaction costs (spread and commission)
                    spread_pips = getattr(self.risk.cfg.trading_costs.defaults, 'spread_pips', 0.0)
                    commission_per_lot = getattr(self.risk.cfg.trading_costs.defaults, 'commission_per_trade', 0.0)

                    # Calculate dynamic slippage if enabled
                    slippage_pips_calculated = 0.0
                    if self.risk.cfg.trading_costs.defaults.dynamic_slippage_enabled:
                        # For dry-run, we use the atr from the trade details and the configured spread
                        slippage_pips_calculated = calculate_dynamic_slippage(
                            self.risk.cfg,
                            symbol,
                            atr, # ATR at the time of entry
                            spread_pips, # Configured spread as an approximation
                            lots
                        )
                    else:
                        slippage_pips_calculated = getattr(self.risk.cfg.trading_costs.defaults, 'slippage_pips', 0.0)

                    # Total transaction cost includes spread, commission, and dynamic slippage
                    transaction_cost = (spread_pips * pip_value * lots) + (commission_per_lot * lots) + (slippage_pips_calculated * pip_value * lots)
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
                        exit_time=datetime.datetime.now(datetime.timezone.utc), # Use current time as simulated exit time
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
                        dist_from_ema_200=trade_details.get("dist_from_ema_200", 0.0),
                        combined_arm_idx=combined_arm_idx,
                        dynamic_risk_base=dynamic_risk_base,
                        dynamic_risk_max=dynamic_risk_max,
                        dynamic_risk_auc_floor=dynamic_risk_auc_floor,
                        dynamic_risk_auc_ceiling=dynamic_risk_auc_ceiling,
                        dynamic_tp_base_mult=dynamic_tp_base_mult,
                        dynamic_tp_max_mult=dynamic_tp_max_mult,
                        dynamic_tp_auc_floor=dynamic_tp_auc_floor,
                        dynamic_tp_auc_ceiling=dynamic_tp_auc_ceiling
                    )
                    closed_trades_list.append(closed_trade)
                    logger.info(f"[DRY-RUN] Simulated closed trade: {closed_trade} ({closure_reason})")
                    if self.alert_manager: self.alert_manager.send_alert(f"[DRY-RUN] Closed {direction} position at {exit_price:.5f}. Entry: {entry_price:.5f}, PnL: {pnl:.2f}, Final Equity: {simulated_exit_equity:.2f}", level="INFO", category="DRY_RUN_TRADE_CLOSED")

                    # Remove the now-closed position from our internal cache
                    self.risk.open_positions_cache.pop(pid, None)
            
            return closed_trades_list

        # --- LIVE MODE LOGIC (existing code) ---
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
                    min_prob_long_idx=trade_details.get("min_prob_long_idx", -1),
                    min_prob_short_idx=trade_details.get("min_prob_short_idx", -1),
                    entry_auc=trade_details.get("entry_auc", 0.5),
                    entry_equity=trade_details.get("entry_equity", 0.0),
                    exit_equity=actual_equity,
                    adx=trade_details.get("adx", 0.0),
                    macd_diff=trade_details.get("macd_diff", 0.0),
                    volatility_10=trade_details.get("volatility_10", 0.0),
                    dist_from_ema_200=trade_details.get("dist_from_ema_200", 0.0),
                    combined_arm_idx=trade_details.get("combined_arm_idx", -1),
                    dynamic_risk_base=trade_details.get("dynamic_risk_base", 0.0),
                    dynamic_risk_max=trade_details.get("dynamic_risk_max", 0.0),
                    dynamic_risk_auc_floor=trade_details.get("dynamic_risk_auc_floor", 0.0),
                    dynamic_risk_auc_ceiling=trade_details.get("dynamic_risk_auc_ceiling", 0.0),
                    dynamic_tp_base_mult=trade_details.get("dynamic_tp_base_mult", 0.0),
                    dynamic_tp_max_mult=trade_details.get("dynamic_tp_max_mult", 0.0),
                    dynamic_tp_auc_floor=trade_details.get("dynamic_tp_auc_floor", 0.0),
                    dynamic_tp_auc_ceiling=trade_details.get("dynamic_tp_auc_ceiling", 0.0)
                )
                closed_trades_list.append(closed_trade)
                logger.info(f"Detected closed trade via reconciliation: {closed_trade}")
                if self.alert_manager: self.alert_manager.send_alert(f"Closed {closed_trade.direction} position at {closed_trade.exit_price:.5f}. Entry: {closed_trade.entry_price:.5f}, PnL: {closed_trade.pnl:.2f}, Final Equity: {closed_trade.exit_equity:.2f}", level="INFO", category="TRADE_CLOSED")

                # Remove the now-closed position from our internal cache
                self.risk.open_positions_cache.pop(pid, None)

        except Exception as e:
            logger.exception(f"Failed to check/reconcile closed trades: {e}")
        
        # Sort closed trades by exit_time before returning
        closed_trades_list.sort(key=lambda trade: trade.exit_time)
        return closed_trades_list

    def trade(self, symbol: str, direction: str, lots: float, price: float, sl: float, tp: float, equity: float, pip_size: float, pip_value: float, X: pd.DataFrame | None = None, atr: float | None = None, auc_score: float | None = 0.5, total_open_risk: float = 0.0, atr_idx: int = -1, min_prob_long_idx: int = -1, min_prob_short_idx: int = -1, trade_params: Dict[str, Any] | None = None) -> Tuple[OrderResult, Optional[Dict[str, Any]]]:

        intended_entry_price = price # Capture the intended price

        type_map = {"long": mt5.ORDER_TYPE_BUY, "short": mt5.ORDER_TYPE_SELL}
        tick = mt5.symbol_info_tick(symbol)
        
        # Capture spread at entry
        spread_at_entry_pips = (abs(float(tick.ask) - float(tick.bid)) / pip_size) if hasattr(tick, "ask") and hasattr(tick, "bid") else 0.0

        deviation_ticks = (float(tick.ask) - float(tick.bid)) if hasattr(tick, "ask") and hasattr(tick, "bid") else 0.0
        deviation = max(10, int(2 * (deviation_ticks) / (pip_size or 1e-6)))

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lots),
            "type": type_map[direction],
            "price": intended_entry_price,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": deviation,
            "magic": self.risk.cfg.magic_number,
            "comment": "ml-bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if self.dry_run:
            simulated_ticket = int(time.time() * 1000000) # Unique enough for simulation
            logger.info(f"[DRY-RUN][{symbol}][{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')}] Opened {direction} position at {intended_entry_price:.5f}. Lots: {lots:.2f}, SL: {sl:.5f}, TP: {tp:.5f}, AUC: {auc_score:.4f}")
            if self.alert_manager: 
                self.alert_manager.send_alert(f"[DRY-RUN] Opened {direction} position at {intended_entry_price:.5f}. Lots: {lots:.2f}, SL: {sl:.5f}, TP: {tp:.5f}, AUC: {auc_score:.4f}", level="INFO", category="DRY_RUN_TRADE")

            # Store comprehensive details for later SimPosition reconstruction in dry-run
            self.risk.open_positions_cache[simulated_ticket] = {
                "risk": float(equity * self.risk._get_dynamic_value(self.risk.risk_cfg.dynamic_risk, auc_score, getattr(self.risk.risk_cfg, "risk_per_trade", 0.005))), # Store the dollar amount at risk
                "ticket": simulated_ticket,
                "symbol": symbol,
                "entry_price": intended_entry_price,
                "direction": direction,
                "lots": float(lots),
                "entry_time": datetime.datetime.now(datetime.timezone.utc),
                "atr": atr,
                "entry_auc": auc_score,
                "risk_fraction": self.risk._get_dynamic_value(self.risk.risk_cfg.dynamic_risk, auc_score, getattr(self.risk.risk_cfg, "risk_per_trade", 0.005)),
                "min_prob_long_idx": min_prob_long_idx,
                "min_prob_short_idx": min_prob_short_idx,
                "combined_arm_idx": trade_params.get("combined_arm_idx", -1) if trade_params else -1,
                "dynamic_risk_base": trade_params.get("dynamic_risk_base", 0.0) if trade_params else 0.0,
                "dynamic_risk_max": trade_params.get("dynamic_risk_max", 0.0) if trade_params else 0.0,
                "dynamic_risk_auc_floor": trade_params.get("dynamic_risk_auc_floor", 0.0) if trade_params else 0.0,
                "dynamic_risk_auc_ceiling": trade_params.get("dynamic_risk_auc_ceiling", 0.0) if trade_params else 0.0,
                "dynamic_tp_base_mult": trade_params.get("dynamic_tp_base_mult", 0.0) if trade_params else 0.0,
                "dynamic_tp_max_mult": trade_params.get("dynamic_tp_max_mult", 0.0) if trade_params else 0.0,
                "dynamic_tp_auc_floor": trade_params.get("dynamic_tp_auc_floor", 0.0) if trade_params else 0.0,
                "dynamic_tp_auc_ceiling": trade_params.get("dynamic_tp_auc_ceiling", 0.0) if trade_params else 0.0,
                "adx": float(X["adx"].iloc[-1]) if X is not None and "adx" in X.columns else 0.0,
                "macd_diff": float(X["macd_diff"].iloc[-1]) if X is not None and "macd_diff" in X.columns else 0.0,
                "volatility_10": float(X["volatility_10"].iloc[-1]) if X is not None and "volatility_10" in X.columns else 0.0,
                "dist_from_ema_200": float(X["dist_from_ema_200"].iloc[-1]) if X is not None and "dist_from_ema_200" in X.columns else 0.0,
            }
            # Return dummy TCA data for dry run
            tca_data = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "symbol": symbol,
                "direction": direction,
                "lots": lots,
                "intended_entry_price": intended_entry_price,
                "actual_entry_price": intended_entry_price, # Same in dry run
                "slippage_pips": 0.0,
                "slippage_currency": 0.0,
                "spread_at_entry_pips": spread_at_entry_pips,
                "commission_per_trade": getattr(self.risk.cfg.trading_costs.defaults, 'commission_per_trade', 0.0),
                "total_transaction_cost_currency": (spread_at_entry_pips * pip_value * lots) + (getattr(self.risk.cfg.trading_costs.defaults, 'commission_per_trade', 0.0) * lots),
                "order_type": "DEAL",
                "fill_type": "IOC",
                "deviation_pips": deviation,
                "retries_taken": 0,
                "entry_auc": auc_score,
                "entry_equity": equity,
                "position_id": simulated_ticket,
            }
            return OrderResult(True, simulated_ticket, "Dry-run prepared"), tca_data

        # --- LIVE TRADING --- 
        num_retries_attempted = 0
        res = None
        for attempt in range(1, self.risk.cfg.trading_costs.defaults.retry_order_send + 1):
            res = mt5.order_send(request)
            num_retries_attempted = attempt
            if res is not None and getattr(res, "retcode", None) == getattr(mt5, "TRADE_RETCODE_DONE", 10009):
                break
            logger.warning(f"Order send failed attempt {attempt}/{self.risk.cfg.trading_costs.defaults.retry_order_send}: {res}")
            time.sleep(1) # Small delay before retry

        if res is None or getattr(res, "retcode", None) != getattr(mt5, "TRADE_RETCODE_DONE", 10009):
            error_msg = f"Order failed for {symbol} after {num_retries_attempted} retries: {res}"
            logger.error(error_msg)
            if self.alert_manager: self.alert_manager.send_alert(error_msg, level="CRITICAL", category="ORDER_FAILURE")
            return OrderResult(False, getattr(res, "order", None) if res else None, f"Order failed: {res}"), None

        deal_ticket = getattr(res, ("deal" if res.deal else "order"), None) # Use deal ticket if available, else order ticket
        if not deal_ticket:
            logger.error(f"Order for {symbol} succeeded but no deal ticket returned. Cannot track position for TCA.")
            return OrderResult(False, None, "Order sent but no deal ticket."), None

        # Fetch the deal to get the position_id, which is the reliable key
        deals = mt5.history_deals_get(ticket=deal_ticket) # This deal corresponds to the order execution
        position_id = deals[0].position_id if deals else None

        if not position_id:
            logger.error(f"Could not fetch deal info for deal {deal_ticket}. Cannot track position for TCA.")
            return OrderResult(False, None, "Failed to fetch deal info."), None

        # Now fetch the actual position to get the filled price
        positions = mt5.positions_get(ticket=position_id)
        if not positions:
            logger.error(f"Could not fetch position info for ID {position_id}. Cannot track position for TCA.")
            return OrderResult(False, position_id, "Failed to fetch position info."), None
        
        position_obj = positions[0]
        actual_entry_price = position_obj.price_open

        # Calculate Slippage
        if direction == "long":
            slippage_pips = (actual_entry_price - intended_entry_price) / pip_size
        else: # short
            slippage_pips = (intended_entry_price - actual_entry_price) / pip_size
        slippage_currency = slippage_pips * pip_value * lots

        commission_per_lot = getattr(self.risk.cfg.trading_costs.defaults, 'commission_per_trade', 0.0)
        total_transaction_cost_currency = slippage_currency + (spread_at_entry_pips * pip_value * lots) + (commission_per_lot * lots)

        logger.info(f"[{symbol}][{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')}] Opened {direction} position at {actual_entry_price:.5f} (intended {intended_entry_price:.5f}). Lots: {lots:.2f}, SL: {sl:.5f}, TP: {tp:.5f}, AUC: {auc_score:.4f} Slippage: {slippage_pips:.2f} pips")
        if self.alert_manager: self.alert_manager.send_alert(f"Opened {direction} position at {actual_entry_price:.5f}. Lots: {lots:.2f}, SL: {sl:.5f}, TP: {tp:.5f}, AUC: {auc_score:.4f}", level="INFO", category="TRADE_EXECUTION")

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
                "entry_price": actual_entry_price, # Store actual entry price
                "direction": direction,
                "lots": float(lots),
                "entry_time": datetime.datetime.now(datetime.timezone.utc), # Use current UTC time
                "atr": atr, # ATR at the time of entry
                "entry_auc": auc_score, # AUC at the time of entry
                "risk_fraction": risk_per_trade, # Store the risk_per_trade as risk_fraction
                "entry_equity": equity, # Store equity at the time of entry
                "sl": sl, # SL at entry
                "tp": tp, # TP at entry
                "pip_size": pip_size,
                "pip_value": pip_value,
                "atr_idx": atr_idx,
                "min_prob_long_idx": min_prob_long_idx,
                "min_prob_short_idx": min_prob_short_idx,
                "combined_arm_idx": trade_params.get("combined_arm_idx", -1) if trade_params else -1,
                "dynamic_risk_base": trade_params.get("dynamic_risk_base", 0.0) if trade_params else 0.0,
                "dynamic_risk_max": trade_params.get("dynamic_risk_max", 0.0) if trade_params else 0.0,
                "dynamic_risk_auc_floor": trade_params.get("dynamic_risk_auc_floor", 0.0) if trade_params else 0.0,
                "dynamic_risk_auc_ceiling": trade_params.get("dynamic_risk_auc_ceiling", 0.0) if trade_params else 0.0,
                "dynamic_tp_base_mult": trade_params.get("dynamic_tp_base_mult", 0.0) if trade_params else 0.0,
                "dynamic_tp_max_mult": trade_params.get("dynamic_tp_max_mult", 0.0) if trade_params else 0.0,
                "dynamic_tp_auc_floor": trade_params.get("dynamic_tp_auc_floor", 0.0) if trade_params else 0.0,
                "dynamic_tp_auc_ceiling": trade_params.get("dynamic_tp_auc_ceiling", 0.0) if trade_params else 0.0,
                "adx": float(X["adx"].iloc[-1]) if X is not None and "adx" in X.columns else 0.0,
                "macd_diff": float(X["macd_diff"].iloc[-1]) if X is not None and "macd_diff" in X.columns else 0.0,
                "volatility_10": float(X["volatility_10"].iloc[-1]) if X is not None and "volatility_10" in X.columns else 0.0,
                "dist_from_ema_200": float(X["dist_from_ema_200"].iloc[-1]) if X is not None and "dist_from_ema_200" in X.columns else 0.0,
                # Add inter_market_feature and mta_feature to open_positions_cache
                "inter_market_feature": float(X["inter_market_feature"].iloc[-1]) if "inter_market_feature" in X.columns else 0.0,
                "mta_feature": float(X["mta_feature"].iloc[-1]) if "mta_feature" in X.columns else 0.0,
            }
        except Exception as e:
            logger.warning(f"Could not record open position in cache: {e}")
            if self.alert_manager: self.alert_manager.send_alert(f"Could not record open position in cache for {symbol}: {e}", level="WARNING", category="POSITION_CACHE")

        tca_data = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "symbol": symbol,
            "direction": direction,
            "lots": lots,
            "intended_entry_price": intended_entry_price,
            "actual_entry_price": actual_entry_price,
            "slippage_pips": slippage_pips,
            "slippage_currency": slippage_currency,
            "spread_at_entry_pips": spread_at_entry_pips,
            "commission_per_trade": commission_per_lot,
            "total_transaction_cost_currency": total_transaction_cost_currency,
            "order_type": "DEAL", # Always DEAL for market orders
            "fill_type": "IOC", # Always IOC for this bot
            "deviation_pips": deviation,
            "retries_taken": num_retries_attempted,
            "entry_auc": auc_score,
            "entry_equity": equity,
            "position_id": position_id,
        }

        return OrderResult(True, position_id, "OK"), tca_data
