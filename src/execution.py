# src/execution.py
from __future__ import annotations
import math
import os
from typing import TYPE_CHECKING
try:
    import MetaTrader5 as mt5  # type: ignore
except ImportError:  # not available on Linux; only needed on the live (Windows) path
    mt5 = None
import copy
import json
from dataclasses import dataclass
from loguru import logger  # type: ignore
import time
from typing import List, Dict, Optional, Any
from .ensemble import Ensemble
from .risk import RiskManager
import pandas as pd  # type: ignore
import datetime
from .notifier import TelegramNotifier


# MT5 trade server return codes (MQL5 docs, "Return Codes of the Trade Server")
RETCODE_DONE = 10009
RETCODE_DONE_PARTIAL = 10010
FILLED_RETCODES = (RETCODE_DONE, RETCODE_DONE_PARTIAL)
# Rejections that guarantee nothing was executed: requote, prices changed, no quotes, too frequent requests.
# Only these may be sent again; any other result (none, an exception, timeout, a rejection) is sent once.
RETRY_SAFE_RETCODES = (10004, 10020, 10021, 10024)


# Filling policy. SymbolInfo.filling_mode is a bitmask of SYMBOL_FILLING_* flags; the request's type_filling takes
# ORDER_FILLING_* values. They are different enums (MQL5 docs, order and symbol properties).
SYMBOL_FILLING_FOK, SYMBOL_FILLING_IOC = 1, 2
ORDER_FILLING_FOK, ORDER_FILLING_IOC, ORDER_FILLING_RETURN = 0, 1, 2


@dataclass
class OrderResult:
    ok: bool
    ticket: int | None
    message: str


if TYPE_CHECKING:
    from .live_performance_monitor import LivePerformanceMonitor
from src.trade_types import ClosedTrade  # Import ClosedTrade from dedicated module


class Execution:
    """ Handles trade decision & order sending with retries + dry-run. """

    def __init__(self, ens_per_symbol_long: Dict[str, Ensemble], ens_per_symbol_short: Dict[str, Ensemble], risk_manager: RiskManager, mt5_client, data_manager, dry_run: bool = False, notifier: Optional[TelegramNotifier] = None, monitor: LivePerformanceMonitor | None = None):
        self.ens_per_symbol_long = ens_per_symbol_long
        self.ens_per_symbol_short = ens_per_symbol_short
        self.risk = risk_manager
        self.mt5_client = mt5_client  # Store MT5 client
        self.data_manager = data_manager  # Store DataManager instance
        self.dry_run = dry_run
        self.notifier = notifier
        self.monitor = monitor  # Store monitor instance
        self._open_tickets = {}   # ticket -> dict of trade details from risk.open_positions_cache
        self._seen_closed = set()  # to avoid reporting the same trade twice
        self._last_deal_time = 0  # Timestamp of the last deal processed
        self.state_file = "results/open_positions_state.json"  # File to persist open positions state
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)  # Ensure directory exists

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

    def _drop_unusable_cache_entries(self, simulated_too: bool) -> None:
        """Caller holds cache_lock. Drops entries that can never be reconciled: those without a symbol (the shape
        older dry-runs saved) and, for a live run, simulated dry-run entries that never existed at the broker."""
        for ticket, details in list(self.risk.open_positions_cache.items()):
            if not details.get("symbol") or (simulated_too and details.get("dry_run")):
                logger.warning(f"Dropping unusable cached position {ticket} (simulated or missing symbol).")
                del self.risk.open_positions_cache[ticket]

    def _close_simulated_positions(self) -> List[ClosedTrade]:
        """Dry-run counterpart of reconciliation: closes simulated positions the latest bar's close put through
        their stop or target."""
        with self.risk.cache_lock:
            self._drop_unusable_cache_entries(simulated_too=False)
            symbols = {d["symbol"] for d in self.risk.open_positions_cache.values()}
        prices = {}
        for symbol in symbols:
            price = self.data_manager.get_latest_bar_close(symbol)
            if price is not None:
                prices[symbol] = float(price)
        return self.check_closed_trades(prices, datetime.datetime.now(datetime.timezone.utc))

    def reconcile_open_positions_with_mt5(self) -> List[ClosedTrade]:
        """
        Robustly reconciles the internal position cache with the broker's state.
        This function uses a simple procedural approach to avoid race conditions and logical errors.
        1. It processes closed trades by checking which cached trades are no longer on the broker.
        2. It discovers new trades by checking which broker trades are not in the cache.
        Returns a list of ClosedTrade objects for newly detected closed trades.
        """
        if self.dry_run:
            return self._close_simulated_positions()

        import MetaTrader5 as mt5  # live path only: the package does not exist on Linux

        with self.risk.cache_lock:
            logger.info("Checking for open positions and reconciling cache with MT5...")
            self._drop_unusable_cache_entries(simulated_too=True)

            closed_trades_list = []
            try:
                # Get the ground truth of open positions from the broker
                mt5_positions = self.mt5_client.positions_get() or []
                broker_positions_map = {p.ticket: p for p in mt5_positions}
                broker_tickets = set(broker_positions_map.keys())

                # Use a copy of the cache keys for safe iteration
                cached_tickets = list(self.risk.open_positions_cache.keys())

                # Positions that carry another magic number (other EAs, manual trades) are not ours. Drop them from
                # the cache silently so they never reach the bandit or the monitor.
                my_magic = self.risk.cfg.magic_number
                for ticket in cached_tickets:
                    broker_pos = broker_positions_map.get(ticket)
                    if broker_pos is not None and broker_pos.magic != my_magic:
                        logger.info(f"Dropping foreign position {ticket} (magic {broker_pos.magic}) from the cache.")
                        del self.risk.open_positions_cache[ticket]
                cached_tickets = list(self.risk.open_positions_cache.keys())

                # --- 1. Process Closed Trades ---
                # A trade is closed if it's in our cache but NOT on the broker anymore.
                for ticket in cached_tickets:
                    if ticket not in broker_tickets:
                        pos_details = self.risk.open_positions_cache[ticket]
                        symbol = pos_details['symbol']

                        deals = self.mt5_client.history_deals_get(position=ticket)
                        if deals and all(d.magic != my_magic for d in deals):
                            logger.info(f"[{symbol}] Dropping foreign closed position {ticket} from the cache.")
                            del self.risk.open_positions_cache[ticket]
                            continue
                        pnl = 0.0
                        exit_price = 0.0
                        exit_time_dt = datetime.datetime.now(datetime.timezone.utc)

                        if deals:
                            latest_exit_time = 0
                            latest_exit_price = 0.0

                            for deal in sorted(deals, key=lambda d: d.time):
                                if deal.entry == mt5.DEAL_ENTRY_OUT:
                                    if deal.time > latest_exit_time:
                                        latest_exit_time = deal.time
                                        latest_exit_price = deal.price

                            pnl = self._position_net_pnl(deals)
                            if latest_exit_time > 0:
                                exit_time_dt = datetime.datetime.fromtimestamp(latest_exit_time, tz=datetime.timezone.utc)
                                exit_price = latest_exit_price
                        else:
                            logger.warning(f"[{symbol}] Position {ticket} closed, but no deal found in history for PnL calculation. PnL will be 0.")

                        closed_trade = ClosedTrade(
                            ticket=ticket, symbol=symbol, direction=pos_details.get('direction'),
                            lots=pos_details.get('lots'), entry_price=pos_details.get('entry_price'),
                            exit_price=exit_price, entry_time=pos_details.get("entry_time"),
                            exit_time=exit_time_dt, pnl=pnl,
                            risk_fraction=pos_details.get("risk_fraction", 0.0),
                            atr=pos_details.get("atr", 0.0), atr_idx=pos_details.get("atr_idx", -1),
                            min_prob_long_idx=pos_details.get("min_prob_long_idx", -1),
                            min_prob_short_idx=pos_details.get("min_prob_short_idx", -1),
                            entry_auc=pos_details.get("entry_auc", 0.5),
                            entry_equity=pos_details.get("entry_equity", 0.0),
                            exit_equity=self.monitor.current_equity,
                            adx=pos_details.get("adx", 0.0), macd_diff=pos_details.get("macd_diff", 0.0),
                            volatility_10=pos_details.get("volatility_10", 0.0),
                            dist_from_ema_200=pos_details.get("dist_from_ema_200", 0.0),
                            context_vector=pos_details.get("context_vector"),
                            risk_amount=pos_details.get("risk_amount"), sl_atr_mult=pos_details.get("sl_atr_mult")
                        )
                        closed_trades_list.append(closed_trade)
                        logger.info(f"[{symbol}] Detected closed trade via reconciliation: {closed_trade}")

                        # Remove from cache after processing
                        del self.risk.open_positions_cache[ticket]

                # --- 2. Discover New (External) Trades ---
                # A trade is new if it's on the broker but NOT in our cache.
                for ticket in broker_tickets:
                    if ticket not in self.risk.open_positions_cache:
                        pos = broker_positions_map[ticket]
                        if pos.magic != my_magic:
                            continue
                        direction = "long" if pos.type == mt5.POSITION_TYPE_BUY else "short"
                        entry_time_dt = datetime.datetime.fromtimestamp(pos.time, tz=datetime.timezone.utc)

                        self.risk.open_positions_cache[pos.ticket] = {
                            "risk": 0.0, "ticket": pos.ticket, "symbol": pos.symbol,
                            "entry_price": pos.price_open, "direction": direction, "lots": pos.volume,
                            "entry_time": entry_time_dt, "atr": 0.0, "entry_auc": 0.5,
                            "risk_fraction": 0.0, "entry_equity": 0.0, "sl": pos.sl, "tp": pos.tp,
                            "pip_size": self.risk.get_pip_size(pos.symbol),
                            "pip_value": self.risk.get_pip_value(pos.symbol),
                            "atr_idx": -1, "min_prob_long_idx": -1, "min_prob_short_idx": -1,
                            "adx": 0.0, "macd_diff": 0.0, "volatility_10": 0.0, "dist_from_ema_200": 0.0,
                            "inter_market_feature": 0.0, "mta_feature": 0.0
                        }
                        logger.info(f'Discovered new external position: Ticket={pos.ticket}, Symbol={pos.symbol}, Direction={direction}, Lots={pos.volume}')

                if closed_trades_list or (broker_tickets - set(cached_tickets)):
                     logger.info("Reconciliation complete. Internal cache synchronized with MT5.")
                else:
                    logger.debug("Reconciliation complete. No changes detected.")

                return closed_trades_list

            except Exception as e:
                logger.exception(f"Failed to reconcile positions with MT5: {e}")
                failure = e

        # Notify after the cache lock is released: the send is a network call.
        if self.notifier:
            self.notifier.send_message(f"<b>ERROR:</b> Failed to reconcile cache with MT5: {failure}", level="ERROR")
        return []

    @staticmethod
    def _stop_fields(price, sl, lots, atr, pip_size, pip_value) -> Dict[str, Any]:
        """Money risked at the stop that was placed (same convention as RiskManager.position_size) and that stop in ATR."""
        distance = abs(float(price) - float(sl))
        out: Dict[str, Any] = {"risk_amount": None, "sl_atr_mult": None}
        if pip_size and pip_value and pip_size > 0 and pip_value > 0 and distance > 0:
            out["risk_amount"] = float(lots) * distance / pip_size * pip_value
        if atr and atr > 0 and distance > 0:
            out["sl_atr_mult"] = distance / float(atr)
        return out

    def _cache_risk(self, symbol: str, stop_fields: Dict[str, Any], nominal: float) -> float:
        """The `risk` a cache entry carries into the portfolio cap: the money at the stop that was placed, at the lots sent.
        Without it (no usable pip value or stop) the nominal risk is used, never None: the cap sums this field."""
        amount = stop_fields.get("risk_amount")
        if amount is not None and math.isfinite(float(amount)) and amount > 0:
            return float(amount)
        logger.warning(f"[{symbol}] No money-at-stop for the cache entry; counting the nominal risk {nominal:.2f} toward the cap.")
        return float(nominal)

    def _send_order_with_retry(self, request: dict, retries: int = -1, delay: float = 1.0):
        """Sends the order, again only after a return code that guarantees it was not executed (RETRY_SAFE_RETCODES).
        A missing result, an exception, a timeout or any other code stops after one send: the order may exist at the broker,
        and the caller looks for it instead (_find_new_position). Returns the last real result, or None."""
        num_retries = self.risk.cfg.trading_costs.defaults.retry_order_send if retries == -1 else retries
        last = None
        for attempt in range(1, num_retries + 1):
            try:
                last = self.mt5_client.order_send(request)
            except Exception as e:
                logger.exception(f"Order send exception on attempt {attempt}; not sending again, the order may exist: {e}")
                return None
            retcode = getattr(last, "retcode", None)
            if retcode in FILLED_RETCODES:
                return last
            if retcode in RETRY_SAFE_RETCODES and attempt < num_retries:
                logger.warning(f"Order send rejected (retcode {retcode}) on attempt {attempt}/{num_retries}; sending again.")
                time.sleep(delay)
                continue
            logger.warning(f"Order send not filled (result {last}); not sending again.")
            return last
        return last

    @staticmethod
    def _deal_net(deal) -> float:
        """profit + commission + swap + fee of one deal, all in account currency (deal.profit is only the price move; the
        cost fields may be missing on older terminals)."""
        return sum(float(getattr(deal, field, 0.0) or 0.0) for field in ("profit", "commission", "swap", "fee"))

    @classmethod
    def _position_net_pnl(cls, deals) -> float:
        """Net result of a closed position: every deal of the position (the entry deal usually carries the commission)."""
        return sum(cls._deal_net(d) for d in deals)

    def _filling_type(self, symbol: str) -> int:
        """ORDER_FILLING_* value the symbol allows: IOC if allowed, else FOK, else RETURN (allowed in every execution mode except
        market execution). If the symbol info or its filling_mode is missing, IOC, which is what the bot always sent."""
        try:
            info = self.mt5_client.symbol_info(symbol)
        except Exception:
            info = None
        mode = getattr(info, "filling_mode", None)
        if mode is None:
            return ORDER_FILLING_IOC
        mode = int(mode)
        if mode & SYMBOL_FILLING_IOC:
            return ORDER_FILLING_IOC
        if mode & SYMBOL_FILLING_FOK:
            return ORDER_FILLING_FOK
        return ORDER_FILLING_RETURN

    def _own_positions(self, request: dict) -> Optional[Dict[int, float]]:
        """ticket -> volume of this bot's open positions on the request's symbol and side; None if they cannot be listed.
        Order and position type enums share their buy/sell values in MT5, so the request type is compared directly."""
        try:
            positions = self.mt5_client.positions_get(symbol=request["symbol"])
        except Exception:
            return None
        if positions is None:
            return None
        return {p.ticket: float(p.volume) for p in positions
                if getattr(p, "magic", None) == request["magic"] and getattr(p, "type", None) == request["type"]}

    def _find_new_position(self, request: dict, before: Optional[Dict[int, float]]):
        """After an order whose outcome is unknown: (ticket, filled volume) of a position that is new since `before`
        (a new ticket, or a netting position whose volume grew), else None. With a snapshot, a ticket missing from it is new even if
        reconcile has meanwhile adopted it into the cache (risk 0, no ATR): the caller then overwrites that entry. Without a snapshot,
        tickets the bot already tracks cannot be told apart from the new one and are skipped."""
        now = self._own_positions(request)
        if now is None:
            return None
        if before is None:
            with self.risk.cache_lock:
                known = set(self.risk.open_positions_cache)
            before = {ticket: volume for ticket, volume in now.items() if ticket in known}
        for ticket, volume in now.items():
            if ticket in before:
                if volume > before[ticket] + 1e-9:
                    return ticket, volume - before[ticket]
            else:
                return ticket, volume
        return None

    def check_closed_trades(self, latest_prices: Dict[str, float], now_utc: datetime.datetime) -> List[ClosedTrade]:
        """
        Reconciles the internal cache of open positions with the broker's state (live)
        or simulates closures based on price action (dry-run).
        Returns a list of newly detected closed trades.
        """
        with self.risk.cache_lock:
            closed_trades_list = []

            if self.dry_run:
                # Simulate closures in dry-run mode
                positions_to_check = list(self.risk.open_positions_cache.items())  # Iterate over a copy
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
                        else:  # short
                            gross_pnl = (entry_price - exit_price) / pip_size * pip_value * lots

                        # Apply transaction costs (spread and commission)
                        # One spread per round trip (bar-close prices are bid-only), in pips -> account currency.
                        spread_pips = float(self.risk.cfg.get_symbol_value(symbol, 'spread_pips', self.risk.cfg.trading_costs.defaults.spread_pips))
                        commission_per_lot = getattr(self.risk.cfg.trading_costs.defaults, 'commission_per_trade', 0.0)

                        transaction_cost = self.risk.pips_to_money(symbol, spread_pips, lots) + (commission_per_lot * lots)
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
                            exit_time=now_utc,  # Use current time as simulated exit time
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
                            context_vector=trade_details.get("context_vector"),  # NEW: Pass stored context vector
                            risk_amount=trade_details.get("risk_amount"), sl_atr_mult=trade_details.get("sl_atr_mult")
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
                    final_profit = self._position_net_pnl(deals)  # deal.profit alone excludes commission, swap and fee
                    last_exit_time = None
                    last_exit_price = None
                    for deal in sorted(deals, key=lambda d: d.time):
                        if deal.entry == mt5.DEAL_ENTRY_OUT:
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
                        dist_from_ema_200=trade_details.get("dist_from_ema_200", 0.0),
                        risk_amount=trade_details.get("risk_amount"), sl_atr_mult=trade_details.get("sl_atr_mult")
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

    def trade(self, symbol: str, direction: str, lots: float, price: float, sl: float, tp: float, equity: float, pip_size: float, pip_value: float, now_utc: datetime.datetime, X: pd.DataFrame | None = None, atr: float | None = None, auc_score: float | None = 0.5, total_open_risk: float = 0.0, atr_idx: int = -1, min_prob_long_idx: int = -1, min_prob_short_idx: int = -1, context_vector: Optional[list[float]] = None) -> OrderResult:
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
            "type_filling": self._filling_type(symbol),
        }

        if self.dry_run:
            simulated_ticket = int(time.time() * 1000000)  # Unique enough for simulation
            logger.info(f"[DRY-RUN][{symbol}][{now_utc.strftime('%Y-%m-%d %H:%M:%S%z')}] Opened {direction} position at {price:.5f}. Lots: {lots:.2f}, SL: {sl:.5f}, TP: {tp:.5f}, AUC: {auc_score:.4f}")
            if self.notifier:
                self.notifier.send_message(f"[DRY-RUN] Prepared {direction} for {symbol}: lots={lots}, SL={sl}, TP={tp}", level="INFO")

            stop_fields = self._stop_fields(price, sl, lots, atr, pip_size, pip_value)
            nominal = float(equity * self.risk._get_dynamic_value(self.risk.risk_cfg.dynamic_risk, auc_score, getattr(self.risk.risk_cfg, "risk_per_trade", 0.005)))
            cache_risk = self._cache_risk(symbol, stop_fields, nominal)
            # Store comprehensive details for later SimPosition reconstruction in dry-run
            with self.risk.cache_lock:
                self.risk.open_positions_cache[simulated_ticket] = {
                    "ticket": simulated_ticket,
                    "symbol": symbol,
                    "direction": direction,
                    "entry_price": price,
                    "lots": float(lots),
                    "dry_run": True,  # simulated: never exists at the broker
                    "risk": cache_risk,  # Money at risk at the placed stop
                    "entry_time": now_utc,
                    "atr": atr,
                    "entry_auc": auc_score,
                    "risk_fraction": self.risk._get_dynamic_value(self.risk.risk_cfg.dynamic_risk, auc_score, getattr(self.risk.risk_cfg, "risk_per_trade", 0.005)),
                    "entry_equity": equity,
                    "sl": sl,
                    "tp": tp,
                    "pip_size": pip_size,
                    "pip_value": pip_value,
                    **stop_fields,
                    "atr_idx": atr_idx,
                    "min_prob_long_idx": min_prob_long_idx,
                    "min_prob_short_idx": min_prob_short_idx,
                    "adx": float(X["adx"].iloc[-1]) if X is not None and "adx" in X.columns else 0.0,
                    "macd_diff": float(X["macd_diff"].iloc[-1]) if X is not None and "macd_diff" in X.columns else 0.0,
                    "volatility_10": float(X["volatility_10"].iloc[-1]) if X is not None and "volatility_10" in X.columns else 0.0,
                    "dist_from_ema_200": float(X["dist_from_ema_200"].iloc[-1]) if X is not None and "dist_from_ema_200" in X.columns else 0.0,
                    "context_vector": context_vector  # NEW: Store the context vector
                }
            return OrderResult(True, simulated_ticket, "Dry-run prepared")

        logger.debug(f"[{symbol}] Sending order request: {request}")
        before = self._own_positions(request)  # for recovery only; sending does not depend on it
        res = self._send_order_with_retry(request)
        retcode = getattr(res, "retcode", None)
        position_id = None
        if retcode in FILLED_RETCODES:
            if retcode == RETCODE_DONE_PARTIAL and getattr(res, "volume", 0):
                lots = float(res.volume)  # partial fill: track what was actually filled
                logger.warning(f"[{symbol}] Order only partly filled: {lots} lots.")
            deal_ticket = getattr(res, "deal", None)
            deals = self.mt5_client.history_deals_get(ticket=deal_ticket) if deal_ticket else None
            if deals:
                position_id = deals[0].position_id
            else:
                logger.error(f"[{symbol}] Order filled but the deal {deal_ticket} could not be fetched; looking for the position.")
        if position_id is None:
            found = self._find_new_position(request, before)
            if found is None:
                known_failure = res is not None and retcode not in (None, 10008, 10011, 10012, 10031, *FILLED_RETCODES)
                if known_failure:
                    error_msg = f"<b>CRITICAL:</b> Order failed for {symbol}: {res}"
                else:
                    error_msg = (f"<b>CRITICAL:</b> Order outcome UNKNOWN for {symbol} ({res}). Not sent again: "
                                 f"check the terminal for a position and close it by hand if it is unwanted.")
                logger.error(error_msg)
                if self.notifier:
                    self.notifier.send_message(error_msg, level="CRITICAL")
                return OrderResult(False, getattr(res, "order", None) if res else None, f"Order failed: {res}")
            position_id, found_lots = found
            if retcode not in FILLED_RETCODES or retcode == RETCODE_DONE_PARTIAL:
                lots = found_lots
            logger.warning(f"[{symbol}] Order result was {res}, but position {position_id} ({lots} lots) is at the broker; tracking it.")

        logger.info(f"[{symbol}][{now_utc.strftime('%Y-%m-%d %H:%M:%S%z')}] Opened {direction} position at {price:.5f}. Lots: {lots:.2f}, SL: {sl:.5f}, TP: {tp:.5f}, AUC: {auc_score:.4f}")
        if self.notifier:
            self.notifier.send_message(f"<b>TRADE EXECUTED:</b> {direction} {lots} lots of {symbol} at {price:.5f}. SL:{sl:.5f} TP:{tp:.5f}", level="INFO")

        # compute effective risk and store in cache keyed by the reliable position_id
        try:
            stop_fields = self._stop_fields(price, sl, lots, atr, pip_size, pip_value)  # lots are the filled lots here
            risk_per_trade = self.risk._get_dynamic_value(self.risk.risk_cfg.dynamic_risk, auc_score, getattr(self.risk.risk_cfg, "risk_per_trade", 0.005))
            risk_amt = self._cache_risk(symbol, stop_fields, equity * risk_per_trade)

            # Store comprehensive details for later SimPosition reconstruction
            with self.risk.cache_lock:
                self.risk.open_positions_cache[position_id] = {  # Use position_id as key
                    "risk": float(risk_amt),  # Store the dollar amount at risk
                    "ticket": position_id,  # Store position_id for consistency
                    "symbol": symbol,  # Store symbol
                    "entry_price": price,
                    "direction": direction,
                    "lots": float(lots),
                    "entry_time": now_utc,  # Use current UTC time
                    "atr": atr,  # ATR at the time of entry
                    "entry_auc": auc_score,  # AUC at the time of entry
                    "risk_fraction": risk_per_trade,  # Store the risk_per_trade as risk_fraction
                    "entry_equity": equity,  # Store equity at the time of entry
                    "sl": sl,  # SL at entry
                    "tp": tp,  # TP at entry
                    "pip_size": pip_size,  # <-- ADD THIS
                    "pip_value": pip_value,  # <-- ADD THIS
                    **stop_fields,
                    "atr_idx": atr_idx,
                    "min_prob_long_idx": min_prob_long_idx,
                    "min_prob_short_idx": min_prob_short_idx,
                    "adx": float(X["adx"].iloc[-1]) if X is not None and "adx" in X.columns else 0.0,
                    "macd_diff": float(X["macd_diff"].iloc[-1]) if X is not None and "macd_diff" in X.columns else 0.0,
                    "volatility_10": float(X["volatility_10"].iloc[-1]) if X is not None and "volatility_10" in X.columns else 0.0,
                    "dist_from_ema_200": float(X["dist_from_ema_200"].iloc[-1]) if X is not None and "dist_from_ema_200" in X.columns else 0.0,
                    # Add inter_market_feature and mta_feature to open_positions_cache
                    "inter_market_feature": float(X["inter_market_feature"].iloc[-1]) if X is not None and "inter_market_feature" in X.columns else 0.0,
                    "mta_feature": float(X["mta_feature"].iloc[-1]) if X is not None and "mta_feature" in X.columns else 0.0,
                    "context_vector": context_vector  # NEW: Store the context dictionary
                }
        except Exception as e:
            logger.warning(f"Could not record open position in cache: {e}")
            if self.notifier:
                self.notifier.send_message(f"<b>WARNING:</b> Could not record open position in cache for {symbol}: {e}", level="WARNING")

        return OrderResult(True, position_id, "OK")
