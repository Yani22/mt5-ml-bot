# src/live_performance_monitor.py
from __future__ import annotations
from loguru import logger  # type: ignore
import datetime
from collections import deque
from typing import List, Optional, Tuple
import json # NEW
import os # NEW

from src.config import Cfg
from src.trade_types import ClosedTrade # Import the new ClosedTrade dataclass

class LivePerformanceMonitor:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.closed_trades: deque[ClosedTrade] = deque() # Use deque for efficient appending/popping
        self.equity_curve: deque[Tuple[datetime.datetime, float]] = deque()
        self.peak_equity: float = cfg.initial_equity # Assuming initial_equity is set in Cfg or passed
        self.current_equity: float = cfg.initial_equity
        self.last_check_time: Optional[datetime.datetime] = None
        self.last_ensemble_auc: float = 0.0 # To track the latest AUC from retraining

        # Ensure initial_equity is set in Cfg or handle it
        if not hasattr(cfg, 'initial_equity'):
            logger.warning("Cfg does not have 'initial_equity'. Initializing with 100.0.")
            self.cfg.initial_equity = 100.0
            self.peak_equity = 100.0
            self.current_equity = 100.0

        logger.info(f"LivePerformanceMonitor initialized with initial equity: {self.current_equity}")

    def update_equity(self, timestamp: datetime.datetime, new_equity: float):
        self.current_equity = new_equity
        self.equity_curve.append((timestamp, new_equity))
        self.peak_equity = max(self.peak_equity, new_equity)

        # Trim equity_curve to lookback_days
        min_timestamp = timestamp - datetime.timedelta(days=self.cfg.monitoring.lookback_days)
        while self.equity_curve and self.equity_curve[0][0] < min_timestamp:
            self.equity_curve.popleft()

    def sync_equity(self, new_equity: float):
        """
        For startup synchronization. Sets the current equity and updates the peak
        without adding a point to the historical equity curve.
        """
        self.current_equity = new_equity
        self.peak_equity = max(self.peak_equity, new_equity)
        logger.info(f"Live monitor equity synchronized to: {new_equity}")

    def add_closed_trade(self, trade: ClosedTrade):
        self.closed_trades.append(trade)

        # Trim closed_trades to lookback_days
        min_timestamp = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=self.cfg.monitoring.lookback_days)
        while self.closed_trades and self.closed_trades[0].exit_time < min_timestamp:
            self.closed_trades.popleft()

    def update_ensemble_auc(self, auc: float):
        self.last_ensemble_auc = auc

    def save_state(self):
        state_path = self.cfg.monitoring.monitor_state_file
        try:
            # Manually build a serializable list of closed trades
            closed_trades_data = []
            for trade in self.closed_trades:
                trade_data = trade.__dict__.copy()
                # Convert datetime objects to ISO format strings for JSON serialization
                if 'entry_time' in trade_data and isinstance(trade_data['entry_time'], datetime.datetime):
                    trade_data['entry_time'] = trade_data['entry_time'].isoformat()
                if 'exit_time' in trade_data and isinstance(trade_data['exit_time'], datetime.datetime):
                    trade_data['exit_time'] = trade_data['exit_time'].isoformat()
                closed_trades_data.append(trade_data)

            equity_curve_data = [(ts.isoformat(), eq) for ts, eq in self.equity_curve]

            self.last_check_time = datetime.datetime.now(datetime.timezone.utc) # Update last_check_time before saving

            state = {
                "closed_trades": closed_trades_data,
                "equity_curve": equity_curve_data,
                "peak_equity": self.peak_equity,
                "current_equity": self.current_equity,
                "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None,
                "last_ensemble_auc": self.last_ensemble_auc,
            }
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=4)
            logger.debug(f"LivePerformanceMonitor state saved to {state_path}")
        except Exception as e:
            logger.error(f"Failed to save LivePerformanceMonitor state: {e}")

    def load_state(self):
        state_path = self.cfg.monitoring.monitor_state_file
        if not os.path.exists(state_path):
            logger.info(f"No existing state file found at {state_path}. Starting fresh.")
            return

        try:
            with open(state_path, 'r') as f:
                state = json.load(f)

            # Reconstruct deque and ClosedTrade objects
            self.closed_trades.clear()
            for trade_data in state.get("closed_trades", []):
                trade = ClosedTrade(
                    ticket=trade_data.get("ticket"),
                    symbol=trade_data.get("symbol"),
                    direction=trade_data.get("direction"),
                    lots=trade_data.get("lots"),
                    entry_price=trade_data.get("entry_price"),
                    exit_price=trade_data.get("exit_price"),
                    entry_time=datetime.datetime.fromisoformat(trade_data["entry_time"]) if trade_data.get("entry_time") else None,
                    exit_time=datetime.datetime.fromisoformat(trade_data["exit_time"]) if trade_data.get("exit_time") else None,
                    pnl=trade_data.get("pnl"),
                    risk_fraction=trade_data.get("risk_fraction"),
                    atr=trade_data.get("atr"),
                    atr_idx=trade_data.get("atr_idx"),
                    min_prob_long_idx=trade_data.get("min_prob_long_idx"),
                    min_prob_short_idx=trade_data.get("min_prob_short_idx"),
                    entry_auc=trade_data.get("entry_auc"),
                    entry_equity=trade_data.get("entry_equity"),
                    exit_equity=trade_data.get("exit_equity"),
                    adx=trade_data.get("adx"),
                    macd_diff=trade_data.get("macd_diff"),
                    volatility_10=trade_data.get("volatility_10"),
                    dist_from_ema_200=trade_data.get("dist_from_ema_200")
                )
                self.closed_trades.append(trade)

            self.equity_curve.clear()
            for ts_str, eq in state.get("equity_curve", []):
                self.equity_curve.append((datetime.datetime.fromisoformat(ts_str), eq))

            self.peak_equity = state.get("peak_equity", self.cfg.initial_equity)
            self.current_equity = state.get("current_equity", self.cfg.initial_equity)
            self.last_check_time = datetime.datetime.fromisoformat(state["last_check_time"]) if state.get("last_check_time") else None
            self.last_ensemble_auc = state.get("last_ensemble_auc", 0.0)

            logger.debug(f"LivePerformanceMonitor state loaded from {state_path}")
        except Exception as e:
            logger.error(f"Failed to load LivePerformanceMonitor state from {state_path}: {e}")
            # Optionally, re-initialize to a clean state if loading fails
            self.__init__(self.cfg) # Re-initialize to default state
