# src/symbol_processor.py
import pandas as pd
from loguru import logger
import datetime
import time
import numpy as np

from src.config import Cfg
from src.mt5_client import MT5Client
from src.data_manager import DataManager
from src.features import FeatureCfg, build_features
from src.ensemble import Ensemble
from src.risk_controller import RiskController
from src.live_performance_monitor import LivePerformanceMonitor
from src.execution import Execution
from src.utils import load_ensemble, load_optuna_params, timeframe_to_mt5_timeframe, log_symbol_specific_configs
from src.risk import RiskManager # NEW

class SymbolProcessor:
    def __init__(self, cfg: Cfg, symbol: str, mt5_client: MT5Client, risk_controller: RiskController, risk_manager: RiskManager, monitor: LivePerformanceMonitor, execution: Execution, dry_run: bool):
        self.cfg = cfg
        self.symbol = symbol
        self.mt5_client = mt5_client
        self.risk_controller = risk_controller
        self.risk_manager = risk_manager # NEW
        self.monitor = monitor
        self.dry_run = dry_run
        self.data_manager = DataManager(cfg)
        self.execution = execution

        self.ens_long: Ensemble | None = None
        self.ens_short: Ensemble | None = None
        self.feature_cfg: FeatureCfg | None = None
        self.mt5_timeframe = timeframe_to_mt5_timeframe(cfg.timeframe)

        self._load_models_and_config()

    def _load_models_and_config(self):
        logger.info(f"[{self.symbol}] Loading models and configuration...")
        # Load best feature params from optuna study
        optuna_params = load_optuna_params(self.symbol, self.cfg)
        feature_params = optuna_params.get('features', {}) if optuna_params else {}
        self.feature_cfg = FeatureCfg(**feature_params)

        # Load ensembles
        self.ens_long = load_ensemble(self.cfg, self.symbol, "long")
        self.ens_short = load_ensemble(self.cfg, self.symbol, "short")

        if self.ens_long and self.ens_long.ensemble_cv_auc_:
            logger.info(f"[{self.symbol}] Active model AUC (Long): {self.ens_long.ensemble_cv_auc_:.4f}")
        if self.ens_short and self.ens_short.ensemble_cv_auc_:
            logger.info(f"[{self.symbol}] Active model AUC (Short): {self.ens_short.ensemble_cv_auc_:.4f}")

    def _fetch_and_prepare_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | tuple[None, None, None]:
        logger.info(f"[{self.symbol}] Bootstrapping local history...")
        # Fetch initial history
        data, _, _ = self.data_manager.fetch_live(self.symbol, self.feature_cfg) # data, X, y are returned, but we only need data here
        if data.empty:
            logger.warning(f"[{self.symbol}] No data fetched for live trading. Skipping symbol.")
            return None, None, None

        # Load context data
        mta_df = None
        if self.cfg.context_features.mta.enabled:
            mta_df = self.data_manager.load_local_history(self.symbol, self.cfg.context_features.mta.timeframe)
        inter_market_df = None
        if self.cfg.context_features.inter_market.enabled:
            im_sym = self.cfg.context_features.inter_market.symbol
            inter_market_df = self.data_manager.load_local_history(im_sym, self.cfg.timeframe)

        # Build features
        X = build_features(data, self.feature_cfg, self.cfg, self.symbol, mta_df, inter_market_df)
        y = pd.DataFrame() # y is not used in live trading

        return data, X, y

    def _make_trade_decision(self, data: pd.DataFrame, X: pd.DataFrame):
        import datetime # Import datetime

        now_utc = datetime.datetime.now(datetime.timezone.utc) # Define now_utc here
        if self.ens_long is None or self.ens_short is None:
            logger.warning(f"[{self.symbol}] Ensembles not loaded. Skipping trade decision.")
            return

        last_features = X.iloc[[-2]] if (X is not None and not X.empty and len(X) >= 2) else pd.DataFrame() # Use last CLOSED bar for features
        current_bar_time = X.index[-1] # Time of the latest bar (potentially forming)
        current_close_price = data["close"].iloc[-1] # Close price of the latest bar (potentially forming)
        atr = X["atr_14"].iloc[-1] if (X is not None and not X.empty) else 0.0 # ATR of the latest bar (potentially forming)

        prob_long = self.ens_long.predict_proba(last_features).iloc[0]
        prob_short = self.ens_short.predict_proba(last_features).iloc[0]

        # Get dynamic risk parameters from RiskController
        context = {
            "vol": atr,
            "equity": self.monitor.current_equity,
            "peak_equity": self.monitor.peak_equity,
            "ensemble_auc": (self.ens_long.ensemble_cv_auc_ + self.ens_short.ensemble_cv_auc_) / 2,
            "adx": float(last_features["adx"].iloc[0]) if "adx" in last_features.columns else 0.0,
            "macd_diff": float(last_features["macd_diff"].iloc[0]) if "macd_diff" in last_features.columns else 0.0,
            "volatility_10": float(last_features["volatility_10"].iloc[0]) if "volatility_10" in last_features.columns else 0.0,
            "dist_from_ema_200": float(last_features["dist_from_ema_200"].iloc[0]) if "dist_from_ema_200" in last_features.columns else 0.0,
        }
        dynamic_risk_params = self.risk_controller.get_params(self.symbol, context)

        atr_multiplier_sl = dynamic_risk_params["atr_multiplier_sl"]
        atr_multiplier_tp = dynamic_risk_params["atr_multiplier_tp"]
        min_prob_long = dynamic_risk_params["min_prob_long"]
        min_prob_short = dynamic_risk_params["min_prob_short"]
        min_ensemble_auc = self.risk_manager.cfg.get_symbol_value(self.symbol, 'min_ensemble_auc', 0.55)

        # Define atr_idx, min_prob_long_idx, min_prob_short_idx here
        atr_idx = dynamic_risk_params["atr_idx"]
        min_prob_long_idx = dynamic_risk_params.get("min_prob_long_idx", -1)
        min_prob_short_idx = dynamic_risk_params.get("min_prob_short_idx", -1)

        direction = None
        auc_score = 0.5        
        if prob_long >= min_prob_long and self.ens_long.ensemble_cv_auc_ >= min_ensemble_auc:
            direction = "long"
            auc_score = self.ens_long.ensemble_cv_auc_
        elif prob_short >= min_prob_short and self.ens_short.ensemble_cv_auc_ >= min_ensemble_auc:
            direction = "short"
            auc_score = self.ens_short.ensemble_cv_auc_
        else:
            logger.info(f"[{self.symbol}] No trade signal details: min_prob_long={min_prob_long:.3f}, min_prob_short={min_prob_short:.3f}, min_ensemble_auc={min_ensemble_auc:.3f}, ens_long_auc={self.ens_long.ensemble_cv_auc_:.3f}, ens_short_auc={self.ens_short.ensemble_cv_auc_:.3f})")
            if prob_long >= min_prob_long and self.ens_long.ensemble_cv_auc_ < min_ensemble_auc:
                logger.info(f"[{self.symbol}] Long trade blocked due to low ensemble confidence (AUC={self.ens_long.ensemble_cv_auc_:.4f} < {min_ensemble_auc:.4f}).")
            if prob_short >= min_prob_short and self.ens_short.ensemble_cv_auc_ < min_ensemble_auc:
                logger.info(f"[{self.symbol}] Short trade blocked due to low ensemble confidence (AUC={self.ens_short.ensemble_cv_auc_:.4f} < {min_ensemble_auc:.4f}).")

        if direction:
            # Get spread for position sizing
            tick = self.mt5_client.symbol_info_tick(self.symbol)
            if not tick:
                logger.warning(f"[{self.symbol}] Could not get tick info for spread. Skipping trade.")
                return
            spread_pips = (tick.ask - tick.bid) / self.mt5_client.symbol_info(self.symbol).point
            spread_value = spread_pips * self.mt5_client.symbol_info(self.symbol).point

            # Calculate total open risk from the risk_manager's cache
            total_open_risk = sum(p["risk"] for p in self.risk_manager.open_positions_cache.values())

            # Determine pip_value and pip_size
            pip_value = self.risk_manager.get_pip_value(self.symbol)
            pip_size = self.risk_manager.get_pip_size(self.symbol)

            # Determine position size and stop/take-profit targets
            lots, effective_risk = self.risk_manager.position_size(
                self.monitor.current_equity, atr, auc_score,
                total_open_risk=total_open_risk, symbol=self.symbol,
                exploration_mult=dynamic_risk_params.get("exploration_risk_mult", 1.0),
                ac_multiplier=dynamic_risk_params.get("ac_multiplier", 1.0)
            )

            # CRITICAL: If position_size returned 0 lots (e.g., due to existing open position for symbol), skip trade execution.
            if lots <= 0:
                logger.info(f"[{self.symbol}] Trade skipped due to risk limits or position size zero (calculated lots: {lots:.4f}).")
                return # Exit early, as no trade can be executed with zero lots

            # If we reach here, a trade direction (long/short) has been identified and lots > 0
            # Proceed with obtaining tick price, calculating SL/TP, and executing the trade.
            
            # Get live tick price for trade execution
            tick = self.mt5_client.symbol_info_tick(self.symbol)
            if not tick:
                logger.warning(f"[{self.symbol}] Could not get tick info for live price. Skipping trade.")
                return # Exit if no tick info

            price = float(tick.ask) if direction == "long" else float(tick.bid) # Define price here

            sl, tp = self.risk_manager.stop_targets(
                current_close_price, atr, direction, auc_score, self.symbol,
                sl_mult=atr_multiplier_sl, tp_mult=atr_multiplier_tp
            )
            # Execute trade
            order_result = self.execution.trade(
                symbol=self.symbol,
                direction=direction,
                lots=lots,
                price=price,
                sl=sl,
                tp=tp,
                equity=self.monitor.current_equity,
                pip_size=pip_size,
                pip_value=pip_value,
                now_utc=now_utc,
                X=X,
                atr=atr,
                auc_score=auc_score,
                total_open_risk=total_open_risk,
                atr_idx=atr_idx,
                min_prob_long_idx=min_prob_long_idx,
                min_prob_short_idx=min_prob_short_idx,
                context_vector=dynamic_risk_params.get("context_vector") # NEW: Pass the context vector
            )
        # The 'else' branch for 'if prob_long/prob_short >= ...' is now implicitly handled higher up by a 'return'
        # if 'direction' remains None. Thus, no final 'else' for logging 'No trade signal' is needed here.
        

    def run_loop(self):
        logger.info(f"[{self.symbol}] Starting processing loop.")
        while True:
            try:
                # Wait for a new bar
                if not self.mt5_client.wait_for_new_bar(self.symbol, self.mt5_timeframe):
                    logger.warning(f"[{self.symbol}] Timeout or error waiting for new bar. Retrying...")
                    time.sleep(self.cfg.timeframe_minutes() * 60 / 2) # Wait half a bar duration before retrying
                    continue

                logger.info(f"[{self.symbol}] New *closed* bar detected.")

                # Fetch and prepare data
                data, X, y = self._fetch_and_prepare_data()
                if data is None:
                    time.sleep(self.cfg.timeframe_minutes() * 60) # Wait a full bar duration before retrying
                    continue

                # Reconcile open positions with MT5 (e.g., if closed externally)
                # This now returns a list of ClosedTrade objects
                closed_trades = self.execution.reconcile_open_positions_with_mt5()
                for trade in closed_trades:
                    # Add to monitor for metrics
                    self.monitor.add_closed_trade(trade)
                    # Update RiskController for learning
                    self.risk_controller.update(trade) # Assuming RiskController.update takes a ClosedTrade object

                # Make trade decisions
                self._make_trade_decision(data, X)

                # Save monitor state periodically
                self.monitor.save_state()

                # Save RiskController state periodically
                self.risk_controller.save_state(self.execution.risk.open_positions_cache)

            except Exception as e:
                logger.exception(f"[{self.symbol}] Error in processing loop: {e}")
                time.sleep(self.cfg.timeframe_minutes() * 60) # Wait a full bar duration on error to avoid rapid error looping
            
            time.sleep(1) # Small sleep to prevent busy-waiting, though wait_for_new_bar should handle most of this
