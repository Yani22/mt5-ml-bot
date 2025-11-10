# backtester.py
from __future__ import annotations
import pandas as pd # type: ignore
from loguru import logger # type: ignore
import os
import datetime
import quantstats as qs # type: ignore
import optuna # type: ignore
import numpy as np  # type: ignore

from src.config import Cfg
from src.features import FeatureCfg
from src.risk import RiskManager
from src.utils import get_training_data, load_ensemble, save_ensemble, setup_logging, safe_retrain_ensemble, load_optuna_params, log_symbol_specific_configs
from src.trade import SimPosition
from src.risk_controller import RiskController
from src.trade_types import ClosedTrade # NEW: Import ClosedTrade for backtester

class HybridBacktester:
    """Adaptive hybrid backtester mirroring main_hybrid_adaptive.py logic."""

    def __init__(self, cfg: Cfg, broker_client=None):
        self.logged_low_confidence = set()
        self.logged_skips = set()
        self.cfg = cfg
        log_symbol_specific_configs(self.cfg) # NEW
        self.equity = cfg.backtesting.initial_equity
        self.initial_equity = cfg.backtesting.initial_equity # Store initial equity for drawdown pruning
        self.consecutive_losses = 0 # NEW: Efficiently track consecutive losses
        self.positions: list[SimPosition] = []
        self.equity_curve = []
        self.risk_manager = RiskManager(cfg, broker_client)
        self.bar_counters = {sym: 0 for sym in cfg.symbols}
        self.risk_controller = RiskController(cfg) # Instantiate RiskController
        self.risk_controller.load_state() # Load previous state if it exists
        self.ts_param_history = [] # To store Thompson Sampling parameter evolution
        self.save_state_every_bars = getattr(cfg, "save_ts_state_every_bars", 500)
        ts_ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        symbol_str = self.cfg.symbols[0].replace('#', '') # Use the exact symbol string from config, sanitized
        self.backtest_ts_state_file = f"results/ts_risk_controller_state_backtest_{symbol_str}_{ts_ts}.json"
        self.ts_history_csv = f"results/ts_param_evolution_backtest_{symbol_str}_{ts_ts}.csv"
        os.makedirs("results", exist_ok=True)

        logger.info(f"Initializing backtester with starting equity: {self.equity}")
        self.ens_per_symbol_long = {sym: load_ensemble(cfg, sym, "long") for sym in cfg.symbols}
        self.ens_per_symbol_short = {sym: load_ensemble(cfg, sym, "short") for sym in cfg.symbols}

    def _manage_trailing_stops(self, sym: str, row: pd.Series, atr: float):
        """Simulated version of the live trailing stop logic."""
        risk_cfg = self.risk_manager.risk_cfg
        if not (risk_cfg.breakeven_at_1R or risk_cfg.trailing_atr_mult > 0):
            return # No trailing logic enabled

        for pos in [p for p in self.positions if p.symbol == sym and p.status == "open"]:
            price = row["close"]
            new_sl = pos.sl

            # --- Breakeven Logic ---
            if risk_cfg.breakeven_at_1R:
                one_r_price_move = risk_cfg.atr_multiplier_sl * pos.atr
                if pos.direction == "long" and price >= pos.entry_price + one_r_price_move and pos.sl < pos.entry_price:
                    new_sl = pos.entry_price
                    logger.info(f"[{sym}] Moving SL to breakeven for long position at {new_sl:.5f}")
                elif pos.direction == "short" and price <= pos.entry_price - one_r_price_move and pos.sl > pos.entry_price:
                    new_sl = pos.entry_price
                    logger.info(f"[{sym}] Moving SL to breakeven for short position at {new_sl:.5f}")

            # --- ATR Trailing Logic ---
            if risk_cfg.trailing_atr_mult > 0:
                trailing_atr_dist = atr * risk_cfg.trailing_atr_mult
                if pos.direction == "long":
                    potential_new_sl = price - trailing_atr_dist
                    if potential_new_sl > new_sl:
                        new_sl = potential_new_sl
                        logger.debug(f"[{sym}] Trailing SL for long position to {new_sl:.5f}")
                else: # Short position
                    potential_new_sl = price + trailing_atr_dist
                    if potential_new_sl < new_sl:
                        new_sl = potential_new_sl
                        logger.debug(f"[{sym}] Trailing SL for short position to {new_sl:.5f}")
            
            pos.sl = new_sl

    def _update_positions(self, sym, row):
        """Check open positions for SL/TP, calculate PnL, and update equity using sequential reconstruction."""
        closed_trades_this_cycle = []

        contract_size = self.risk_manager.get_contract_size(sym)
        pip_value = self.risk_manager.get_pip_value(sym)
        
        # Get commission and slippage from config for backtesting PnL calculation
        commission_per_trade = self.cfg.trading_costs.defaults.commission_per_trade
        slippage_pips = self.cfg.trading_costs.defaults.slippage_pips
        adaptive_slippage_multiplier = self.cfg.trading_costs.defaults.adaptive_slippage_multiplier

        # This loop identifies trades that close on the current bar
        for pos in [p for p in self.positions if p.symbol==sym and p.status=="open"]:
            price = row["close"]
            exit_reason = None
            
            if pos.direction == "long":
                if price <= pos.sl:
                    exit_reason = "Stop Loss"
                elif price >= pos.tp:
                    exit_reason = "Take Profit"
            elif pos.direction == "short":
                if price >= pos.sl:
                    exit_reason = "Stop Loss"
                elif price <= pos.tp:
                    exit_reason = "Take Profit"

            if exit_reason:
                gross_pnl = ((price - pos.entry_price) * pos.lots * contract_size) if pos.direction == "long" else ((pos.entry_price - price) * pos.lots * contract_size)
                
                # Calculate slippage cost for backtesting
                backtest_slippage_cost_value = 0.0
                if self.cfg.trading_costs.defaults.adaptive_slippage:
                    # In backtesting, we use slippage_pips as a base for adaptive calculation
                    backtest_slippage_cost_value = (slippage_pips * pip_value * pos.lots) * adaptive_slippage_multiplier
                else:
                    backtest_slippage_cost_value = slippage_pips * pip_value * pos.lots

                # Calculate total transaction cost for backtesting
                transaction_cost = (commission_per_trade * pos.lots) + backtest_slippage_cost_value
                net_pnl = gross_pnl - transaction_cost

                # Update consecutive losses counter
                if net_pnl < 0:
                    self.consecutive_losses += 1
                else:
                    self.consecutive_losses = 0
                
                # Update equity
                self.equity += net_pnl

                # Create ClosedTrade object from SimPosition
                closed_trade = ClosedTrade(
                    ticket=pos.ticket, # SimPosition already has a unique ticket-like ID
                    symbol=pos.symbol,
                    direction=pos.direction,
                    lots=pos.lots,
                    entry_price=pos.entry_price,
                    exit_price=price,
                    entry_time=pos.entry_time,
                    exit_time=row.name, # Use bar timestamp as exit_time
                    pnl=net_pnl,
                    risk_fraction=pos.risk_fraction,
                    atr=pos.atr, # ATR at entry
                    atr_idx=pos.atr_idx,
                    min_prob_long_idx=pos.min_prob_long_idx,
                    min_prob_short_idx=pos.min_prob_short_idx,
                    entry_auc=pos.entry_auc,
                    entry_equity=pos.entry_equity,
                    exit_equity=self.equity, # Current equity after this trade
                    adx=getattr(pos, 'adx', 0.0),
                    macd_diff=getattr(pos, 'macd_diff', 0.0),
                    volatility_10=getattr(pos, 'volatility_10', 0.0),
                    dist_from_ema_200=getattr(pos, 'dist_from_ema_200', 0.0),
                    context_vector=getattr(pos, 'trade_context', None) # NEW: Pass stored context vector
                )

                # Update the bandit with the ClosedTrade object
                self.risk_controller.update(closed_trade)

                # Close the position object (SimPosition)
                pos.close(price, row.name.to_pydatetime().replace(tzinfo=datetime.timezone.utc), net_pnl, self.equity)

                logger.info(
                    f"[{sym}] Closed {pos.direction} position at {pos.exit_price:.5f}. "
                    f"Entry: {pos.entry_price:.5f}, PnL: {pos.pnl:.2f}, Final Equity: {self.equity:.2f}"
                )
    
    def _perform_retraining(self, sym: str, bar_time: pd.Timestamp, i: int, data: pd.DataFrame, X: pd.DataFrame, y_long: pd.Series, y_short: pd.Series):
        """
        Handles the logic for retraining the model.
        """
        if not self.cfg.backtesting.enable_retraining:
            logger.debug(f"[{sym}] Retraining disabled by configuration. Skipping.")
            return self.ens_per_symbol_long[sym], self.ens_per_symbol_short[sym]

        if self.bar_counters[sym] > 0 and self.bar_counters[sym] % self.cfg.retrain_every_bars == 0:
            window_size = min(self.cfg.history_bars, i + 1)

            # --- FIX: Guard against retraining with insufficient data ---
            # A safe threshold to ensure enough samples for cross-validation.
            # This value should be comfortably larger than the sum of CV splits and min samples per model.
            MIN_BARS_FOR_RETRAIN = 100
            if window_size < MIN_BARS_FOR_RETRAIN:
                logger.warning(f"[{sym}] Skipping retraining at {bar_time}: not enough data in window ({window_size} < {MIN_BARS_FOR_RETRAIN} bars).")
                return self.ens_per_symbol_long[sym], self.ens_per_symbol_short[sym]

            train_data = data.iloc[i - window_size + 1: i + 1]
            logger.info(
                f"[{sym}] Ensemble retraining at {bar_time} using last {len(train_data)} bars..."
            )

            ens_old_long = self.ens_per_symbol_long[sym]
            ens_old_short = self.ens_per_symbol_short[sym]
            
            # Use the shared safe_retrain_ensemble function
            # IMPORTANT: A dry_run=True flag should be added here to prevent overwriting prod models.
            ens_new_long = safe_retrain_ensemble(self.cfg, sym, ens_old_long, train_data[X.columns], y_long.loc[train_data.index], train_data["close"] if "close" in train_data.columns else None, dry_run=True, model_type="long")
            ens_new_short = safe_retrain_ensemble(self.cfg, sym, ens_old_short, train_data[X.columns], y_short.loc[train_data.index], train_data["close"] if "close" in train_data.columns else None, dry_run=True, model_type="short")
            
            # Update the ensemble in the backtester's state
            self.ens_per_symbol_long[sym] = ens_new_long
            self.ens_per_symbol_short[sym] = ens_new_short
            
        return self.ens_per_symbol_long[sym], self.ens_per_symbol_short[sym]

    def _check_and_prune(self, trial: optuna.Trial, i: int):
        """Checks if the trial should be pruned."""
        current_returns = pd.Series([eq for _, eq in self.equity_curve]).pct_change().dropna()
        if not current_returns.empty:
            intermediate_sharpe = 0.0
            if current_returns.std() != 0:
                timeframe_minutes = self.cfg.timeframe_minutes()
                if timeframe_minutes is not None:
                    annualization_factor = np.sqrt(252 * (24 * 60 / timeframe_minutes))
                    intermediate_sharpe = current_returns.mean() / current_returns.std() * annualization_factor
            trial.report(intermediate_sharpe, i)
            if trial.should_prune():
                raise optuna.TrialPruned()

    def _process_bar(self, sym: str, data: pd.DataFrame, X: pd.DataFrame, y_long: pd.Series, y_short: pd.Series, trial: optuna.Trial | None = None, pruning_interval: int = 0):
        """Processes each bar of data for a given symbol."""
        ens_long = self.ens_per_symbol_long[sym]
        ens_short = self.ens_per_symbol_short[sym]
        risk_mgr = self.risk_manager

        logger.info(f"Processing {len(data)} bars for {sym}...")
        # Iterate over the aligned features (X), which do not include the forming bar
        for i in range(20, len(X)):
            bar_time = X.index[i] # Use X's index for bar_time
            current_row = data.loc[[bar_time]].iloc[0] # Get the corresponding row from the original data using X's index
            self.bar_counters[sym] += 1
            last_features = X.iloc[[i]] # X is already aligned, so X.iloc[[i]] is correct
            atr = X["atr_14"].iloc[i]

            # Manage existing positions first
            self._manage_trailing_stops(sym, current_row, atr)
            self._update_positions(sym, current_row)

            # --- Drawdown and Cooldown Check ---
            risk_mgr._update_equity_peak(self.equity)
            if risk_mgr._drawdown_exceeded(self.equity):
                if risk_mgr.cooldown_until is None:  # Only trigger if not already in cooldown
                    logger.warning(f"[{sym}][{bar_time}] Drawdown threshold exceeded. Triggering cooldown.")
                    now_utc = bar_time.to_pydatetime().replace(tzinfo=datetime.timezone.utc)
                    risk_mgr._trigger_cooldown(now=now_utc)

            # --- Consecutive Loss Check ---
            if risk_mgr.watchdog_cfg.enabled:
                max_losses = getattr(risk_mgr.watchdog_cfg, "max_consecutive_losses", None)
                if max_losses is not None and max_losses > 0:
                    if self.consecutive_losses >= max_losses:
                        if risk_mgr.cooldown_until is None:
                            logger.warning(f"[{sym}][{bar_time}] Watchdog: consecutive losses {consecutive_losses} >= threshold {max_losses}. Triggering cooldown.")
                            now_utc = bar_time.to_pydatetime().replace(tzinfo=datetime.timezone.utc)
                            risk_mgr._trigger_cooldown(now=now_utc)

            now_utc = bar_time.to_pydatetime().replace(tzinfo=datetime.timezone.utc)
            if risk_mgr.cooldown_active(now=now_utc):
                logger.info(f"[{sym}][{bar_time}] Trading blocked: watchdog cooldown active.")
                self.equity_curve.append((bar_time, self.equity))
                # --- Pruning Check (if in tuning mode) ---
                if trial and pruning_interval > 0 and (i % pruning_interval == 0) and self.cfg.symbols.index(sym) == 0:
                    current_returns = pd.Series([eq for _, eq in self.equity_curve]).pct_change().dropna()
                    if not current_returns.empty:
                        intermediate_sharpe = 0.0
                        if current_returns.std() != 0:
                            timeframe_minutes = self.cfg.timeframe_minutes()
                            if timeframe_minutes is not None:
                                annualization_factor = np.sqrt(252 * (24 * 60 / timeframe_minutes))
                                intermediate_sharpe = current_returns.mean() / current_returns.std() * annualization_factor
                        trial.report(intermediate_sharpe, i)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                continue

            # Retrain if needed
            ens_long, ens_short = self._perform_retraining(sym, bar_time, i, data, X, y_long, y_short)

            # Decide on new trades
            prob_long = ens_long.predict_proba(last_features).iloc[0]
            prob_short = ens_short.predict_proba(last_features).iloc[0]

            # Get dynamic risk parameters from RiskController
            context = {
                "vol": atr,
                "equity": self.equity,
                "peak_equity": self.risk_manager.equity_peak,
                "ensemble_auc": (ens_long.ensemble_cv_auc_ + ens_short.ensemble_cv_auc_) / 2, # Pass current model confidence
                "adx": float(last_features["adx"].iloc[0]) if "adx" in last_features.columns else 0.0,
                "macd_diff": float(last_features["macd_diff"].iloc[0]) if "macd_diff" in last_features.columns else 0.0,
                "volatility_10": float(last_features["volatility_10"].iloc[0]) if "volatility_10" in last_features.columns else 0.0,
                "dist_from_ema_200": float(last_features["dist_from_ema_200"].iloc[0]) if "dist_from_ema_200" in last_features.columns else 0.0,
            }
            dynamic_risk_params = self.risk_controller.get_params(sym, context)

            atr_multiplier_sl = dynamic_risk_params["atr_multiplier_sl"]
            atr_multiplier_tp = dynamic_risk_params["atr_multiplier_tp"]
            trailing_atr_mult = dynamic_risk_params["trailing_atr_mult"]
            min_prob_long = dynamic_risk_params["min_prob_long"]
            min_prob_short = dynamic_risk_params["min_prob_short"]
            min_ensemble_auc = risk_mgr.cfg.get_symbol_value(sym, 'min_ensemble_auc', 0.55)
            atr_idx = dynamic_risk_params["atr_idx"]
            min_prob_long_idx = dynamic_risk_params.get("min_prob_long_idx", -1)
            min_prob_short_idx = dynamic_risk_params.get("min_prob_short_idx", -1)

            direction = None
            auc_score = 0.5
            if prob_long >= min_prob_long and ens_long.ensemble_cv_auc_ >= min_ensemble_auc:
                direction = "long"
                auc_score = ens_long.ensemble_cv_auc_
            elif prob_short >= min_prob_short and ens_short.ensemble_cv_auc_ >= min_ensemble_auc:
                direction = "short"
                auc_score = ens_short.ensemble_cv_auc_
            else:
                if prob_long >= min_prob_long and ens_long.ensemble_cv_auc_ < min_ensemble_auc:
                    logger.info(f"[{sym}] Long trade blocked due to low ensemble confidence (AUC={ens_long.ensemble_cv_auc_:.4f} < {min_ensemble_auc:.4f}).")
                if prob_short >= min_prob_short and ens_short.ensemble_cv_auc_ < min_ensemble_auc:
                    logger.info(f"[{sym}] Short trade blocked due to low ensemble confidence (AUC={ens_short.ensemble_cv_auc_:.4f} < {min_ensemble_auc:.4f}).")

            if direction:
                total_open_risk = sum(p.entry_equity * p.risk_fraction for p in self.positions if p.status == "open")

                # Determine pip_value for position sizing
                pip_value = self.risk_manager.get_pip_value(sym)
                pip_size = self.risk_manager.get_pip_size(sym)

                lots, effective_risk = self.risk_manager.position_size(
                    self.equity, atr, auc_score, total_open_risk=total_open_risk, symbol=sym,
                    exploration_mult=dynamic_risk_params.get("exploration_risk_mult", 1.0),
                    ac_multiplier=dynamic_risk_params.get("ac_multiplier", 1.0)
                )

                if lots > 0:
                    price = current_row["close"]
                    # Use dynamic SL/TP multipliers
                    sl, tp = self.risk_manager.stop_targets(price, atr, direction, auc_score, sym, sl_mult=atr_multiplier_sl, tp_mult=atr_multiplier_tp)
                    pos = SimPosition(
                        sym, direction, lots, price, sl, tp, bar_time, atr, auc_score, effective_risk,
                        entry_equity=self.equity,
                        atr_idx=atr_idx,
                        min_prob_long_idx=min_prob_long_idx,
                        min_prob_short_idx=min_prob_short_idx, # Store discrete choices
                        trade_context=dynamic_risk_params.get("context_vector") # NEW: Store the context vector
                    )
                    self.positions.append(pos)
                    logger.info(
                        f"[{sym}][{bar_time}] Opened {direction} position at {price:.5f}. "f"Lots: {lots:.2f}, SL: {sl:.5f}, TP: {tp:.5f}, AUC: {auc_score:.4f}"
                    )
                else:
                    logger.info(f"[{sym}] Trade skipped due to risk limits or position size zero.")
            else:
                logger.info(f"[{sym}] No trade signal. Probs: (Long: {prob_long:.3f}, Short: {prob_short:.3f}) ")

            self.equity_curve.append((bar_time, self.equity))
            
            # Periodic persistence of Thompson state + CSV (avoid overwriting prod state)
            total_bars = sum(self.bar_counters.values())
            if total_bars % self.save_state_every_bars == 0:
                # write to the backtest-specific file so you don't overwrite live/prod state
                self._persist_bandit_state(force_path=self.backtest_ts_state_file)


            # --- Pruning Check (if in tuning mode) ---
            if trial and pruning_interval > 0 and (i % pruning_interval == 0) and self.cfg.symbols.index(sym) == 0:
                self._check_and_prune(trial, i)

    def _generate_results(self):
        """Generates and saves the backtesting results."""
        eq_df = pd.DataFrame(self.equity_curve, columns=["time", "equity"]).set_index("time")
        trades_df = pd.DataFrame([p.__dict__ for p in self.positions])

        os.makedirs("results", exist_ok=True)
        symbol_str = "_".join([s.replace('#', '') for s in self.cfg.symbols])

        eq_df.to_csv(f"results/equity_curve_{symbol_str}_hybrid_adaptive.csv")
        trades_df.to_csv(f"results/trades_{symbol_str}_hybrid_adaptive.csv")

        # Generate quantstats report
        try:
            trades = [p for p in self.positions]
            long_trades = [t for t in trades if t.direction == "long"]
            short_trades = [t for t in trades if t.direction == "short"]
            logger.info(f"Number of long trades: {len(long_trades)}")
            logger.info(f"Number of short trades: {len(short_trades)}")

            returns = pd.Series([t.pnl for t in trades if t.pnl is not None], index=[t.exit_time for t in trades if t.pnl is not None])
            returns.index = pd.to_datetime(returns.index)
            
            if not eq_df.empty:
                try:
                    report_path = f"results/report_{symbol_str}_hybrid_adaptive.html"
                    qs.reports.html(eq_df["equity"], output=report_path, title=f"{symbol_str} Hybrid Adaptive Strategy")
                    logger.info(f"QuantStats report saved to {report_path}")
                except Exception as e:
                    logger.warning(f"Failed to generate QuantStats report: {e}")
            else:
                logger.warning("Skipping QuantStats report generation: No returns data available.")
        except Exception as e:
            logger.exception(f"Failed to generate QuantStats report: {e}")

        logger.info(f"=== Hybrid Adaptive Backtest Complete. Final Equity: {self.equity:.2f} === ")
        logger.info("Results saved to 'results/' directory.")

        # NEW: Generate Thompson Sampling parameter evolution report
        if self.ts_param_history:
            ts_df = pd.DataFrame(self.ts_param_history)
            ts_report_path = f"results/ts_param_evolution_{symbol_str}_hybrid_adaptive.csv"
            ts_df.to_csv(ts_report_path, index=False)
            logger.info(f"Thompson Sampling parameter evolution report saved to {ts_report_path}")

        return trades_df, eq_df

    def _persist_bandit_state(self, force_path: str | None = None):
        """
        Persist RiskController state and ts_param_history CSV.
        If force_path supplied, temporarily write the TS JSON to that path.
        """
        try:
            # optionally override cfg path just for this save
            original_state_file = None
            if force_path:
                original_state_file = self.cfg.thompson_sampling.state_file
                self.cfg.thompson_sampling.state_file = force_path

            # Save RiskController internal JSON (uses RiskController.save_state)
            try:
                self.risk_controller.save_state()
            except Exception:
                logger.exception("Failed to save RiskController state with risk_controller.save_state()")

            # Save human-readable CSV of ts history for offline analysis
            if self.ts_param_history:
                import pandas as pd # type: ignore
                df = pd.DataFrame(self.ts_param_history)
                df.to_csv(self.ts_history_csv, index=False)

            logger.info(f"Persisted bandit state to {self.cfg.thompson_sampling.state_file} and CSV to {self.ts_history_csv}")
        except Exception as e:
            logger.exception(f"_persist_bandit_state failed: {e}")
        finally:
            if force_path and original_state_file is not None:
                # restore original
                self.cfg.thompson_sampling.state_file = original_state_file


    def run(self, trial: optuna.Trial | None = None, pruning_interval: int = 0):
        logger.info("=== Starting Hybrid Adaptive Backtest ===")
        try:
            for sym in self.cfg.symbols:
                logger.info(f"--- Backtesting Symbol: {sym} ---")
                
                # Load best feature params from optuna study
                optuna_params = load_optuna_params(sym, self.cfg)
                feature_params = optuna_params.get('features', {}) if optuna_params else {}
                feature_cfg = FeatureCfg(**feature_params)

                # Load context data
                from src.data_manager import DataManager
                dm = DataManager(self.cfg)
                mta_df = None
                if self.cfg.context_features.mta.enabled:
                    mta_df = dm.load_local_history(sym, self.cfg.context_features.mta.timeframe)
                inter_market_df = None
                if self.cfg.context_features.inter_market.enabled:
                    im_sym = self.cfg.context_features.inter_market.symbol
                    inter_market_df = dm.load_local_history(im_sym, self.cfg.timeframe)

                data, X, y_long, y_short = get_training_data(self.cfg, sym, feature_cfg=feature_cfg, source=self.cfg.data_source, min_pct_change=feature_cfg.min_pct_change, mta_df=mta_df, inter_market_df=inter_market_df, return_long_short_labels=True)
                if data.empty:
                    logger.warning(f"No data for {sym}, skipping.")
                    continue

                self._process_bar(sym, data, X, y_long, y_short, trial, pruning_interval)

                logger.info(f"--- Completed Backtest for Symbol: {sym} ---")

                # --- Close any positions left open for the current symbol ---
                logger.info(f"Closing any remaining open positions for {sym}...")
                contract_size = self.risk_manager.get_contract_size(sym)
                pip_value = self.risk_manager.get_pip_value(sym)
                cost_pips_per_lot = getattr(self.cfg.risk, 'transaction_cost_pips', 0.0)
                cost_per_lot = cost_pips_per_lot * pip_value
                for pos in [p for p in self.positions if p.symbol == sym and p.status == "open"]:
                    last_row = data.iloc[-1]
                    last_price = last_row["close"]
                    gross_pnl = ((last_price - pos.entry_price) * pos.lots * contract_size) if pos.direction == "long" else ((pos.entry_price - last_price) * pos.lots * contract_size)
                    transaction_cost = cost_per_lot * pos.lots
                    net_pnl = gross_pnl - transaction_cost

                    pos.close(last_price, last_row.name, net_pnl, self.equity + net_pnl)
                    self.equity += net_pnl
                    logger.info(
                        f"[{pos.symbol}] Force-closed open {pos.direction} position at final price {last_price:.5f}. "
                        f"PnL: {net_pnl:.2f}, Final Equity: {self.equity:.2f}"
                    )
        except KeyboardInterrupt:
            logger.warning("Backtest interrupted by user. Generating results for completed portion...")
        
        return self._generate_results()

if __name__ == "__main__":
    import numpy as np  # type: ignore
    import random
    import sys

    np.random.seed(42)
    random.seed(42)
    cfg = Cfg.from_yaml("config.yaml")
    setup_logging(level=cfg.logging["level"], to_file=cfg.logging["to_file"], rotate=cfg.logging["rotate"], retention=cfg.logging["retention"])
    
    mt5_client = None
    if cfg.data_source == "mt5":
        from src.mt5_client import MT5Client
        mt5_client = MT5Client(
            login=os.getenv("MT5_LOGIN"),
            password=os.getenv("MT5_PASSWORD"),
            server=os.getenv("MT5_SERVER"),
            path=os.getenv("MT5_PATH"),
        )
        if not mt5_client.connect():
            logger.error("Failed to connect to MT5, falling back to csv data source.")
            cfg.data_source = "csv"
            mt5_client = None

    cfg.thompson_sampling.state_file = f"ts_risk_controller_state_backtest_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"

    
    bt = HybridBacktester(cfg, mt5_client)
    try:
        trades_df, eq_df = bt.run()
        print("\n--- Trades Summary ---")
        print(trades_df.tail())
        print("\n--- Equity Curve ---")
        print(eq_df.tail())
    finally:
        try:
            bt._persist_bandit_state(force_path=bt.backtest_ts_state_file)
        except Exception:
            logger.exception("Final persist of bandit state failed.")
        if mt5_client:
            mt5_client.shutdown()
