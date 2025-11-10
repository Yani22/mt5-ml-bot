# main.py
import argparse
import os
import time
import copy
from multiprocessing import Process
from dotenv import load_dotenv
from loguru import logger
import pandas as pd
from src.config import Cfg, FeatureCfg
from src.mt5_client import MT5Client
from src.risk import RiskManager
from src.execution import Execution
from src.utils import setup_logging, get_training_data, load_ensemble, save_ensemble, safe_retrain_ensemble, load_optuna_params, log_symbol_specific_configs, log_startup_summary, timeframe_to_seconds, ensure_min_grid_size, timeframe_to_mt5_timeframe
from src.live_performance_monitor import LivePerformanceMonitor
from src.notifier import TelegramNotifier

from src.risk_controller import RiskController
import datetime
import json
from typing import Dict, Any
from src.data_manager import DataManager
from src.labels import generate_long_short_labels
from src.bandit_warmstart import find_latest_backtest_state, merge_warmstart
import csv
import yaml
import numpy as np
from typing import List

import threading
from src.symbol_processor import SymbolProcessor


# --- Initial Setup ---
load_dotenv()
setup_logging()

# --- Metrics Logging Setup ---
METRICS_CSV_FILE = "results/risk_metrics.csv"
METRICS_HEADERS = [
    "timestamp", "symbol", "event_type", "atr_idx", "min_prob_long_idx", "min_prob_short_idx",
    "atr_mult_sl", "atr_mult_tp", "min_prob_long", "min_prob_short",
    "rule_scale", "reward", "equity", "peak_equity", "drawdown", "ensemble_auc"
]

def _initialize_metrics_csv():
    if not os.path.exists(METRICS_CSV_FILE):
        with open(METRICS_CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(METRICS_HEADERS)
        logger.info(f"Initialized metrics CSV file: {METRICS_CSV_FILE}")

# Call initialization at startup
_initialize_metrics_csv()

def log_metrics_to_csv(data: Dict[str, Any]):
    with open(METRICS_CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        row = [data.get(header, "") for header in METRICS_HEADERS]
        writer.writerow(row)

def print_dashboard(cfg, risk, ens, X, sym, bar_counter, is_first_symbol, equity, balance):
    """ Prints a live portfolio dashboard for a single symbol, throttled. """
    if is_first_symbol:
        drawdown = 1 - (equity / balance) if balance else 0.0

        total_open_risk = sum([pos.get('risk', 0.0) for pos in risk.open_positions_cache.values()])

        logger.info("=== PORTFOLIO DASHBOARD ===")
        logger.info(f"Equity: {equity:.2f} | Balance: {balance:.2f} | Drawdown: {drawdown:.3%} | Total Open Risk: {total_open_risk:.3%}")

    if X is not None and not X.empty and ens is not None:
        try:
            atr = float(X["atr_14"].iloc[-1])
        except Exception:
            atr = 0.0
        try:
            last_features = X.iloc[[-1]]
            prob_up = ens.predict_proba(last_features)
            p_up = float(prob_up.iloc[0])
        except Exception:
            p_up = 0.5
        # Correctly find and format positions for the current symbol
        positions_for_symbol = [
            pos for pos in risk.open_positions_cache.values()
            if pos.get('symbol') == sym
        ]
        open_pos_str = ", ".join([
            f"Ticket({p.get('ticket')}, {p.get('direction')}, {p.get('lots')} lots)"
            for p in positions_for_symbol
        ]) if positions_for_symbol else "None"

def run_retraining_in_background(cfg, sym, feature_cfg, dry_run, notifier, optuna_params_per_symbol):
    """
    A wrapper function to run the entire retraining pipeline for both long and short models in a separate process.
    This function now saves optimized parameters to the symbol_overrides section of config.yaml.
    """
    try:
        data_manager = DataManager(cfg)
        full_data, full_X, _ = data_manager.load_cached(sym, feature_cfg, count=cfg.retraining_window_bars, min_pct_change=feature_cfg.min_pct_change)

        if full_X is None or full_X.empty:
            message = f"[{sym}] <b>WARNING:</b> No data for retraining, background process exiting."
            logger.warning(message)
            if notifier: notifier.send_message(message, level="WARNING")
            return

        # Retrieve tuned prediction_horizon for this symbol
        tuned_prediction_horizon = optuna_params_per_symbol[sym].get('prediction_horizon', cfg.prediction_horizon)

        y_long, y_short = generate_long_short_labels(full_data, tuned_prediction_horizon, feature_cfg.min_pct_change)

        logger.info(f"[{sym}] Retraining LONG model...")
        ens_old_long = load_ensemble(cfg, sym, "long", model_params=optuna_params_per_symbol[sym])
        safe_retrain_ensemble(cfg, sym, ens_old_long, full_X, y_long, full_data["close"], dry_run=dry_run, model_type="long", model_params=optuna_params_per_symbol[sym])

        logger.info(f"[{sym}] Retraining SHORT model...")
        ens_old_short = load_ensemble(cfg, sym, "short", model_params=optuna_params_per_symbol[sym])
        safe_retrain_ensemble(cfg, sym, ens_old_short, full_X, y_short, full_data["close"], dry_run=dry_run, model_type="short", model_params=optuna_params_per_symbol[sym])

        logger.info(f"[{sym}] Background retraining process for LONG and SHORT models finished.")

    except Exception as e:
        logger.exception(f"[{sym}] Background retraining process failed: {e}")

def _handle_model_acceptance(sym, cfg, ens_per_symbol_long, ens_per_symbol_short, active_model_auc, live_monitor, notifier, optuna_params_per_symbol):
    """Loads newly trained models, compares them, and accepts them if they are an improvement."""
    logger.info(f"[{sym}] Handling model acceptance...")
    try:
        new_ens_long = load_ensemble(cfg, sym, "long")
        new_ens_short = load_ensemble(cfg, sym, "short")

        old_ens_long = ens_per_symbol_long[sym]
        old_ens_short = ens_per_symbol_short[sym]

        new_auc_long = getattr(new_ens_long, "ensemble_cv_auc_", 0.5)
        new_auc_short = getattr(new_ens_short, "ensemble_cv_auc_", 0.5)
        old_auc_long = getattr(old_ens_long, "ensemble_cv_auc_", 0.5)
        old_auc_short = getattr(old_ens_short, "ensemble_cv_auc_", 0.5)

        # Use the new helper to get the symbol-specific value, falling back to the global default
        min_auc_improvement = cfg.get_symbol_value(sym, 'min_auc_improvement', 0.005)

        long_accepted = new_auc_long >= old_auc_long + min_auc_improvement
        short_accepted = new_auc_short >= old_auc_short + min_auc_improvement

        if long_accepted:
            ens_per_symbol_long[sym] = new_ens_long
            active_model_auc[sym] = new_auc_long
            live_monitor.update_ensemble_auc(new_auc_long)
            message = f"[{sym}] New LONG model accepted (AUC: {old_auc_long:.4f} -> {new_auc_long:.4f})."
            logger.info(message)
            if notifier: notifier.send_message(message, level="INFO")
        else:
            message = f"[{sym}] New LONG model rejected (AUC: {old_auc_long:.4f} -> {new_auc_long:.4f}). Keeping old model."
            logger.warning(message)
            if notifier: notifier.send_message(message, level="WARNING")

        if short_accepted:
            ens_per_symbol_short[sym] = new_ens_short
            message = f"[{sym}] New SHORT model accepted (AUC: {old_auc_short:.4f} -> {new_auc_short:.4f})."
            logger.info(message)
            if notifier: notifier.send_message(message, level="INFO")
        else:
            message = f"[{sym}] New SHORT model rejected (AUC: {old_auc_short:.4f} -> {new_auc_short:.4f}). Keeping old model."
            logger.warning(message)
            if notifier: notifier.send_message(message, level="WARNING")

    except Exception as e:
        logger.exception(f"[{sym}] Error during model acceptance: {e}")

def _check_and_trigger_retraining(cfg: Cfg, sym: str, feature_cfg_per_symbol: Dict[str, FeatureCfg], dry_run: bool, notifier: TelegramNotifier, optuna_params_per_symbol: Dict[str, Any], retraining_processes: Dict[str, Process], retraining_status: Dict[str, bool], last_retrain_date: Dict[str, datetime.date], risk_controller: RiskController):
    """
    Checks if retraining should be triggered for a given symbol based on retrain_time_utc.
    """
    current_utc_datetime = datetime.datetime.now(datetime.timezone.utc)
    current_utc_time = current_utc_datetime.time()
    current_utc_date = current_utc_datetime.date()

    retrain_time_value = cfg.get_symbol_value(sym, 'retrain_time_utc', None)
    retrain_times = [retrain_time_value] if isinstance(retrain_time_value, str) else retrain_time_value
    if not retrain_times:
        # If retrain_time_utc is not configured, fall back to retrain_every_bars logic if needed
        # For now, we'll just return if no specific time is set.
        return

    for retrain_time_str in retrain_times:
        try:
            retrain_hour, retrain_minute = map(int, retrain_time_str.split(':'))
            retrain_datetime = datetime.datetime.combine(current_utc_date, datetime.time(retrain_hour, retrain_minute), tzinfo=datetime.timezone.utc)

            # Check if current time is past the retrain time and it hasn't been retrained today
            if current_utc_datetime >= retrain_datetime and last_retrain_date[sym] != current_utc_date:
                if not retraining_status[sym]:
                    if cfg.fetch.retrain_in_background:
                        logger.info(f"[{sym}] Triggering background retraining at {current_utc_datetime.time().strftime('%H:%M')} UTC...")
                        notifier.send_message(f"[{sym}] Triggering background retraining.", level="INFO")
                    else:
                        logger.info(f"[{sym}] Triggering foreground retraining at {current_utc_datetime.time().strftime('%H:%M')} UTC...")
                        notifier.send_message(f"[{sym}] Triggering foreground retraining.", level="INFO")

                    if cfg.fetch.retrain_in_background:
                        logger.info(f"[{sym}] Starting background retraining process...")
                        process = Process(
                            target=run_retraining_in_background,
                            args=(cfg, sym, feature_cfg_per_symbol[sym], dry_run, notifier, optuna_params_per_symbol)
                        )
                        process.start()
                        retraining_processes[sym] = process
                        retraining_status[sym] = True
                    else:
                        logger.info(f"[{sym}] Running foreground retraining (blocking)...")
                        run_retraining_in_background(cfg, sym, feature_cfg_per_symbol[sym], dry_run, notifier, optuna_params_per_symbol)
                        logger.info(f"[{sym}] Foreground retraining finished.")
                        # For foreground retraining, it's immediately done, so no process to track
                        retraining_status[sym] = False # Mark as done
                        # No need to add to retraining_processes as it's not a background process
                    last_retrain_date[sym] = current_utc_date  # Mark as retrained for today
                    risk_controller.update_last_daily_retrain_date(sym, current_utc_date) # Update RiskController's internal state
                else:
                    logger.info(f"[{sym}] Retraining already in progress for today. Skipping.")
                return # Only trigger once per day per symbol
        except ValueError:
            logger.error(f"[{sym}] Invalid retrain_time_utc format: {retrain_time_str}. Expected HH:MM.")
        except Exception as e:
            logger.exception(f"[{sym}] Error checking or triggering retraining: {e}")

def run(dry_run: bool = False):
    """ Production-ready main loop for hybrid adaptive MT5 ML bot. """
    RECONNECTION_RETRY_SECONDS = 60
    cfg = Cfg.from_yaml("config.yaml")
    setup_logging(level=cfg.logging['level'])

    cfg.dashboard_every_bars = getattr(cfg, "dashboard_every_bars", 10)

    if cfg.startup_logging:
        log_startup_summary(cfg)

    logger.info("=== Starting MT5 ML Bot (Hybrid Adaptive) ===")
    logger.info(f"Dry-run mode: {dry_run}")
    logger.info(f"Symbols: {cfg.symbols if hasattr(cfg,'symbols') else []}")

    # Initialize notifier
    notifier = TelegramNotifier(cfg)

    # Initialize DataManager
    data_manager = DataManager(cfg)

    # --- Initial Feature Config Loading ---
    optuna_params_per_symbol = {}
    feature_cfg_per_symbol = {}
    for sym in cfg.symbols:
        optuna_params = load_optuna_params(sym, cfg)
        optuna_params_per_symbol[sym] = optuna_params
        feature_params = optuna_params.get('features', {}) if optuna_params else {}
        feature_cfg_per_symbol[sym] = FeatureCfg(**feature_params)

    retraining_processes = {}
    retraining_status = {sym: False for sym in cfg.symbols}  # Track if retraining is active
    trading_blocked_by_low_new_model_auc = {sym: False for sym in cfg.symbols}  # Track if trading is blocked due to low new model AUC
    last_diagnostics_log_time = 0.0  # For throttling diagnostics logging
    last_retrain_date = {sym: None for sym in cfg.symbols}  # Track last retraining date per symbol

    live_monitor = None
    risk_controller = None
    risk_manager = None # Declare risk_manager here
    exe = None # Declare exe here
    mt5c = None
    try:
        # Outer loop for MT5 reconnection attempts
        while True:
            try:
                # --- MT5 Connection ---
                mt5c = MT5Client(
                    os.getenv("MT5_LOGIN"),
                    os.getenv("MT5_PASSWORD"),
                    os.getenv("MT5_SERVER"),
                    os.getenv("MT5_PATH"),
                )
                if not mt5c.connect():
                    logger.error("MT5 initial connection failed. Retrying in 60 seconds...")
                    notifier.send_message("<b>CRITICAL:</b> MT5 initial connection failed. Retrying...", level="CRITICAL")
                    time.sleep(RECONNECTION_RETRY_SECONDS)
                    continue  # Try connecting again

                logger.debug("MT5 connection established.")
                notifier.send_message("MT5 connection established.", level="INFO")

                # Get initial equity from MT5 account info
                account_info = mt5c.account_info()
                initial_equity = getattr(account_info, "equity", 100.0) if account_info else 100.0
                cfg.initial_equity = initial_equity  # Set initial equity in Cfg for the monitor

                live_monitor = LivePerformanceMonitor(cfg)
                live_monitor.load_state()  # Load previous state on startup

                # --- Load Ensembles and Feature Configs ---
                ens_per_symbol_long = {}
                ens_per_symbol_short = {}
                active_model_auc = {}  # To store AUC of currently active model

                # Single pass: load ensembles and feature configs once, and bootstrap history via DataManager
                for sym in cfg.symbols:
                    ens_per_symbol_long[sym] = load_ensemble(cfg, sym, "long", model_params=optuna_params_per_symbol[sym])
                    ens_per_symbol_short[sym] = load_ensemble(cfg, sym, "short", model_params=optuna_params_per_symbol[sym])
                    active_model_auc[sym] = getattr(ens_per_symbol_long[sym], "ensemble_cv_auc_", getattr(ens_per_symbol_long[sym], "cv_auc_", 0.5))
                    logger.info(f"[{sym}] Active model AUC (Long): {active_model_auc[sym]:.4f}")

                    # Bootstrap historical data with caching (chunked fetch if needed)
                    logger.info(f"[{sym}] Bootstrapping local history...")
                    # support both nested fetch config and legacy cfg.initial_fetch_bars
                    initial_bars = getattr(cfg.fetch, "initial_fetch_bars", getattr(cfg, "history_bars", 30000))

                    data_manager.bootstrap_history(sym, initial_bars=initial_bars)

                bar_counters = {sym: 0 for sym in cfg.symbols}
                last_bar_time = {sym: None for sym in cfg.symbols}
                X_per_symbol = {}

                # Warm-start bandit: merge latest backtest priors into live state file BEFORE instantiating RiskController
                try:
                    for sym in cfg.symbols:

                        latest_backtest = find_latest_backtest_state(symbol=sym, results_dir="results")
                        if latest_backtest:
                            warm_weight = getattr(getattr(cfg, "thompson_sampling", {}), "warmstart_weight", 1.0)
                            logger.info(f"Found backtest bandit state for {sym}: {latest_backtest}; merging into live state (weight={warm_weight})")

                            # Ensure results directory exists for the state file
                            os.makedirs("results", exist_ok=True)
                            live_state_path = getattr(getattr(cfg, "thompson_sampling", {}), "state_file", "ts_risk_controller_state.json")
                            merge_warmstart(latest_backtest, live_state_path, warmstart_weight=warm_weight)
                        else:
                            logger.info(f"No backtest bandit state file found to warm-start for {sym}.")
                except Exception:
                    logger.exception(f"Warmstart merge failed; continuing without warmstart.")

                # Instantiate risk manager and risk controller AFTER warmstart merge so they load the merged state
                risk_manager = RiskManager(cfg, mt5c, notifier=notifier) # Pass notifier
                risk_controller = RiskController(cfg, notifier=notifier) # Instantiate RiskController
                loaded_open_positions = risk_controller.load_state() # Load state again to get open_positions_cache

                # Execution object (single instance)
                exe = Execution(ens_per_symbol_long, ens_per_symbol_short, risk_manager, mt5c, data_manager, dry_run=dry_run, notifier=notifier, monitor=live_monitor)
                exe.risk.open_positions_cache.update(loaded_open_positions)  # Initialize exe's cache with loaded data

                # Reconcile open positions with MT5 to ensure accuracy
                exe.reconcile_open_positions_with_mt5()

                retraining_status = {sym: False for sym in cfg.symbols}  # Track if retraining is active
                trading_blocked_by_low_new_model_auc = {sym: False for sym in cfg.symbols}  # Track if trading is blocked due to low new model AUC
                last_diagnostics_log_time = 0.0  # For throttling diagnostics logging
                # Initialize last_retrain_date from RiskController's loaded state
                last_retrain_date = risk_controller.last_daily_retrain_date

                # --- Start Symbol Processors ---
                symbol_threads = [] # Initialize here to prevent UnboundLocalError
                for sym in cfg.symbols:
                    # Each MT5Client instance needs to be independent for thread safety
                    # Initialize a new MT5Client for each SymbolProcessor
                    mt5_client_per_symbol = MT5Client(
                        login=os.getenv("MT5_LOGIN"),
                        password=os.getenv("MT5_PASSWORD"),
                        server=os.getenv("MT5_SERVER"),
                        path=os.getenv("MT5_PATH"),
                    )
                    if not mt5_client_per_symbol.connect():
                        logger.error(f"Failed to connect MT5 client for symbol {sym}. This symbol will not be processed.")
                        continue

                    processor = SymbolProcessor(cfg, sym, mt5_client_per_symbol, risk_controller, risk_manager, live_monitor, exe, dry_run)
                    thread = threading.Thread(target=processor.run_loop, daemon=True)
                    symbol_threads.append({"symbol": sym, "thread": thread, "processor": processor, "mt5_client": mt5_client_per_symbol})
                    thread.start()
                    logger.info(f"Started processing thread for symbol: {sym}")

                # Main thread now monitors symbol threads and performs global tasks
                while True:
                    # Periodically save global states and check thread health
                    live_monitor.save_state()
                    risk_controller.save_state(exe.risk.open_positions_cache)

                    for i, symbol_data in enumerate(symbol_threads):
                        if not symbol_data["thread"].is_alive():
                            logger.error(f"Thread for {symbol_data['symbol']} died unexpectedly. Attempting to restart...")
                            # For simplicity, we'll log and exit the main loop for now.
                            # A more robust solution would re-initialize the processor and restart the thread.
                            notifier.send_message(f"<b>CRITICAL:</b> Thread for {symbol_data['symbol']} died. Shutting down bot.", level="CRITICAL")
                            raise RuntimeError(f"Thread for {symbol_data['symbol']} died.")

                    # --- Retraining Logic ---
                    for sym in cfg.symbols:
                        # Check and trigger retraining if conditions are met
                        _check_and_trigger_retraining(
                            cfg, sym, feature_cfg_per_symbol, dry_run, notifier,
                            optuna_params_per_symbol, retraining_processes,
                            retraining_status, last_retrain_date, risk_controller
                        )

                        # Check if a retraining process has finished
                        if retraining_status[sym] and not retraining_processes[sym].is_alive():
                            logger.info(f"[{sym}] Background retraining process finished.")
                            _handle_model_acceptance(
                                sym, cfg, ens_per_symbol_long, ens_per_symbol_short,
                                active_model_auc, live_monitor, notifier, optuna_params_per_symbol
                            )
                            retraining_status[sym] = False
                            del retraining_processes[sym] # Clean up the process entry

                    # Sleep for a short interval before checking again
                    time.sleep(5) # Check every 5 seconds

            except Exception as e:
                logger.exception(f"MT5 connection lost or critical error in trading loop: {e}. Attempting to reconnect...")
                notifier.send_message(f"<b>CRITICAL:</b> MT5 connection lost or critical error: {e}. Attempting to reconnect...", level="CRITICAL")
                try:
                    # Shutdown all symbol-specific MT5 clients before attempting main reconnection
                    for symbol_data in symbol_threads:
                        symbol_data["mt5_client"].shutdown()
                except Exception:
                    logger.exception("Failed to shutdown symbol MT5 clients after error.")
                try:
                    mt5c.shutdown()  # Ensure old main connection is closed
                except Exception:
                    logger.exception("Failed to shutdown main mt5 client after error.")
                time.sleep(RECONNECTION_RETRY_SECONDS)  # Wait before retrying connection

    except KeyboardInterrupt:
        logger.info("=== Stopping MT5 ML Bot ===")
        notifier.send_message("MT5 ML Bot stopped by user (KeyboardInterrupt).", level="WARNING")
        # Threads are daemon, so they will exit when main thread exits.
        # No explicit join needed for graceful shutdown in this simple daemon setup.
    finally:
        logger.info("Shutting down MT5 clients and saving final states...")
        # Ensure symbol_threads is defined even if an error occurred before its initialization
        if 'symbol_threads' in locals():
            for symbol_data in symbol_threads:
                try:
                    symbol_data["mt5_client"].shutdown()
                except Exception:
                    logger.exception(f"Failed to shutdown MT5 client for {symbol_data['symbol']} cleanly.")

        if live_monitor:
            try:
                live_monitor.save_state()  # Save state on shutdown
            except Exception:
                logger.error("Failed to save live monitor state on shutdown.", exc_info=True)

        if risk_controller and exe: # Check if exe is also defined
            try:
                # Use the latest open positions cache from the live monitor, as it's the aggregate from all threads
                risk_controller.save_state(exe.risk.open_positions_cache) # Save final state
            except Exception:
                logger.exception("Failed to save RiskController state on shutdown.")
        logger.info("MT5 ML Bot shutdown complete.")
        notifier.send_message("MT5 ML Bot shutdown complete.", level="INFO")

if __name__ == "__main__":
    # Default to dry-run to be safe; change to False when you are ready.
    # run(dry_run=True)
    run(dry_run=False)