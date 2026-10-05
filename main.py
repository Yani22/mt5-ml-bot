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
from src.mt5_client import MT5Client, teardown_connection
from src.risk import RiskManager
from src.execution import Execution
from src.utils import setup_logging, get_training_data, load_ensemble, save_ensemble, safe_retrain_ensemble, load_optuna_params, log_symbol_specific_configs, log_startup_summary, timeframe_to_seconds, ensure_min_grid_size, timeframe_to_mt5_timeframe, _initialize_metrics_csv, log_metrics_to_csv, METRICS_CSV_FILE, METRICS_HEADERS
from src.live_performance_monitor import LivePerformanceMonitor, equity_from_account
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
from src.live_guard import require_live_permission
from src.state_paths import apply_mode_state_paths
from src.symbol_processor import SymbolProcessor, stop_symbol_threads
from src.retraining import _check_and_trigger_retraining, _handle_model_acceptance, retraining_status_for
from src.utils import _initialize_metrics_csv


# --- Initial Setup ---
load_dotenv()
setup_logging()
_initialize_metrics_csv()


def _warmstart_bandit(cfg):
    """Merges the latest backtest bandit priors into the live state file (before RiskController loads it)."""
    try:
        live_state_path = getattr(getattr(cfg, "thompson_sampling", {}), "state_file", "ts_risk_controller_state.json")
        os.makedirs("results", exist_ok=True)  # Ensure results directory exists for the state file

        # Find the latest backtest file once (now finds the latest overall backtest file)
        latest_backtest_file = find_latest_backtest_state(results_dir="results")

        if latest_backtest_file:
            warm_weight = getattr(getattr(cfg, "thompson_sampling", {}), "warmstart_weight", 1.0)
            logger.info(f"Found latest backtest bandit state: {latest_backtest_file}; merging into live state for all symbols (weight={warm_weight})")

            # Merge the entire backtest file into the live state.
            # merge_warmstart will handle extracting symbol-specific states for all symbols in the file.
            merge_warmstart(latest_backtest_file, live_state_path, warmstart_weight=warm_weight)
            logger.info(f"Warm-start merge complete for all symbols from {latest_backtest_file}.")
        else:
            logger.info(f"No backtest bandit state file found to warm-start any symbol.")

    except Exception:
        logger.exception(f"Warmstart merge failed; continuing without warmstart.")


def _process_closed_trades(exe, live_monitor, risk_controller, ens_per_symbol_long, ens_per_symbol_short, mt5c):
    """Reconciles closed positions with MT5, feeds them to the monitor and bandit, and logs reward metrics."""
    closed_trades = exe.reconcile_open_positions_with_mt5()
    if closed_trades:
        for trade in closed_trades:
            # Add to monitor for metrics
            live_monitor.add_closed_trade(trade)
            # Update RiskController for learning
            risk_controller.update(trade)

            # Log metrics for the closed trade
            try:
                # Need to get the correct ensemble for the trade's symbol
                ens = ens_per_symbol_long[trade.symbol] if trade.direction == "long" else ens_per_symbol_short[trade.symbol]
                trade_auc = ens.ensemble_cv_auc_ if ens else 0.5

                normalized_reward = trade.pnl / max(1.0, trade.exit_equity) if trade.exit_equity else 0.0
                drawdown = 1 - (live_monitor.current_equity / live_monitor.peak_equity) if live_monitor.peak_equity > 0 else 0.0

                log_metrics_to_csv({
                    "timestamp": trade.exit_time.isoformat(),
                    "symbol": trade.symbol,
                    "event_type": "trade_reward",
                    "atr_idx": trade.atr_idx,
                    "min_prob_long_idx": trade.min_prob_long_idx,
                    "min_prob_short_idx": trade.min_prob_short_idx,
                    "reward": normalized_reward,
                    "equity": trade.exit_equity,
                    "peak_equity": live_monitor.peak_equity,
                    "drawdown": drawdown,
                    "ensemble_auc": trade_auc
                })
            except Exception as e:
                logger.exception(f"[{trade.symbol}] Failed to log metrics for closed trade {trade.ticket}: {e}")

        # After processing closed trades, update the equity from the broker
        account_info = mt5c.account_info()
        if account_info:
            live_monitor.update_equity(datetime.datetime.now(datetime.timezone.utc), account_info.equity)
            logger.info(f"Global equity updated to: {account_info.equity}")


def run(dry_run: bool = True):
    """ Production-ready main loop for hybrid adaptive MT5 ML bot. """
    require_live_permission(dry_run)
    RECONNECTION_RETRY_SECONDS = 60
    cfg = Cfg.from_yaml("config.yaml")
    apply_mode_state_paths(cfg, dry_run)  # dry-run learning must not seed live bandit/monitor state
    setup_logging(level=cfg.logging['level'])

    cfg.dashboard_every_bars = getattr(cfg, "dashboard_every_bars", 10)

    if cfg.startup_logging:
        log_startup_summary(cfg)

    logger.info("=== Starting MT5 ML Bot (Hybrid Adaptive) ===")
    logger.info(f"Dry-run mode: {dry_run}")
    logger.info(f"Symbols: {cfg.symbols if hasattr(cfg, 'symbols') else []}")

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
    risk_manager = None  # Declare risk_manager here
    exe = None  # Declare exe here
    mt5c = None
    try:
        # Outer loop for MT5 reconnection attempts
        while True:
            symbol_threads = []  # the except path below must never act on a list from an earlier iteration
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
                initial_equity = equity_from_account(account_info)
                if initial_equity is None:  # never start on a made-up equity: it skews drawdown and sizing (K15)
                    logger.error("MT5 gave no usable account equity. Retrying in 60 seconds...")
                    notifier.send_message("<b>CRITICAL:</b> MT5 account info unavailable. Retrying...", level="CRITICAL")
                    time.sleep(RECONNECTION_RETRY_SECONDS)
                    continue
                cfg.initial_equity = initial_equity  # Set initial equity in Cfg for the monitor

                live_monitor = LivePerformanceMonitor(cfg)
                live_monitor.account_id = f"{getattr(account_info, 'login', None)}@{getattr(account_info, 'server', None)}"
                live_monitor.load_state()  # Load previous state on startup

                # Immediately after loading state, sync the current_equity with the live account value
                # This ensures the bot starts with the ground truth from the broker.
                live_monitor.sync_equity(initial_equity)

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

                    # Bootstrap context feature history as well
                    if cfg.context_features.mta.enabled:
                        logger.info(f"[{sym}] Bootstrapping MTA history...")
                        data_manager.bootstrap_history(sym, initial_bars=initial_bars, timeframe=cfg.context_features.mta.timeframe)
                    if cfg.context_features.inter_market.enabled:
                        im_sym = cfg.context_features.inter_market.symbol
                        logger.info(f"[{sym}] Bootstrapping Inter-Market history for {im_sym}...")
                        data_manager.bootstrap_history(im_sym, initial_bars=initial_bars, timeframe=cfg.timeframe)

                bar_counters = {sym: 0 for sym in cfg.symbols}
                last_bar_time = {sym: None for sym in cfg.symbols}
                X_per_symbol = {}

                # Warm-start bandit: merge latest backtest priors into live state file BEFORE instantiating RiskController
                _warmstart_bandit(cfg)

                # Instantiate risk manager and risk controller AFTER warmstart merge so they load the merged state
                lock = threading.Lock()
                risk_manager = RiskManager(cfg, mt5c, lock, notifier=notifier)  # Pass notifier
                risk_controller = RiskController(cfg, notifier=notifier)  # Instantiate RiskController
                loaded_open_positions = risk_controller.load_state()  # Load state again to get open_positions_cache

                # Execution object (single instance)
                exe = Execution(ens_per_symbol_long, ens_per_symbol_short, risk_manager, mt5c, data_manager, dry_run=dry_run, notifier=notifier, monitor=live_monitor)
                exe.risk.open_positions_cache.update(loaded_open_positions)  # Initialize exe's cache with loaded data

                # Reconcile open positions with MT5 to ensure accuracy
                exe.reconcile_open_positions_with_mt5()

                retraining_status = retraining_status_for(cfg.symbols, retraining_processes)  # a child that outlived the reconnect stays tracked (K22)
                trading_blocked_by_low_new_model_auc = {sym: False for sym in cfg.symbols}  # Track if trading is blocked due to low new model AUC
                last_diagnostics_log_time = 0.0  # For throttling diagnostics logging
                # Initialize last_retrain_date from RiskController's loaded state
                last_retrain_date = risk_controller.last_daily_retrain_date

                # --- Start Symbol Processors ---
                symbol_threads = []  # Initialize here to prevent UnboundLocalError
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
                    risk_controller.save_state(exe.risk.cache_snapshot())  # snapshot under the lock; the save and any alert run outside it

                    # --- Centralized Reconciliation ---
                    # This is the single point of truth for reconciling closed positions
                    _process_closed_trades(exe, live_monitor, risk_controller, ens_per_symbol_long, ens_per_symbol_short, mt5c)

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
                            retraining_status, last_retrain_date, risk_controller, mt5c
                        )

                        # Check if a retraining process has finished
                        if retraining_status[sym] and not retraining_processes[sym].is_alive():
                            logger.info(f"[{sym}] Background retraining process finished.")
                            _handle_model_acceptance(
                                sym, cfg, ens_per_symbol_long, ens_per_symbol_short,
                                active_model_auc, live_monitor, notifier, optuna_params_per_symbol
                            )
                            retraining_status[sym] = False
                            del retraining_processes[sym]  # Clean up the process entry

                    # Sleep for a short interval before checking again
                    time.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.exception(f"MT5 connection lost or critical error in trading loop: {e}. Attempting to reconnect...")
                notifier.send_message(f"<b>CRITICAL:</b> MT5 connection lost or critical error: {e}. Attempting to reconnect...", level="CRITICAL")
                # Stop the old symbol threads first (K7): otherwise they keep looping next to the new ones after a reconnect.
                try:
                    stop_symbol_threads(symbol_threads)
                except Exception:
                    logger.exception("Failed to stop symbol threads after error.")
                # Shutdown all symbol-specific MT5 clients before attempting main reconnection
                for symbol_data in symbol_threads:
                    try:
                        symbol_data["mt5_client"].shutdown()
                    except Exception:
                        logger.exception(f"Failed to shutdown MT5 client for {symbol_data['symbol']} after error.")
                try:
                    mt5c.shutdown()  # Ensure old main connection is closed
                except Exception:
                    logger.exception("Failed to shutdown main mt5 client after error.")
                try:
                    teardown_connection()  # close the real connection; every old client (also the DataManager ones) now reads disconnected
                except Exception:
                    logger.exception("Failed to tear down the MT5 connection after error.")
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
            try:
                stop_symbol_threads(symbol_threads)
            except Exception:
                logger.exception("Failed to stop symbol threads on shutdown.")
            for symbol_data in symbol_threads:
                try:
                    symbol_data["mt5_client"].shutdown()
                except Exception:
                    logger.exception(f"Failed to shutdown MT5 client for {symbol_data['symbol']} cleanly.")

        try:
            teardown_connection()
        except Exception:
            logger.exception("Failed to tear down the MT5 connection on shutdown.")

        if live_monitor:
            try:
                live_monitor.save_state()  # Save state on shutdown
            except Exception:
                logger.error("Failed to save live monitor state on shutdown.", exc_info=True)

        if risk_controller and exe:  # Check if exe is also defined
            try:
                # Use the latest open positions cache from the live monitor, as it's the aggregate from all threads
                risk_controller.save_state(exe.risk.cache_snapshot())  # Save final state
            except Exception:
                logger.exception("Failed to save RiskController state on shutdown.")
        logger.info("MT5 ML Bot shutdown complete.")
        notifier.send_message("MT5 ML Bot shutdown complete.", level="INFO")


if __name__ == "__main__":
    # Dry-run by default. Live orders need run(dry_run=False) AND ALLOW_LIVE_TRADING=1 in the environment.
    run(dry_run=True)
    # run(dry_run=False)
