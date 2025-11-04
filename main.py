# main.py
import os
import time
import copy
from multiprocessing import Process
from dotenv import load_dotenv
from loguru import logger
import pandas as pd
from src.config import Cfg, FeatureCfg
import MetaTrader5 as mt5  # type: ignore
from src.mt5_client import MT5Client
from src.risk import RiskManager
from src.execution import Execution
from src.utils import setup_logging, get_training_data, load_ensemble, save_ensemble, safe_retrain_ensemble, load_optuna_params, log_symbol_specific_configs, log_startup_summary
from src.live_performance_monitor import LivePerformanceMonitor
from src.notifier import TelegramNotifier
from src.alert_manager import AlertManager # NEW
from src.risk_controller import RiskController
import datetime
import json
from typing import Dict, Any, List
from src.data_manager import DataManager, DataStatus
from src.tca_analyzer import TcaAnalyzer
from src.labels import generate_long_short_labels
from src.bandit_warmstart import find_latest_backtest_state, merge_warmstart
import csv
import yaml
import numpy as np
from typing import List

def _ensure_min_grid_size(thresholds: List[float], best_thr: float, min_size: int = 5, spread: float = 0.02) -> List[float]:
    """Ensures a list of thresholds has at least min_size elements, expanding around best_thr if needed."""
    if len(thresholds) >= min_size:
        return sorted(list(set(thresholds)))  # Ensure unique and sorted

    # If not enough, generate a new grid around best_thr
    new_thresholds = [best_thr]
    half_size = (min_size - 1) // 2
    for i in range(1, half_size + 1):
        new_thresholds.append(best_thr + i * spread)
        new_thresholds.append(best_thr - i * spread)

    # Combine with existing and ensure unique, sorted, and within reasonable bounds
    combined = sorted(list(set(thresholds + new_thresholds)))

    # Filter to reasonable range (e.g., 0.0 to 1.0 for probabilities)
    combined = [round(x, 2) for x in combined if 0.0 <= x <= 1.0]

    # If still not enough after filtering, just take a wider range
    if len(combined) < min_size:
        combined = np.linspace(max(0.0, best_thr - spread * min_size), min(1.0, best_thr + spread * min_size), min_size).tolist()
        combined = [round(x, 2) for x in combined]

    return sorted(list(set(combined)))

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

TCA_CSV_FILE = "results/tca_metrics.csv"
TCA_HEADERS = [
    "timestamp", "symbol", "direction", "lots", "intended_entry_price", "actual_entry_price",
    "slippage_pips", "slippage_currency", "spread_at_entry_pips", "commission_per_trade",
    "total_transaction_cost_currency", "order_type", "fill_type", "deviation_pips",
    "retries_taken", "entry_auc", "entry_equity", "position_id"
]

def _initialize_metrics_csv():
    if not os.path.exists(METRICS_CSV_FILE):
        with open(METRICS_CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(METRICS_HEADERS)
        logger.info(f"Initialized metrics CSV file: {METRICS_CSV_FILE}")

def _initialize_tca_csv():
    if not os.path.exists(TCA_CSV_FILE):
        with open(TCA_CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(TCA_HEADERS)
        logger.info(f"Initialized TCA CSV file: {TCA_CSV_FILE}")

# Call initialization at startup
_initialize_metrics_csv()
_initialize_tca_csv()

def log_metrics_to_csv(data: Dict[str, Any]):
    with open(METRICS_CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        row = [data.get(header, "") for header in METRICS_HEADERS]
        writer.writerow(row)

def log_tca_to_csv(data: Dict[str, Any]):
    with open(TCA_CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        row = [data.get(header, "") for header in TCA_HEADERS]
        writer.writerow(row)

def log_tca_to_csv(data: Dict[str, Any]):
    with open(TCA_CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        row = [data.get(header, "") for header in TCA_HEADERS]
        writer.writerow(row)

def sleep_until_next_bar(cfg: Cfg, current_bar_timestamp: datetime.datetime, buffer_seconds: float = 5.0):
    """
    Calculates the precise time to sleep until the next bar opens, with a buffer.
    This prevents drift and ensures the bot wakes up just before new data is available.
    """
    timeframe_seconds = cfg.timeframe_seconds()
    if timeframe_seconds is None:
        logger.warning("Invalid timeframe_seconds, defaulting to 60 seconds sleep.")
        time.sleep(60)
        return

    # Calculate the expected open time of the *next* bar
    # current_bar_timestamp is the timestamp of the *last completed* bar
    # So, next_bar_open_time is current_bar_timestamp + timeframe_seconds
    next_bar_open_timestamp = current_bar_timestamp + datetime.timedelta(seconds=timeframe_seconds)

    # Calculate how long we need to sleep
    time_to_sleep = (next_bar_open_timestamp - datetime.datetime.now(datetime.timezone.utc)).total_seconds()

    if time_to_sleep <= buffer_seconds:
        # If we are already too close or past the next bar open time, just log and proceed
        logger.warning(f"Already past or too close to next bar open time. Proceeding without full sleep. Time to next bar: {time_to_sleep:.2f}s")
        # Small sleep to prevent busy-waiting if already past
        if time_to_sleep < -timeframe_seconds: # If we are more than a full timeframe behind, something is wrong
            logger.error(f"Bot is significantly behind market data. Current time: {datetime.datetime.now(datetime.timezone.utc)}, Expected next bar: {next_bar_open_timestamp}")
        time.sleep(1) # Sleep for a second to avoid busy loop
        return

    # Sleep for most of the duration, leaving a buffer
    sleep_duration = time_to_sleep - buffer_seconds
    if sleep_duration > 0:
        logger.debug(f"Sleeping for {sleep_duration:.2f} seconds until {buffer_seconds:.2f}s before next bar open.")
        time.sleep(sleep_duration)

    # Busy-wait for the remaining buffer time to ensure precise wake-up
    while datetime.datetime.now(datetime.timezone.utc) < next_bar_open_timestamp:
        time.sleep(0.1) # Small sleep to avoid 100% CPU usage
    logger.debug(f"Woke up precisely at next bar open time: {datetime.datetime.now(datetime.timezone.utc)}")

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

def run_retraining_in_background(cfg, sym, feature_cfg, dry_run, alert_manager, optuna_params_per_symbol):
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
            alert_manager.send_alert(message, level="WARNING", category="RETRAINING")
            return

        y_long, y_short = generate_long_short_labels(full_data, cfg.prediction_horizon, feature_cfg.min_pct_change)

        logger.info(f"[{sym}] Retraining LONG model...")
        ens_old_long = load_ensemble(cfg, sym, "long", model_params=optuna_params_per_symbol[sym])
        safe_retrain_ensemble(cfg, sym, ens_old_long, full_X, y_long, full_data["close"], dry_run=dry_run, model_type="long", model_params=optuna_params_per_symbol[sym])

        logger.info(f"[{sym}] Retraining SHORT model...")
        ens_old_short = load_ensemble(cfg, sym, "short", model_params=optuna_params_per_symbol[sym])
        safe_retrain_ensemble(cfg, sym, ens_old_short, full_X, y_short, full_data["close"], dry_run=dry_run, model_type="short", model_params=optuna_params_per_symbol[sym])

        logger.info(f"[{sym}] Background retraining process for LONG and SHORT models finished.")

    except Exception as e:
        logger.exception(f"[{sym}] Background retraining process failed: {e}")
        alert_manager.send_alert(f"[{sym}] Background retraining process failed: {e}", level="ERROR", category="RETRAINING")

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
            alert_manager.send_alert(message, level="INFO", category="MODEL_ACCEPTANCE")
        else:
            message = f"[{sym}] New LONG model rejected (AUC: {old_auc_long:.4f} -> {new_auc_long:.4f}). Keeping old model."
            logger.warning(message)
            alert_manager.send_alert(message, level="WARNING", category="MODEL_ACCEPTANCE")

        if short_accepted:
            ens_per_symbol_short[sym] = new_ens_short
            message = f"[{sym}] New SHORT model accepted (AUC: {old_auc_short:.4f} -> {new_auc_short:.4f})."
            logger.info(message)
            alert_manager.send_alert(message, level="INFO", category="MODEL_ACCEPTANCE")
        else:
            message = f"[{sym}] New SHORT model rejected (AUC: {old_auc_short:.4f} -> {new_auc_short:.4f}). Keeping old model."
            logger.warning(message)
            alert_manager.send_alert(message, level="WARNING", category="MODEL_ACCEPTANCE")

    except Exception as e:
        logger.exception(f"[{sym}] Error during model acceptance: {e}")

def run(dry_run: bool = False):
    """ Production-ready main loop for hybrid adaptive MT5 ML bot. """
    RECONNECTION_RETRY_SECONDS = 60
    cfg = Cfg.from_yaml("config.yaml")
    log_symbol_specific_configs(cfg)

    cfg.dashboard_every_bars = getattr(cfg, "dashboard_every_bars", 10)

    if cfg.startup_logging:
        log_startup_summary(cfg)

    logger.info("=== Starting MT5 ML Bot (Hybrid Adaptive) ===")
    logger.info(f"Dry-run mode: {dry_run}")
    logger.info(f"Symbols: {cfg.symbols if hasattr(cfg,'symbols') else []}")

    # Initialize notifier
    notifier = TelegramNotifier(cfg)

    # Initialize AlertManager
    alert_manager = AlertManager(cfg, notifier) # NEW

    # Initialize TCA Analyzer
    tca_analyzer = TcaAnalyzer(cfg)
    tca_analyzer.set_alert_manager(alert_manager) # Pass alert_manager

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
    trading_paused_due_to_data_issue = {sym: False for sym in cfg.symbols} # Track if trading is paused due to data issues
    consecutive_data_issues = {sym: 0 for sym in cfg.symbols} # Track consecutive data issues for recovery
    last_diagnostics_log_time = 0.0  # For throttling diagnostics logging
    last_retrain_date = {sym: None for sym in cfg.symbols}  # Track last retraining date per symbol

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
                    alert_manager.send_alert("MT5 initial connection failed. Retrying...", level="CRITICAL", category="MT5_CONNECTION")
                    time.sleep(RECONNECTION_RETRY_SECONDS)
                    continue  # Try connecting again

                logger.info("MT5 connection established.")
                alert_manager.send_alert("MT5 connection established.", level="INFO", category="MT5_CONNECTION")

                # Get initial equity from MT5 account info
                account_info = mt5.account_info()
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

                # Instantiate risk controller AFTER warmstart merge so it loads the merged state
                risk = RiskManager(cfg, alert_manager=alert_manager)  # Pass alert_manager
                risk_controller = RiskController(cfg, alert_manager=alert_manager)  # Pass alert_manager
                loaded_open_positions = risk_controller.load_state()  # Load state again to get open_positions_cache

                # Execution object (single instance)
                exe = Execution(ens_per_symbol_long, ens_per_symbol_short, risk, mt5c, data_manager, dry_run=dry_run, alert_manager=alert_manager)
                exe.risk.open_positions_cache.update(loaded_open_positions)  # Initialize exe's cache with loaded data

                # Reconcile open positions with MT5 to ensure accuracy
                exe.reconcile_open_positions_with_mt5()

                retraining_status = {sym: False for sym in cfg.symbols}  # Track if retraining is active
                trading_blocked_by_low_new_model_auc = {sym: False for sym in cfg.symbols}  # Track if trading is blocked due to low new model AUC
                last_diagnostics_log_time = 0.0  # For throttling diagnostics logging
                last_retrain_date = {sym: None for sym in cfg.symbols}  # Track last retraining date per symbol
                last_tca_analysis_time = 0.0 # For throttling TCA analysis
                last_state_save_time = 0.0 # For throttling state saving

                # Inner loop for trading operations
                while True:
                    current_loop_time = time.time()  # Capture current time for throttling
                    now_utc = datetime.datetime.now(datetime.timezone.utc)

                    # refresh account info once per loop
                    account_info = mt5.account_info()
                    equity = getattr(account_info, "equity", 0.0) if account_info else 0.0
                    live_monitor.update_equity(datetime.datetime.now(datetime.timezone.utc), equity)
                    balance = getattr(account_info, "balance", 0.0) if account_info else 0.0
                    drawdown = 0.0
                    try:
                        drawdown = 1 - (equity / balance) if balance else 0.0
                    except Exception:
                        drawdown = 0.0

                    # --- Process closed trades first so bandit gets rewards before opening new trades ---
                    latest_prices = {sym: mt5.symbol_info_tick(sym).ask for sym in cfg.symbols}
                    closed_trades_this_cycle = exe.check_closed_trades(latest_prices)

                    # --- FIX: Reconstruct sequential equity to provide accurate reward normalization ---
                    # The `equity` from account_info is after all trades in the cycle have closed.
                    # We work backwards to find the equity state before this cycle's trades.
                    if closed_trades_this_cycle:
                        # Assuming trades are sorted by close time from check_closed_trades()
                        total_profit_this_cycle = sum(t.pnl for t in closed_trades_this_cycle)
                        equity_before_cycle = equity - total_profit_this_cycle

                        running_equity = equity_before_cycle
                        for trade in closed_trades_this_cycle:
                            try:
                                # Calculate the exact equity at the moment this trade closed
                                exit_equity = running_equity + trade.pnl

                                # Set exit_equity on the trade object so RiskController can compute accurate reward
                                # This assumes the trade object is mutable and the controller knows to use this attribute.
                                trade.exit_equity = exit_equity

                                live_monitor.add_closed_trade(trade)
                                risk_controller.update_after_trade(trade.symbol, trade)

                                # For logging, calculate the accurate normalized reward
                                normalized_reward = trade.pnl / max(1.0, exit_equity)
                                trade_auc = active_model_auc.get(trade.symbol, 0.5)
                                log_metrics_to_csv({
                                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                    "symbol": trade.symbol,
                                    "event_type": "trade_reward",
                                    "atr_idx": trade.atr_idx,
                                    "min_prob_long_idx": trade.min_prob_long_idx,
                                    "min_prob_short_idx": trade.min_prob_short_idx,
                                    "reward": normalized_reward,
                                    "equity": exit_equity,  # Log the accurate equity
                                    "peak_equity": risk.equity_peak,
                                    "drawdown": drawdown,
                                    "ensemble_auc": trade_auc
                                })

                                # Update running_equity for the next trade in the sequence
                                running_equity = exit_equity

                            except Exception:
                                logger.exception("Error processing closed trade.")

                    # --- Per-symbol processing: fetch, update cache, optionally retrain, then make trade decision for that symbol ---
                    for sym in cfg.symbols:
                        try:
                            # --- Check for finished retraining processes for this symbol ---
                            if sym in retraining_processes and not retraining_processes[sym].is_alive():
                                logger.info(f"[{sym}] Background retraining process finished. Joining and reloading models.")
                                p = retraining_processes[sym]
                                p.join()

                                if p.exitcode != 0:
                                    message = f"[{sym}] <b>ERROR:</b> Background retraining process failed with exit code {p.exitcode}. Keeping current models."
                                    logger.error(message)
                                    alert_manager.send_alert(message, level="ERROR", category="RETRAINING")
                                else:
                                    # Call the centralized model acceptance function
                                    _handle_model_acceptance(sym, cfg, ens_per_symbol_long, ens_per_symbol_short, active_model_auc, live_monitor, alert_manager, optuna_params_per_symbol)

                                del retraining_processes[sym]
                                retraining_status[sym] = False
                                logger.info(f"[{sym}] Model handling complete.")

                            # --- Fetch latest bar data using the centralized DataManager pipeline ---
                            feature_cfg = feature_cfg_per_symbol[sym]
                            data, X, y, data_status, data_msg = data_manager.fetch_live(sym, feature_cfg, feature_cfg.min_pct_change, notifier)

                            # --- Data Validation Logic ---
                            if data_status != DataStatus.OK:
                                consecutive_data_issues[sym] += 1
                                if not trading_paused_due_to_data_issue[sym]:
                                    if consecutive_data_issues[sym] >= cfg.fetch.max_consecutive_data_issues:
                                        trading_paused_due_to_data_issue[sym] = True
                                        logger.critical(f"[{sym}] CRITICAL: Trading paused due to persistent data issues: {data_msg}")
                                        alert_manager.send_alert(f"[{sym}] Trading paused due to persistent data issues. {data_msg}", level="CRITICAL", category="DATA_ISSUE")
                                    else:
                                        logger.warning(f"[{sym}] Data issue detected (consecutive: {consecutive_data_issues[sym]}): {data_msg}")
                                        alert_manager.send_alert(f"[{sym}] Data issue detected. {data_msg}", level="WARNING", category="DATA_ISSUE")
                                else:
                                    logger.warning(f"[{sym}] Trading remains paused due to data issues: {data_msg}")
                                continue # Skip further processing for this symbol if data is not OK
                            else:
                                # Data is OK. Check for recovery.
                                if trading_paused_due_to_data_issue[sym]:
                                    # Increment valid fetches to recover
                                    data_manager.consecutive_valid_fetches[sym] += 1
                                    if data_manager.consecutive_valid_fetches[sym] >= cfg.fetch.min_valid_fetches_to_recover:
                                        trading_paused_due_to_data_issue[sym] = False
                                        data_manager.consecutive_valid_fetches[sym] = 0 # Reset counter
                                        logger.info(f"[{sym}] Data feed recovered. Resuming trading.")
                                        alert_manager.send_alert(f"[{sym}] Data feed recovered. Resuming trading.", level="INFO", category="DATA_ISSUE")
                                    else:
                                        logger.info(f"[{sym}] Data OK, but still recovering ({data_manager.consecutive_valid_fetches[sym]}/{cfg.fetch.min_valid_fetches_to_recover} valid fetches). Trading remains paused.")
                                else:
                                    # Data is OK and not paused, reset consecutive issues
                                    consecutive_data_issues[sym] = 0
                                    data_manager.consecutive_valid_fetches[sym] = 0 # Ensure this is reset on continuous OK

                            if data.empty:
                                continue

                            # Skip further processing if trading is paused for this symbol due to data issues
                            if trading_paused_due_to_data_issue[sym]:
                                continue

                            X_per_symbol[sym] = X
                            latest_bar_time = data.index[-1]
                            if last_bar_time[sym] == latest_bar_time:
                                continue  # skip if no new bar
                            last_bar_time[sym] = latest_bar_time
                            bar_counters[sym] += 1
                            logger.info(f"[{sym}] New bar detected at {latest_bar_time}")

                            # --- Live Dashboard (throttled per-symbol) ---
                            if bar_counters[sym] % cfg.dashboard_every_bars == 0:
                                print_dashboard(
                                    cfg,
                                    risk,
                                    ens_per_symbol_long.get(sym),  # Use long model for dashboard prob
                                    X_per_symbol.get(sym),
                                    sym,
                                    bar_counters[sym],
                                    is_first_symbol=(cfg.symbols.index(sym) == 0),
                                    equity=equity,
                                    balance=balance
                                )

                            # Delta append new bars to cache (atomic write)
                            data_manager.append_new_bars(sym, data)

                            # --- Conditional Retraining Logic ---
                            should_retrain = False
                            time_to_retrain_today = False
                            
                            # Get symbol-specific or global retrain time
                            retrain_time_str = cfg.get_symbol_value(sym, 'retrain_time_utc', cfg.fetch.retrain_time_utc)

                            if retrain_time_str:
                                if last_retrain_date[sym] is None or last_retrain_date[sym] < now_utc.date():
                                    retrain_hour, retrain_minute = map(int, retrain_time_str.split(':'))
                                    if now_utc.hour > retrain_hour or (now_utc.hour == retrain_hour and now_utc.minute >= retrain_minute):
                                        time_to_retrain_today = True
                                        last_retrain_date[sym] = now_utc.date()

                            if time_to_retrain_today:
                                should_retrain = True
                            elif not retrain_time_str:  # Fallback to bar count if time-based is disabled for this symbol
                                if bar_counters[sym] > 0 and bar_counters[sym] % cfg.retrain_every_bars == 0:
                                    should_retrain = True

                            if should_retrain:
                                if cfg.fetch.retrain_in_background:
                                    if sym not in retraining_processes:
                                        logger.info(f"[{sym}] Triggering background retraining (time-based: {time_to_retrain_today}).")
                                        alert_manager.send_alert(f"[{sym}] Background retraining started.", level="INFO", category="RETRAINING")
                                        p = Process(target=run_retraining_in_background, args=(cfg, sym, feature_cfg, dry_run, alert_manager, optuna_params_per_symbol))
                                        p.start()
                                        retraining_processes[sym] = p
                                        retraining_status[sym] = True
                                    else:
                                        logger.info(f"[{sym}] Retraining already in progress. Skipping trigger.")
                                else:
                                    # Synchronous retraining
                                    logger.info(f"[{sym}] Starting synchronous retraining. Trading loop will pause.")
                                    alert_manager.send_alert(f"[{sym}] Synchronous retraining started. Bot is paused.", level="INFO", category="RETRAINING")

                                    run_retraining_in_background(cfg, sym, feature_cfg, dry_run, alert_manager, optuna_params_per_symbol)

                                    logger.info(f"[{sym}] Synchronous retraining finished. Reloading models...")
                                    _handle_model_acceptance(sym, cfg, ens_per_symbol_long, ens_per_symbol_short, active_model_auc, live_monitor, alert_manager, optuna_params_per_symbol)

                                    logger.info(f"[{sym}] Models reloaded. Resuming trading loop.")
                                    alert_manager.send_alert(f"[{sym}] Synchronous retraining finished. Resuming operations.", level="INFO", category="RETRAINING")

                            # --- Now handle trading decision for this symbol ---
                            try:
                                atr = float(X["atr_14"].iloc[-1]) if (X is not None and not X.empty) else 0.0
                            except Exception:
                                atr = 0.0

                            risk.manage_open_positions(sym, atr)

                            # --- Consecutive Loss Watchdog Check ---
                            if cfg.watchdog.enabled:
                                max_losses = getattr(cfg.watchdog, "max_consecutive_losses", 0)
                                if max_losses > 0:
                                    consecutive_losses = risk_controller.symbol_states[sym].consecutive_losses
                                    if consecutive_losses >= max_losses:
                                        if risk.cooldown_until is None:  # Only trigger if not already in cooldown
                                            logger.warning(f"[{sym}] Watchdog triggered: {consecutive_losses} consecutive losses >= threshold ({max_losses}). Pausing trading.")
                                            risk._trigger_cooldown(cooldown_hours=getattr(cfg.watchdog, "cooldown_hours", 1))
                                            alert_manager.send_alert(f"[{sym}] Watchdog triggered due to {consecutive_losses} consecutive losses. Trading paused.", level="WARNING", category="WATCHDOG")

                            last_features = X.iloc[[-1]] if (X is not None and not X.empty) else pd.DataFrame()

                            # Permission checks (drawdown/session)
                            if not risk.should_trade(pd.Timestamp.now(), drawdown):
                                logger.info(f"[{sym}] Trade skipped due to drawdown/session rules")
                                continue

                            # Get ensembles for this symbol
                            ens_long = ens_per_symbol_long[sym]
                            ens_short = ens_per_symbol_short[sym]

                            # Get symbol-specific thresholds using the new helper
                            min_prob_long = cfg.get_symbol_value(sym, 'min_prob_long', 0.55)
                            min_prob_short = cfg.get_symbol_value(sym, 'min_prob_short', 0.55)

                            # Check ensemble confidence before trading
                            min_required_auc = cfg.get_symbol_value(sym, 'min_ensemble_auc', 0.55)
                            if ens_long.ensemble_cv_auc_ < min_required_auc and ens_short.ensemble_cv_auc_ < min_required_auc:
                                logger.info(f"[{sym}] Trading blocked. Both models below min AUC. Long AUC: {ens_long.ensemble_cv_auc_:.4f}, Short AUC: {ens_short.ensemble_cv_auc_:.4f}")
                                continue

                            # Get dynamic risk params from RiskController
                            context = {
                                "vol": atr,
                                "equity": equity,
                                "peak_equity": risk.equity_peak,
                                "ensemble_auc": ens_long.ensemble_cv_auc_,  # Use long model's AUC for context
                                "adx": float(last_features["adx"].iloc[0]) if "adx" in last_features.columns else 0.0,
                                "macd_diff": float(last_features["macd_diff"].iloc[0]) if "macd_diff" in last_features.columns else 0.0,
                                "volatility_10": float(last_features["volatility_10"].iloc[0]) if "volatility_10" in last_features.columns else 0.0,
                                "dist_from_ema_200": float(last_features["dist_from_ema_200"].iloc[0]) if "dist_from_ema_200" in last_features.columns else 0.0,
                            }
                            dynamic_risk_params = risk_controller.get_params(sym, context)

                            # The risk controller returns the correct thresholds, whether from the bandit or static config
                            min_prob_long = dynamic_risk_params["min_prob_long"]
                            min_prob_short = dynamic_risk_params["min_prob_short"]

                            # Decision / trade
                            if last_features.empty:
                                logger.debug(f"[{sym}] Skipping decision: no features for latest bar")
                                continue

                            prob_long = ens_long.predict_proba(last_features).iloc[0]
                            prob_short = ens_short.predict_proba(last_features).iloc[0]

                            direction = None
                            auc_score = 0.5
                            if prob_long >= min_prob_long and ens_long.ensemble_cv_auc_ >= min_required_auc:
                                direction = "long"
                                auc_score = ens_long.ensemble_cv_auc_
                            elif prob_short >= min_prob_short and ens_short.ensemble_cv_auc_ >= min_required_auc:
                                direction = "short"
                                auc_score = ens_short.ensemble_cv_auc_

                            if direction:
                                total_open_risk = sum([pos.get('risk', 0.0) for pos in risk.open_positions_cache.values()])

                                symbol_info = mt5.symbol_info(sym)
                                if not symbol_info:
                                    logger.warning(f"[{sym}] Symbol info unavailable. Skipping position size calculation.")
                                    continue

                                pip_size = getattr(symbol_info, "point", 0.0001)
                                contract_size = getattr(symbol_info, "trade_contract_size", 100000.0)
                                pip_value = pip_size * contract_size

                                spread_pips = getattr(cfg.trading_costs.defaults, 'spread_pips', 2.0)
                                spread_value = spread_pips * pip_size

                                lots, effective_risk = risk.position_size(equity, atr, pip_value, pip_size, auc_score, spread_value, total_open_risk, symbol=sym)

                                if lots <= 0:
                                    logger.info(f"[{sym}] Trade skipped due to risk limits or position size zero.")
                                else:
                                    price = float(mt5.symbol_info_tick(sym).ask) if direction == "long" else float(mt5.symbol_info_tick(sym).bid)
                                    sl, tp = risk.stop_targets(price, atr, direction, auc_score, sym, sl_mult=dynamic_risk_params["atr_multiplier_sl"], tp_mult=dynamic_risk_params["atr_multiplier_tp"])

                                    result, tca_data = exe.trade(
                                        symbol=sym,
                                        direction=direction,
                                        lots=lots,
                                        price=price,
                                        sl=sl,
                                        tp=tp,
                                        equity=equity,
                                        pip_size=pip_size,
                                        pip_value=pip_value,
                                        X=last_features,
                                        atr=atr,
                                        auc_score=auc_score,
                                        total_open_risk=total_open_risk,
                                        atr_idx=dynamic_risk_params["atr_idx"],
                                        min_prob_long_idx=dynamic_risk_params.get("min_prob_long_idx") if direction == "long" else -1,
                                        min_prob_short_idx=dynamic_risk_params.get("min_prob_short_idx") if direction == "short" else -1
                                    )

                                    if tca_data:
                                        log_tca_to_csv(tca_data)

                                    if result.ok:
                                        logger.info(f"[{sym}] Trade executed: {result.message}")
                                    else:
                                        logger.info(f"[{sym}] Trade skipped: {result.message}")
                            else:
                                logger.info(f"[{sym}] No trade signal. Probs: (Long: {prob_long:.3f}, Short: {prob_short:.3f})")
                        except Exception:
                            logger.exception(f"Per-symbol loop failed for {sym}")

                    # --- Update RiskManager's equity peak for drawdown tracking (done for whole loop) ---
                    risk._update_equity_peak(equity)

                    # --- Diagnostics Logging ---
                    # Throttled logging for RiskController diagnostics
                    if current_loop_time - last_diagnostics_log_time >= (cfg.timeframe_seconds() * cfg.dashboard_every_bars):
                        ts_diagnostics = risk_controller.diagnostics()
                        # logger.info(f"RiskController Diagnostics: {json.dumps(ts_diagnostics, indent=2)}")
                        last_diagnostics_log_time = current_loop_time

                    # --- Automated TCA Analysis ---
                    if cfg.tca.enabled and (current_loop_time - last_tca_analysis_time) >= (cfg.tca.analysis_interval_hours * 3600):
                        logger.info("Triggering automated TCA analysis...")
                        tca_analyzer.run_analysis_and_notify(lookback_days=cfg.tca.lookback_days)
                        last_tca_analysis_time = current_loop_time

                    # --- Periodic State Saving ---
                    if (current_loop_time - last_state_save_time) >= (cfg.monitoring.state_save_interval_minutes * 60):
                        logger.info("Triggering periodic state save...")
                        try:
                            live_monitor.save_state()
                            exe._save_open_positions_state()
                            risk_controller.save_state(open_positions_cache=exe.risk.open_positions_cache)
                            logger.info("Periodic state save complete.")
                        except Exception as e:
                            logger.exception(f"Failed during periodic state save: {e}")
                            alert_manager.send_alert(f"Failed during periodic state save: {e}", level="ERROR", category="STATE_SAVE")
                        last_state_save_time = current_loop_time

                    # --- Periodic State Saving ---
                    if (current_loop_time - last_state_save_time) >= (cfg.monitoring.state_save_interval_minutes * 60):
                        logger.info("Triggering periodic state save...")
                        try:
                            live_monitor.save_state()
                            exe._save_open_positions_state()
                            risk_controller.save_state(open_positions_cache=exe.risk.open_positions_cache)
                            logger.info("Periodic state save complete.")
                        except Exception as e:
                            logger.exception(f"Failed during periodic state save: {e}")
                            alert_manager.send_alert(f"Failed during periodic state save: {e}", level="ERROR", category="STATE_SAVE")
                        last_state_save_time = current_loop_time

                    # --- Sleep until the next bar opens ---
                    # We need the latest_bar_time from *any* symbol that successfully fetched a new bar.
                    # If no new bar was detected for any symbol, we might have an issue or just be waiting.
                    # For simplicity, we'll use the latest_bar_time of the last processed symbol, or current time if none.
                    # A more robust solution might track the *earliest* next expected bar across all symbols.
                    # For now, assuming all symbols are on the same timeframe and roughly synchronized.
                    valid_bar_times = [t.to_pydatetime() for t in last_bar_time.values() if t is not None] # Convert to datetime
                    if valid_bar_times:
                        latest_overall_bar_time = max(valid_bar_times)
                        sleep_until_next_bar(cfg, latest_overall_bar_time)
                    else:
                        # If no new bar was detected for any symbol, just sleep for the timeframe duration
                        logger.warning("No new bar detected for any symbol. Sleeping for full timeframe duration.")
                        time.sleep(cfg.timeframe_seconds() or 60)

            except Exception as e:
                logger.exception(f"MT5 connection lost or critical error in trading loop: {e}. Attempting to reconnect...")
                alert_manager.send_alert(f"MT5 connection lost or critical error: {e}. Attempting to reconnect...", level="CRITICAL", category="MT5_CONNECTION")
                try:
                    mt5c.shutdown()  # Ensure old connection is closed
                except Exception:
                    logger.exception("Failed to shutdown mt5 client after error.")
                time.sleep(RECONNECTION_RETRY_SECONDS)  # Wait before retrying connection

    except KeyboardInterrupt:
        logger.info("=== Stopping MT5 ML Bot ===")
        alert_manager.send_alert("MT5 ML Bot stopped by user (KeyboardInterrupt).", level="WARNING", category="SHUTDOWN")
        # Optional: clean up any running child processes
        for sym, p in retraining_processes.items():
            if p.is_alive():
                logger.warning(f"[{sym}] Terminating running retraining process due to bot shutdown.")
                p.terminate()
                p.join()
    finally:
        try:
            live_monitor.save_state()  # Save state on shutdown
        except Exception:
            logger.exception("Failed to save live monitor state on shutdown.")
        # Save open positions state
        try:
            if 'exe' in locals() and exe is not None:
                exe._save_open_positions_state()
        except Exception:
            logger.exception("Failed to save open positions state on shutdown.")
        # save risk_controller state if present
        try:
            open_pos_cache_to_save = {}
            if 'exe' in locals() and exe is not None and hasattr(exe, 'risk') and hasattr(exe.risk, 'open_positions_cache'):
                open_pos_cache_to_save = exe.risk.open_positions_cache
            risk_controller.save_state(open_positions_cache=open_pos_cache_to_save)
        except Exception:
            logger.exception("Failed to save risk_controller state on shutdown.")
        try:
            mt5c.shutdown()
        except Exception:
            logger.exception("Failed to shutdown mt5 client cleanly.")
        logger.info("MT5 shutdown complete.")
        alert_manager.send_alert("MT5 ML Bot shutdown complete.", level="INFO", category="SHUTDOWN")

if __name__ == "__main__":
    # Default to dry-run to be safe; change to False when you are ready.
    run(dry_run=True)
