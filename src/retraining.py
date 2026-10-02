# src/retraining.py
"""Scheduled background retraining and model-acceptance logic for the live bot."""
import datetime
from multiprocessing import Process
from typing import Any, Dict

from loguru import logger

from src.config import Cfg, FeatureCfg
from src.data_manager import DataManager
from src.labels import generate_long_short_labels
from src.mt5_client import MT5Client
from src.notifier import TelegramNotifier
from src.risk_controller import RiskController
from src.utils import load_ensemble, safe_retrain_ensemble


def run_retraining_in_background(cfg, sym, feature_cfg, dry_run, notifier, optuna_params_per_symbol):
    """
    A wrapper function to run the entire retraining pipeline for both long and short models in a separate process.
    Tuned parameters come from optuna_params/<symbol>_best_params.pkl; config.yaml is never modified.
    """
    try:
        data_manager = DataManager(cfg)

        # Retrieve tuned prediction_horizon and min_pct_change for this symbol
        tuned_prediction_horizon = optuna_params_per_symbol[sym].get('prediction_horizon', cfg.prediction_horizon)
        tuned_min_pct_change = optuna_params_per_symbol[sym].get('min_pct_change', feature_cfg.min_pct_change)

        full_data, full_X, _ = data_manager.load_cached(sym, feature_cfg, count=cfg.retraining_window_bars, min_pct_change=tuned_min_pct_change)

        y_long, y_short = generate_long_short_labels(full_data, tuned_prediction_horizon, tuned_min_pct_change)
        # Labels drop the unknown-future tail; align features and prices to them.
        full_X = full_X.loc[y_long.index]
        full_data = full_data.loc[y_long.index]

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
            if notifier:
                notifier.send_message(message, level="INFO")
        else:
            message = f"[{sym}] New LONG model rejected (AUC: {old_auc_long:.4f} -> {new_auc_long:.4f}). Keeping old model."
            logger.warning(message)
            if notifier:
                notifier.send_message(message, level="WARNING")

        if short_accepted:
            ens_per_symbol_short[sym] = new_ens_short
            message = f"[{sym}] New SHORT model accepted (AUC: {old_auc_short:.4f} -> {new_auc_short:.4f})."
            logger.info(message)
            if notifier:
                notifier.send_message(message, level="INFO")
        else:
            message = f"[{sym}] New SHORT model rejected (AUC: {old_auc_short:.4f} -> {new_auc_short:.4f}). Keeping old model."
            logger.warning(message)
            if notifier:
                notifier.send_message(message, level="WARNING")

    except Exception as e:
        logger.exception(f"[{sym}] Error during model acceptance: {e}")


def _check_and_trigger_retraining(cfg: Cfg, sym: str, feature_cfg_per_symbol: Dict[str, FeatureCfg], dry_run: bool, notifier: TelegramNotifier, optuna_params_per_symbol: Dict[str, Any], retraining_processes: Dict[str, Process], retraining_status: Dict[str, bool], last_retrain_date: Dict[str, datetime.date], risk_controller: RiskController, mt5c: MT5Client):
    """
    Checks if retraining should be triggered for a given symbol based on retrain_time_utc.
    """
    current_utc_datetime = mt5c.now_utc()
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
                    logger.info(f"[{sym}] Triggering background retraining at {current_utc_datetime.time().strftime('%H:%M')} UTC...")
                    notifier.send_message(f"[{sym}] Triggering background retraining.", level="INFO")

                    process = Process(
                        target=run_retraining_in_background,
                        args=(cfg, sym, feature_cfg_per_symbol[sym], dry_run, notifier, optuna_params_per_symbol)
                    )
                    process.start()
                    retraining_processes[sym] = process
                    retraining_status[sym] = True

                    last_retrain_date[sym] = current_utc_date  # Mark as retrained for today
                    risk_controller.update_last_daily_retrain_date(sym, current_utc_date)  # Update RiskController's internal state
                else:
                    logger.info(f"[{sym}] Retraining already in progress for today. Skipping.")
                return  # Only trigger once per day per symbol
        except ValueError:
            logger.error(f"[{sym}] Invalid retrain_time_utc format: {retrain_time_str}. Expected HH:MM.")
        except Exception as e:
            logger.exception(f"[{sym}] Error checking or triggering retraining: {e}")
