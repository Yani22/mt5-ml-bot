# src/retraining.py
"""Scheduled background retraining and model-acceptance logic for the live bot."""
import datetime
import os
from multiprocessing import Process
from typing import Any, Dict

from loguru import logger

from src.config import Cfg, FeatureCfg
from src.data_manager import DataManager
from src.ensemble import Ensemble
from src.labels import generate_long_short_labels
from src.mt5_client import MT5Client
from src.notifier import TelegramNotifier
from src.risk_controller import RiskController
from src.utils import (discard_staged_ensemble, load_ensemble, model_dir_for, promote_staged_ensemble,
                       safe_retrain_ensemble)


def retraining_status_for(symbols, retraining_processes) -> Dict[str, bool]:
    """Per-symbol "a retrain child is being tracked" flags, derived from the process table so a reconnect cannot forget a child
    that is still running (it would be started again) or has finished and still waits for model acceptance."""
    return {sym: sym in retraining_processes for sym in symbols}


def run_retraining_in_background(cfg, sym, feature_cfg, dry_run, notifier, optuna_params_per_symbol):
    """
    A wrapper function to run the entire retraining pipeline for both long and short models in a separate process.
    Tuned parameters come from optuna_params/<symbol>_best_params.json; config.yaml is never modified.
    """
    try:
        # The child saves into models/_staging/, never into the live folder; `_handle_model_acceptance` promotes what it accepts.
        for side in ("long", "short"):
            discard_staged_ensemble(sym, side)   # nothing left over from an earlier or crashed run
        min_improvement = cfg.get_symbol_value(sym, 'min_auc_improvement', cfg.risk.min_auc_improvement)  # the acceptance threshold
        data_manager = DataManager(cfg)

        # Retrieve tuned prediction_horizon and min_pct_change for this symbol
        tuned = optuna_params_per_symbol[sym] or {}   # None when there is no tuned-params file: use the config values
        tuned_prediction_horizon = tuned.get('prediction_horizon', cfg.prediction_horizon)
        tuned_min_pct_change = tuned.get('min_pct_change', feature_cfg.min_pct_change)

        full_data, full_X, _ = data_manager.load_cached(sym, feature_cfg, count=cfg.retraining_window_bars, min_pct_change=tuned_min_pct_change)
        if full_data.empty or full_X.empty:
            logger.error(f"[{sym}] No training data (the cache or an enabled context is empty); retrain skipped, nothing staged.")
            return

        y_long, y_short = generate_long_short_labels(full_data, tuned_prediction_horizon, tuned_min_pct_change)
        # Labels drop the unknown-future tail; align features and prices to them.
        full_X = full_X.loc[y_long.index]
        full_data = full_data.loc[y_long.index]

        logger.info(f"[{sym}] Retraining LONG model...")
        ens_old_long = load_ensemble(cfg, sym, "long", model_params=optuna_params_per_symbol[sym])
        safe_retrain_ensemble(cfg, sym, ens_old_long, full_X, y_long, full_data["close"], dry_run=dry_run, model_type="long", model_params=optuna_params_per_symbol[sym],
                              staged=True, min_improvement=min_improvement)

        logger.info(f"[{sym}] Retraining SHORT model...")
        ens_old_short = load_ensemble(cfg, sym, "short", model_params=optuna_params_per_symbol[sym])
        safe_retrain_ensemble(cfg, sym, ens_old_short, full_X, y_short, full_data["close"], dry_run=dry_run, model_type="short", model_params=optuna_params_per_symbol[sym],
                              staged=True, min_improvement=min_improvement)

        logger.info(f"[{sym}] Background retraining process for LONG and SHORT models finished.")

    except Exception as e:
        logger.exception(f"[{sym}] Background retraining process failed: {e}")


def _push_to_processor(processors, sym, **sides):
    processor = (processors or {}).get(sym)
    if processor is not None:
        processor.set_models(**sides)


def _load_staged(cfg, sym, side, model_params):
    """The staged ensemble, or None when the child staged nothing. Raises when what is there cannot be loaded (an unsigned or
    half-written folder fails the manifest check)."""
    path = model_dir_for(sym, side, staged=True)
    if not os.path.isdir(path):
        return None
    return Ensemble.load(path, cfg, model_params=model_params)


def _handle_model_acceptance(sym, cfg, ens_per_symbol_long, ens_per_symbol_short, active_model_auc, live_monitor, notifier, optuna_params_per_symbol, processors=None):
    """Judges the models the retrain child staged and promotes the ones that improve on the live model.

    Each side is judged on its own, with the per-symbol `min_auc_improvement` (the threshold the child used). An accepted model is
    promoted on disk first, then replaced in `main`'s dicts and pushed into the running processor (K25: `processors` maps symbol ->
    SymbolProcessor; a symbol whose MT5 client did not connect has none). A rejected or unreadable one is deleted, so the live folder
    is never touched by a model that was not accepted.
    """
    logger.info(f"[{sym}] Handling model acceptance...")
    min_auc_improvement = cfg.get_symbol_value(sym, 'min_auc_improvement', cfg.risk.min_auc_improvement)
    model_params = (optuna_params_per_symbol or {}).get(sym)

    def alert(message, level):
        if notifier:
            notifier.send_message(message, level=level)

    def judge(side, live_models):
        label = side.upper()
        try:
            new_ens = _load_staged(cfg, sym, side, model_params)
        except Exception as e:
            logger.exception(f"[{sym}] Staged {label} model could not be loaded: {e}")
            discard_staged_ensemble(sym, side)
            alert(f"[{sym}] Staged {label} model could not be loaded and was discarded. Keeping old model.", "WARNING")
            return
        if new_ens is None:
            logger.info(f"[{sym}] No new {label} model was staged (the retrain was no improvement, it failed (see the child's log), or this is a dry run). Keeping the current one.")
            return

        new_auc = getattr(new_ens, "ensemble_cv_auc_", None)
        old_auc = getattr(live_models[sym], "ensemble_cv_auc_", None)   # None: no incumbent AUC, so any staged model with one is accepted
        if new_auc is None or (old_auc is not None and new_auc < old_auc + min_auc_improvement):
            shown = lambda v: "n/a" if v is None else f"{v:.4f}"  # noqa: E731
            message = f"[{sym}] New {label} model rejected (AUC: {shown(old_auc)} -> {shown(new_auc)}). Keeping old model."
            logger.warning(message)
            alert(message, "WARNING")
            discard_staged_ensemble(sym, side)
            return

        try:
            promote_staged_ensemble(sym, side)   # on disk first: a reconnect must not bring the old model back
        except Exception as e:
            logger.exception(f"[{sym}] Could not promote the staged {label} model: {e}")
            discard_staged_ensemble(sym, side)
            alert(f"[{sym}] Could not promote the new {label} model ({e}). Keeping old model.", "WARNING")
            return

        live_models[sym] = new_ens
        processor = (processors or {}).get(sym)
        if processor is not None:
            processor.set_models(**{side: new_ens})
        if side == "long":
            active_model_auc[sym] = new_auc
            live_monitor.update_ensemble_auc(new_auc)
        message = f"[{sym}] New {label} model accepted (AUC: {'n/a' if old_auc is None else f'{old_auc:.4f}'} -> {new_auc:.4f})."
        logger.info(message)
        alert(message, "INFO")

    for side, live_models in (("long", ens_per_symbol_long), ("short", ens_per_symbol_short)):
        try:
            judge(side, live_models)
        except Exception as e:   # never raise into main's loop: it would reconnect, and the finished child stays tracked
            logger.exception(f"[{sym}] Error during {side} model acceptance: {e}")
            discard_staged_ensemble(sym, side)


def _check_and_trigger_retraining(cfg: Cfg, sym: str, feature_cfg_per_symbol: Dict[str, FeatureCfg], dry_run: bool, notifier: TelegramNotifier, optuna_params_per_symbol: Dict[str, Any], retraining_processes: Dict[str, Process], retraining_status: Dict[str, bool], last_retrain_date: Dict[str, datetime.date], risk_controller: RiskController, mt5c: MT5Client, now_fn=None):
    """
    Checks if retraining should be triggered for a given symbol based on retrain_time_utc.

    The time is compared with real UTC (`now_fn`, the system clock by default), not the terminal's server clock, which is hours off UTC and
    would move the scheduled time with whichever source answers. `mt5c` is kept for the caller's signature and is not asked.
    """
    current_utc_datetime = (now_fn or (lambda: datetime.datetime.now(datetime.timezone.utc)))()
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
