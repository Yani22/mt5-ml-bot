# src/utils.py
from __future__ import annotations
import json
import math
import os
import shutil
import sys
from loguru import logger  # type: ignore
import pandas as pd  # type: ignore
from src.features import FeatureCfg, build_features
from src.ignored_settings import ignored_setting_warnings
from src.labels import generate_labels, generate_long_short_labels
from src.ensemble import Ensemble
from src.config import Cfg, ConfigError, check_tuned_params
from src import data_manager
from src.data import merge_features_labels
import numpy as np
import csv
from typing import Dict, Any

MODEL_DIR = "models"
PARAMS_DIR = "optuna_params"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)

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


def log_metrics_to_csv(data: Dict[str, Any]):
    with open(METRICS_CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        row = [data.get(header, "") for header in METRICS_HEADERS]
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def setup_logging(level="INFO", to_file=True, rotate="10 MB", retention="7 days"):
    logger.remove()
    # Use enqueue=True to make logging from multiple processes safe for shared sinks (stderr and the log file).
    # This will prevent messages from being garbled, but they will still be mixed.
    # The log messages themselves should contain context like the symbol name.
    logger.add(sys.stderr, level=level, enqueue=True)
    if to_file:
        os.makedirs("logs", exist_ok=True)
        logger.add("logs/bot.log", level=level, rotation=rotate, retention=retention, enqueue=True)


def setup_logging_from_config(logging_cfg: dict) -> None:
    """`setup_logging` with the `logging:` block of config.yaml (level, to_file, rotate, retention); a missing key keeps its default."""
    logging_cfg = logging_cfg or {}
    setup_logging(level=logging_cfg.get("level", "INFO"), to_file=logging_cfg.get("to_file", True),
                  rotate=logging_cfg.get("rotate", "10 MB"), retention=logging_cfg.get("retention", "7 days"))


def optuna_params_filename(symbol: str) -> str:
    """File name for a symbol's tuned params, shared by the tuner (writer) and the bot (reader). The '#' is dropped."""
    return f"{symbol.replace('#', '')}_best_params.json"


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"tuned params must be plain data, got {type(value).__name__}")


def save_optuna_params(symbol: str, params: dict) -> str:
    """Writes the tuned params as JSON (never a pickle: nothing in `optuna_params/` is unpickled) and returns the path."""
    os.makedirs(PARAMS_DIR, exist_ok=True)
    path = os.path.join(PARAMS_DIR, optuna_params_filename(symbol))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, default=_json_default)
    os.replace(tmp, path)
    return path


def load_optuna_params(symbol: str, cfg: Cfg) -> dict | None:
    file_path = os.path.join(PARAMS_DIR, optuna_params_filename(symbol))
    if not os.path.exists(file_path):
        legacy = os.path.join(PARAMS_DIR, f"{symbol.replace('#', '')}_best_params.pkl")
        if os.path.exists(legacy):
            raise ConfigError(f"[{symbol}] {legacy} is a pickled tuned-params file from an older version. It is never loaded "
                              f"(a pickle can run code). Delete it, or re-run tuner.py to write {os.path.basename(file_path)}.")
        logger.warning(f"[{symbol}] No Optuna params found at {file_path}, using defaults from config.")
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            loaded_params = json.load(f)
    except Exception as e:
        raise ConfigError(f"[{symbol}] tuned params in {file_path} cannot be read ({e}). Fix or delete the file, "
                          f"or re-run tuner.py.") from e

    if not isinstance(loaded_params, dict) or "models" not in loaded_params:
        logger.warning(f"[{symbol}] Optuna params format unexpected; using empty model params.")
        # Ensure min_pct_change is always present, even if optuna_params is empty
        return {"lgbm": {}, "xgb": {}, "rf": {}, "logreg": {}, "features": {"min_pct_change": cfg.features.min_pct_change}}

    check_tuned_params(loaded_params, file_path)

    # Ensure min_pct_change is always present in features, falling back to cfg default
    if "features" not in loaded_params:
        loaded_params["features"] = {}
    if "min_pct_change" not in loaded_params["features"]:
        loaded_params["features"]["min_pct_change"] = cfg.features.min_pct_change

    logger.debug(f"[{symbol}] Loaded Optuna best params from {file_path}")
    return loaded_params


def get_training_data(cfg: Cfg, symbol: str, feature_cfg: FeatureCfg, count: int | None = None, source: str = "csv", load_all_data: bool = False, build_dynamic: bool = True, min_pct_change: float = 0.0, prediction_horizon: int = 0, mta_df: pd.DataFrame | None = None, inter_market_df: pd.DataFrame | None = None, return_long_short_labels: bool = False, return_context: bool = False):
    """
    New centralized data pipeline.
    - If build_dynamic is True, returns (data, X, y) or (data, X, y_long, y_short) for trainers/backtesters.
    - If build_dynamic is False, returns (X, y, df) or (X, y_long, y_short, df) for the tuner.
    """
    fetch_count = None if load_all_data else (count if count is not None else cfg.history_bars)

    # --- 1. Initialize DataManager ---
    dm = data_manager.DataManager(cfg)

    # --- 2. Fetch All Dataframes ---
    logger.info(f"[{symbol}] Fetching primary data ({fetch_count or 'all'} bars, {cfg.timeframe}) from {cfg.data_source.upper()}...")
    if cfg.data_source == "csv":
        df = dm.load_local_history(symbol, cfg.timeframe, count=fetch_count)
    elif cfg.data_source == "mt5":
        df = dm._fetch_bars_from_mt5_chunked(symbol, cfg.timeframe, fetch_count)
    else:
        raise ValueError(f"Unknown data source: {cfg.data_source}")

    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype="float64")

    mta_df = None
    if cfg.context_features.mta.enabled:
        logger.info(f"[{symbol}] Fetching MTA data ({fetch_count or 'all'} bars, {cfg.context_features.mta.timeframe}) from {cfg.data_source.upper()}...")
        if cfg.data_source == "csv":
            mta_df = dm.load_local_history(symbol, cfg.context_features.mta.timeframe, count=fetch_count)
        elif cfg.data_source == "mt5":
            mta_df = dm._fetch_bars_from_mt5_chunked(symbol, cfg.context_features.mta.timeframe, fetch_count)
        else:
            raise ValueError(f"Unknown data source: {cfg.data_source}")
        if mta_df is None or mta_df.empty:
            raise RuntimeError(f"[{symbol}] No MTA data for timeframe {cfg.context_features.mta.timeframe}, but it is enabled: "
                               f"a model trained without it would not match the live features. Fetch the history or disable it in config.yaml.")

    inter_market_df = None
    if cfg.context_features.inter_market.enabled:
        im_sym = cfg.context_features.inter_market.symbol
        logger.info(f"[{symbol}] Fetching Inter-Market data for {im_sym} ({fetch_count or 'all'} bars, {cfg.timeframe}) from {cfg.data_source.upper()}...")
        if cfg.data_source == "csv":
            inter_market_df = dm.load_local_history(im_sym, cfg.timeframe, count=fetch_count)
        elif cfg.data_source == "mt5":
            inter_market_df = dm._fetch_bars_from_mt5_chunked(im_sym, cfg.timeframe, fetch_count)
        else:
            raise ValueError(f"Unknown data source: {cfg.data_source}")
        if inter_market_df is None or inter_market_df.empty:
            raise RuntimeError(f"[{symbol}] No Inter-Market data for symbol {im_sym}, but it is enabled: a model trained without it "
                               f"would not match the live features. Fetch the history or disable it in config.yaml.")

    # --- 3. Build Feature Set ---
    logger.info(f"[{symbol}] Building full feature set...")

    # Build all features using the unified build_features function
    X = build_features(df.copy(), feature_cfg, cfg, symbol=symbol, mta_df=mta_df, inter_market_df=inter_market_df)

    if return_long_short_labels:
        y_long, y_short = generate_long_short_labels(df, prediction_horizon, min_pct_change)
        # Align X and y by index
        aligned_idx = X.index.intersection(y_long.index)
        X = X.loc[aligned_idx]
        y_long = y_long.loc[aligned_idx]
        y_short = y_short.loc[aligned_idx]
    else:
        y = generate_labels(df, prediction_horizon, min_pct_change)
        # Align X and y by index
        aligned_idx = X.index.intersection(y.index)
        X = X.loc[aligned_idx]
        y = y.loc[aligned_idx]

    if not build_dynamic:
        # Return the intermediate artifacts needed by the tuner
        logger.info(f"[{symbol}] Data pipeline complete for tuner. Returning features and labels.")
        if return_long_short_labels:
            return X, y_long, y_short, df
        elif return_context:
            return X, y, df, mta_df, inter_market_df   # the tuner rebuilds the whole matrix per trial and needs the context frames
        else:
            return X, y, df  # Return X, y, df for consistency

    # For trainer/backtester, X and y are already built
    if return_long_short_labels:
        data = merge_features_labels(df, X, y_long)  # merge with y_long for consistency
    else:
        data = merge_features_labels(df, X, y)

    if data is None or data.empty:
        if return_long_short_labels:
            return pd.DataFrame(), X if X is not None else pd.DataFrame(), pd.Series(dtype="float64"), pd.Series(dtype="float64")
        else:
            return pd.DataFrame(), X if X is not None else pd.DataFrame(), y if y is not None else pd.Series(dtype="float64")

    logger.info(f"[{symbol}] Data pipeline complete. Final shape: {data.shape}")
    if return_long_short_labels:
        return data, X, y_long, y_short
    else:
        return data, X, y


def _model_name(symbol: str, model_type: str) -> str:
    return f"{symbol.replace('#', '')}_ensemble_{model_type}"


def model_dir_for(symbol: str, model_type: str, staged: bool = False) -> str:
    """The live model folder, or the staging folder a retrain child writes to until it is accepted."""
    base = os.path.join(MODEL_DIR, "_staging") if staged else MODEL_DIR
    return os.path.join(base, _model_name(symbol, model_type))


def recover_interrupted_promotion(symbol: str, model_type: str) -> None:
    """A crash between the two renames of `promote_staged_ensemble` leaves no live folder and a `.bak`: put it back, or the
    next load would build an untrained ensemble."""
    live = model_dir_for(symbol, model_type)
    if not os.path.isdir(live) and os.path.isdir(live + ".bak"):
        logger.warning(f"[{symbol}] {model_type} model folder missing; restoring {live}.bak (an interrupted promotion).")
        os.rename(live + ".bak", live)


def promote_staged_ensemble(symbol: str, model_type: str) -> None:
    """Makes the staged model the live one: live -> .bak, staged -> live, drop .bak; the old model is put back if the
    second rename fails. The manifest signs file contents and relative paths only, so a rename keeps it valid."""
    live, staged = model_dir_for(symbol, model_type), model_dir_for(symbol, model_type, staged=True)
    backup = live + ".bak"
    if os.path.isdir(backup):
        shutil.rmtree(backup)
    had_live = os.path.isdir(live)
    if had_live:
        os.rename(live, backup)
    try:
        os.rename(staged, live)
    except Exception:
        if had_live:
            os.rename(backup, live)
        raise
    if had_live:
        shutil.rmtree(backup, ignore_errors=True)


def discard_staged_ensemble(symbol: str, model_type: str) -> None:
    shutil.rmtree(model_dir_for(symbol, model_type, staged=True), ignore_errors=True)


def load_ensemble(cfg: Cfg, symbol: str, model_type: str, model_params: dict | None = None) -> Ensemble:
    recover_interrupted_promotion(symbol, model_type)
    # New: ensemble is saved in a directory, not a single file
    model_dir_path = model_dir_for(symbol, model_type)

    # Load model_params if not provided
    if model_params is None:
        model_params = load_optuna_params(symbol, cfg)

    if os.path.isdir(model_dir_path):
        logger.debug(f"[{symbol}] Loading saved ensemble from directory {model_dir_path}")
        try:
            # Use the new class method to load
            return Ensemble.load(model_dir_path, cfg, model_params=model_params)
        except Exception as e:
            logger.exception(f"[{symbol}] Failed to load ensemble from directory: {e}; creating a new one.")

    logger.info(f"[{symbol}] No saved ensemble directory found. Creating a new one.")
    ens = Ensemble(cfg, model_params=model_params)
    return ens


def save_ensemble(ensemble: Ensemble, symbol: str, model_type: str, staged: bool = False):
    # New: save to a directory
    model_dir_path = model_dir_for(symbol, model_type, staged=staged)
    try:
        if staged:
            shutil.rmtree(model_dir_path, ignore_errors=True)    # never mix files of two retrains
        # Use the new instance method to save
        ensemble.save(model_dir_path)
        logger.info(f"[{symbol}] Ensemble model saved to directory {model_dir_path}")
    except Exception as e:
        logger.error(f"[{symbol}] Failed to save ensemble: {e}")


def retrain_is_usable(ens_new) -> tuple[bool, str]:
    """Whether a freshly fitted ensemble may replace the live one. The newest model replaces it whenever it fitted (a member has
    its feature names; `Ensemble.fit` skips a small sample with a warning and then predicts 0.5 for every bar) and reports a
    finite AUC. It is not compared with the old model's stored AUC (another window, noisy at a daily retrain): the
    `min_ensemble_auc` gate decides whether it trades (C2)."""
    new_auc = getattr(ens_new, "ensemble_cv_auc_", getattr(ens_new, "cv_auc_", None))
    if new_auc is None or not math.isfinite(float(new_auc)):
        return False, "it reports no usable AUC"
    names = getattr(ens_new, "feature_names", None)
    if callable(names) and names() is None:
        return False, "no member was fitted (too few samples)"
    return True, ""


def safe_retrain_ensemble(cfg: Cfg, symbol: str, ens_old: Ensemble, X_train: pd.DataFrame, y_train: pd.Series, prices: pd.Series, dry_run: bool = False, model_type: str = "long", model_params: dict | None = None, staged: bool = False, n_jobs: int = -1) -> Ensemble:
    """
    Retrains an ensemble and returns the new one when it fitted, otherwise the old one.

    Args:
        cfg: The configuration object.
        symbol: The symbol being trained.
        ens_old: The existing ensemble model.
        X_train: The training features.
        y_train: The training labels.
        prices: The close prices for the training period.
        dry_run: If True, the new model will not be saved.
        model_type: The type of model being retrained ("long" or "short").
        model_params: Pre-loaded Optuna parameters for the model.
        staged: save an accepted model to the staging folder instead of the live one (the live bot promotes it after its own check).
        n_jobs: threads the new ensemble may use while fitting (-1 = every core).

    Returns:
        The retrained ensemble if it fitted (see `retrain_is_usable`), otherwise the old ensemble.
    """
    logger.info(f"[{symbol}] Starting safe retraining...")

    old_auc = getattr(ens_old, "ensemble_cv_auc_", getattr(ens_old, "cv_auc_", None))

    # Create a new ensemble to avoid feature mismatch issues
    # Load model_params if not provided
    if model_params is None:
        model_params = load_optuna_params(symbol, cfg)
    ens_new = Ensemble(cfg, model_params=model_params, n_jobs=n_jobs)

    try:
        ens_new.fit(X_train, y_train, prices=prices, model_type=model_type)
        new_auc = getattr(ens_new, "ensemble_cv_auc_", getattr(ens_new, "cv_auc_", None))

        usable, why = retrain_is_usable(ens_new)
        if not usable:
            logger.warning(f"[{symbol}] {model_type.upper()} retrain refused: {why}. Keeping the old model.")
            return ens_old
        if not dry_run:
            save_ensemble(ens_new, symbol, model_type, staged=staged)
        logger.info(f"[{symbol}] {model_type.upper()} Retrain accepted. old_auc={old_auc} new_auc={new_auc}")
        return ens_new
    except Exception as e:
        logger.exception(f"[{symbol}] Retraining failed: {e}")
        return ens_old


def log_symbol_specific_configs(cfg: "Cfg"):
    """Logs the resolved symbol-specific configurations to verify overrides."""
    logger.info("--- Verifying Symbol-Specific Configurations ---")
    logger.info(f"Thompson Sampling Enabled: {cfg.thompson_sampling.enabled}")

    # List of all keys that can be overridden per symbol
    keys_to_check = [
        'min_prob_long', 'min_prob_short', 'atr_multiplier_sl',
        'atr_multiplier_tp', 'trailing_atr_mult', 'min_ensemble_auc',
        'min_auc_improvement', 'atr_grid', 'min_prob_grid_long',
        'min_prob_grid_short'
    ]

    for sym in cfg.symbols:
        logger.info(f"--- Settings for Symbol: {sym} ---")
        for key in keys_to_check:
            # Get the resolved value for the symbol
            resolved_value = cfg.get_symbol_value(sym, key)

            # Get the global default to compare against
            global_default = None
            if hasattr(cfg.risk, key):
                global_default = getattr(cfg.risk, key)
            elif hasattr(cfg.thompson_sampling, key):
                global_default = getattr(cfg.thompson_sampling, key)

            # Determine if an override was used. This is a heuristic for logging.
            is_override = " (Override)" if resolved_value != global_default and global_default is not None else " (Default)"

            # For list (grid) comparison, the above is not sufficient. Let's refine.
            if isinstance(resolved_value, list):
                if sorted(resolved_value) != sorted(global_default):
                    is_override = " (Override)"
                else:
                    is_override = " (Default)"

            logger.info(f"  > {key}: {resolved_value}{is_override}")


def log_startup_summary(cfg: "Cfg"):
    """Logs a summary of key configuration settings at startup."""
    logger.info("--- Bot Startup Configuration Summary ---")
    logger.info(f"Symbols: {cfg.symbols}")
    logger.info(f"Timeframe: {cfg.timeframe}")
    logger.info(f"History Bars: {cfg.history_bars}")
    logger.info(f"Prediction Horizon: {cfg.prediction_horizon}")
    logger.info(f"Thompson Sampling Enabled: {cfg.thompson_sampling.enabled}")
    logger.info(f"Max Portfolio Risk: {cfg.risk.max_portfolio_risk}")
    logger.info(f"Dynamic Risk Enabled: {cfg.risk.dynamic_risk['enabled']}")
    for message in ignored_setting_warnings(cfg):
        logger.warning(f"Setting without effect: {message}")
    logger.info("--- End of Summary ---")


def ensure_min_grid_size(thresholds: list[float], best_thr: float, min_size: int = 5, spread: float = 0.02) -> list[float]:
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
