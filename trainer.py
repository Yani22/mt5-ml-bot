# trainer.py
"""
Adaptive retraining module. Intended to be run periodically or from the main loop.
Performs safe retraining with no lookahead. On failure, keeps the previous ensemble.
Saves ensembles via utils.save_ensemble() and optionally writes training artifacts.
"""

from __future__ import annotations
import os
from loguru import logger  # type: ignore
from typing import List # Added List

from src.config import Cfg
from src.features import FeatureCfg
from src.utils import load_ensemble, save_ensemble, safe_retrain_ensemble, load_optuna_params
from src.data_manager import DataManager
from src.ensemble import Ensemble
from src.labels import generate_long_short_labels
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import random

# Safe retraining parameters
MIN_SAMPLES_TO_RETRAIN = 1000  # don't retrain if less than this many samples

def train_and_save_model(cfg: Cfg, symbol: str, model_type: str, X: pd.DataFrame, y: pd.Series, prices: pd.Series, dry_run: bool = True) -> dict:
    """
    Trains and saves a single model (long or short).
    """
    logger.info(f"[{symbol}] Training {model_type} model...")
    ens_old = load_ensemble(cfg, symbol, model_type)
    if ens_old is None:
        logger.info(f"[{symbol}] No existing {model_type} ensemble; creating a new one")
        ens_old = Ensemble(cfg)

    ens_new = safe_retrain_ensemble(cfg, symbol, ens_old, X, y, prices, dry_run=dry_run, model_type=model_type)

    if ens_new is ens_old:
        new_auc = getattr(ens_new, "ensemble_cv_auc_", getattr(ens_new, "cv_auc_", None))
        old_auc = getattr(ens_old, "ensemble_cv_auc_", getattr(ens_old, "cv_auc_", None))
        return {"ok": False, "reason": "insufficient_improvement_or_failed", "old_auc": old_auc, "new_auc": new_auc}
    else:
        new_auc = getattr(ens_new, "ensemble_cv_auc_", getattr(ens_new, "cv_auc_", None))
        old_auc = getattr(ens_old, "ensemble_cv_auc_", getattr(ens_old, "cv_auc_", None))
        return {"ok": True, "old_auc": old_auc, "new_auc": new_auc}

def retrain_symbol(cfg: Cfg, symbol: str, dry_run: bool = True) -> dict:
    """
    Retrain ensemble for one symbol safely and update the config.yaml with symbol-specific overrides.
    """
    logger.info(f"[{symbol}] Starting safe retrain (dry_run={dry_run})")

    optuna_params = load_optuna_params(symbol, cfg)
    feature_params = optuna_params.get('features', {}) if optuna_params else {}
    feature_cfg = FeatureCfg(**feature_params)

    logger.info(f"[{symbol}] Loading full history via DataManager...")
    dm = DataManager(cfg)
    data, X, _ = dm.load_cached(symbol, feature_cfg, min_pct_change=feature_cfg.min_pct_change)

    if X is None or X.empty or len(X) < MIN_SAMPLES_TO_RETRAIN:
        msg = f"[{symbol}] Not enough data to retrain: {0 if X is None else len(X)} samples"
        logger.warning(msg)
        return {"ok": False, "reason": msg}

    y_long, y_short = generate_long_short_labels(data, cfg.prediction_horizon, feature_cfg.min_pct_change)

    results = {}
    results["long"] = train_and_save_model(cfg, symbol, "long", X, y_long, data["close"], dry_run=dry_run)
    results["short"] = train_and_save_model(cfg, symbol, "short", X, y_short, data["close"], dry_run=dry_run)

    return results

def retrain_all(cfg: Cfg, symbols: list[str], dry_run: bool = True) -> dict:
    results = {}
    for s in symbols:
        try:
            results[s] = retrain_symbol(cfg, s, dry_run=dry_run)
        except Exception as e:
            logger.exception(f"[{s}] retrain_all error: {e}")
            results[s] = {"ok": False, "reason": str(e)}
    return results


def _ensure_min_grid_size(thresholds: List[float], best_thr: float, min_size: int = 5, spread: float = 0.02) -> List[float]:
    """Ensures a list of thresholds has at least min_size elements, expanding around best_thr if needed."""
    if len(thresholds) >= min_size:
        return sorted(list(set(thresholds))) # Ensure unique and sorted

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


if __name__ == "__main__":
    # import numpy as np # Removed from here
    # import random # Removed from here
    np.random.seed(42)
    random.seed(42)
    import os
    from dotenv import load_dotenv  # type: ignore
    try:
        import MetaTrader5 as mt5 # type: ignore
    except ImportError:
        mt5 = None
        logger.warning("MetaTrader5 module not found. Live MT5 operations will be disabled.")
    from src.config import Cfg

    load_dotenv()

    # Establish MT5 connection
    if mt5:
        login_id_str = os.getenv("MT5_LOGIN")
        if not login_id_str:
            print("MT5_LOGIN not found in environment variables. Exiting.")
            quit()

        if not mt5.initialize(
            login=int(login_id_str),
            password=os.getenv("MT5_PASSWORD"),
            server=os.getenv("MT5_SERVER"),
            path=os.getenv("MT5_PATH")
        ):
            print(f"mt5.initialize() failed, error code = {mt5.last_error()}")
            quit()
        
        print("MT5 connection initialized.")
    else:
        print("MT5 connection skipped: MetaTrader5 module not available.")
    
    try:
        cfg = Cfg.from_yaml("config.yaml")
        symbols = getattr(cfg, "symbols", [])
        if not symbols:
            print("No symbols found in config.yaml. Exiting.")
            quit()
        print("Running retrain_all with dry_run=False (model files will be overwritten).")
        res = retrain_all(cfg, symbols, dry_run=False)
        print(res)
    finally:
        # Shutdown MT5 connection
        if mt5:
            mt5.shutdown()
            print("MT5 connection shut down.")
