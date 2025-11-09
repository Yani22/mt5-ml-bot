# trainer.py
"""
Adaptive retraining module. Intended to be run periodically or from the main loop.
Performs safe retraining with no lookahead. On failure, keeps the previous ensemble.
Saves ensembles via utils.save_ensemble() and optionally writes training artifacts.
"""

from __future__ import annotations
import os
from loguru import logger  # type: ignore
import numpy as np
from typing import List

from src.config import Cfg
from src.features import FeatureCfg
from src.utils import load_ensemble, save_ensemble, safe_retrain_ensemble, load_optuna_params, get_training_data
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

def retrain_symbol(cfg: Cfg, symbol: str, dry_run: bool = True, mt5_instance=None) -> dict:
    """
    Retrain ensemble for one symbol safely.
    """
    logger.info(f"[{symbol}] Starting safe retrain (dry_run={dry_run})")

    if mt5_instance is None:
        logger.error(f"[{symbol}] MT5 connection instance not provided to retrain_symbol. Cannot proceed.")
        return {"ok": False, "reason": "MT5 connection not provided"}

    optuna_params = load_optuna_params(symbol, cfg)
    feature_params = optuna_params.get('features', {}) if optuna_params else {}
    feature_cfg = FeatureCfg(**feature_params)

    logger.info(f"[{symbol}] Loading full history via DataManager...")
    dm = DataManager(cfg)
    
    # Use get_training_data with build_dynamic=False and return_long_short_labels=True to get untrimmed data and features
    X, y_long, y_short, data = get_training_data(
        cfg=cfg,
        symbol=symbol,
        feature_cfg=feature_cfg,
        count=cfg.retraining_window_bars,
        min_pct_change=feature_cfg.min_pct_change,
        build_dynamic=False, # We need raw data (df) and features (X), will generate labels (y) later
        return_long_short_labels=True
    )

    if X is None or X.empty or len(X) < MIN_SAMPLES_TO_RETRAIN:
        msg = f"[{symbol}] Not enough data to retrain: {0 if X is None else len(X)} samples"
        logger.warning(msg)
        return {"ok": False, "reason": msg}

    # CRITICAL: Align X and y to their common index to prevent training on mismatched data
    common_idx = X.index.intersection(y_long.index)
    X = X.loc[common_idx]
    y_long = y_long.loc[common_idx]
    y_short = y_short.loc[common_idx]
    close_prices = data["close"].loc[common_idx]

    logger.info(f"[{symbol}] Aligned training data to {len(X)} bars.")

    results = {}
    results["long"] = train_and_save_model(cfg, symbol, "long", X, y_long, close_prices, dry_run=dry_run)
    results["short"] = train_and_save_model(cfg, symbol, "short", X, y_short, close_prices, dry_run=dry_run)

    return results

def retrain_all(cfg: Cfg, symbols: list[str], dry_run: bool = True, mt5_instance=None) -> dict:
    results = {}
    for s in symbols:
        try:
            results[s] = retrain_symbol(cfg, s, dry_run=dry_run, mt5_instance=mt5_instance)
        except Exception as e:
            logger.exception(f"[{s}] retrain_all error: {e}")
            results[s] = {"ok": False, "reason": str(e)}
    return results








if __name__ == "__main__":
    # import numpy as np # Removed from here
    # import random # Removed from here
    np.random.seed(42)
    random.seed(42)
    import os
    from dotenv import load_dotenv  # type: ignore
    from src.config import Cfg
    from src.mt5_client import MT5Client

    load_dotenv()

    # Establish MT5 connection
    mt5_client = MT5Client(
        login=os.getenv("MT5_LOGIN"),
        password=os.getenv("MT5_PASSWORD"),
        server=os.getenv("MT5_SERVER"),
        path=os.getenv("MT5_PATH"),
    )
    if not mt5_client.connect():
        logger.error("Failed to connect to MT5, exiting.")
        quit()

    try:
        cfg = Cfg.from_yaml("config.yaml")
        symbols = getattr(cfg, "symbols", [])
        if not symbols:
            print("No symbols found in config.yaml. Exiting.")
            quit()
        print("Running retrain_all with dry_run=False (model files will be overwritten).")
        res = retrain_all(cfg, symbols, dry_run=False, mt5_instance=mt5_client)
        print(res)
    finally:
        # Shutdown MT5 connection
        mt5_client.shutdown()
        print("MT5 connection shut down.")
