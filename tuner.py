# tuner.py
from __future__ import annotations
import hashlib
import json
import os
import pandas as pd  # type: ignore
import optuna  # type: ignore
from loguru import logger  # type: ignore
import yaml
from functools import partial
from joblib import Parallel, delayed  # type: ignore
import traceback  # Added for detailed error logging

from src.config import Cfg
from src.features import add_relative_features, build_dynamic_features, model_matrix, resolve_feature_cfg
from src.data_colab import merge_features_labels
from src.utils import get_training_data, save_optuna_params
from src.ensemble import Ensemble
from sklearn.model_selection import TimeSeriesSplit  # type: ignore
from sklearn.metrics import roc_auc_score  # type: ignore

# --- Detect Colab and set path ---
# Assumes drive is already mounted if running in Colab.
try:
    import google.colab  # type: ignore  # noqa: F401  (availability probe)
    # This path should point to the location in your Google Drive where params are stored.
    PARAMS_DIR = "/content/drive/MyDrive/mt5_ml_bot_params/optuna_params"
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    PARAMS_DIR = "optuna_params"

os.makedirs(PARAMS_DIR, exist_ok=True)

# --- Load config ---
cfg = Cfg.from_yaml("config.yaml")

with open("config.yaml", "r") as f:
    yaml_cfg = yaml.safe_load(f)


def suggest_params(trial, prefix: str, param_ranges: dict):
    params = {}
    for k, v in param_ranges.items():
        if isinstance(v, list) and len(v) in [2, 3]:
            if len(v) == 3 and v[2] == "log":
                params[k] = trial.suggest_float(f"{prefix}_{k}", v[0], v[1], log=True)
            else:
                if isinstance(v[0], int) and isinstance(v[1], int):
                    params[k] = trial.suggest_int(f"{prefix}_{k}", v[0], v[1])
                else:
                    params[k] = trial.suggest_float(f"{prefix}_{k}", v[0], v[1])
        else:
            params[k] = v
    return params


from src.labels import generate_labels  # NEW IMPORT

# ... (rest of imports) ...


def fold_auc(y_val: pd.Series, p_val) -> float:
    """AUC of one validation block. A fitted model drops the rows it cannot score (warmup NaN), so its output can be shorter
    than the labels: score the labels it did score."""
    if isinstance(p_val, pd.Series):
        y_val = y_val.reindex(p_val.index)
    return roc_auc_score(y_val, p_val)


def study_signature(label: tuple, feature_ranges: dict, models: list, columns: list, cv_samples: int, roc_lags_options=None) -> str:
    """A short hash of everything a trial's score depends on: the (fixed) label, the feature and model search spaces and
    defaults, the model's input columns and the CV size. A change starts a new study instead of adding trials to an old one."""
    payload = json.dumps({"label": list(label), "features": feature_ranges, "models": models, "columns": sorted(columns),
                          "cv_samples": cv_samples, "roc_lags_options": roc_lags_options}, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def study_name_for(sym: str, history_bars: int, signature: str) -> str:
    return f"feature_model_tuning_{sym.replace('#', '_')}_history_{history_bars}_{signature}"


def objective(trial, df: pd.DataFrame, static_features: pd.DataFrame, symbol: str):
    try:
        # --- 0. The label is fixed by the config, not tuned: AUCs of different label definitions are not comparable, and
        # tuning them picks the label that is easiest to predict, not one worth trading (C3).
        prediction_horizon = cfg.prediction_horizon
        min_pct_change = cfg.features.min_pct_change

        # --- 1. Suggest Feature Parameters ---
        feature_params_raw = suggest_params(trial, "feature", yaml_cfg.get("features", {}))

        roc_lags_options = yaml_cfg.get("roc_lags_options", [
            (1, 3, 5, 10), (1, 2, 4, 8), (2, 5, 10, 15), (1, 2, 3)
        ])

        if "roc_lags" in feature_params_raw:
            del feature_params_raw["roc_lags"]

        roc_lags_choice = trial.suggest_categorical("feature_roc_lags", roc_lags_options)
        feature_params_raw["roc_lags"] = roc_lags_choice

        feature_cfg = resolve_feature_cfg(cfg, feature_params_raw)

        # --- 2. Build Features for this Trial (using cached static features) ---
        X = add_relative_features(build_dynamic_features(df, static_features, feature_cfg, symbol), df)   # the matrix the live models are trained on

        # --- Generate Labels for this Trial ---
        y = generate_labels(df, prediction_horizon, min_pct_change)

        # Align X and y by index
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        data = merge_features_labels(df, X, y)

        X_train = data.drop(columns=["y", "close", "high", "low", "volume"])
        y_train = data["y"]

        # --- 3. Suggest Model Hyperparameters ---
        model_params = {}
        for model in yaml_cfg["models"]:
            model_name = model["name"]
            model_params[model_name] = suggest_params(trial, f"model_{model_name}", model.get("tune", {}))

        # --- 4. Evaluate Ensemble ---
        ens = Ensemble(cfg, model_params=model_params)

        cv_samples_per_split = yaml_cfg.get("cv_samples_per_split", 300)
        n_splits_calculated = min(5, max(2, len(X_train) // cv_samples_per_split))
        logger.debug(f"Calculated n_splits for TimeSeriesSplit: {n_splits_calculated}")

        tscv = TimeSeriesSplit(n_splits=n_splits_calculated, gap=int(prediction_horizon))   # labels look this far ahead
        aucs = []
        for i, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            ens.fit(X_tr, y_tr, cv=False)
            p_val = ens.predict_proba(X_val)
            auc = fold_auc(y_val, p_val)
            aucs.append(auc)

            trial.report(1 - auc, i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_auc = float(pd.Series(aucs).mean())
        return 1 - mean_auc

    except optuna.exceptions.TrialPruned as e:
        raise e  # Allow Optuna to handle pruning
    except Exception as e:
        tb_str = traceback.format_exc()
        logger.error(f"--- Trial Failed ---\nError: {e}\nTraceback:\n{tb_str}")
        return float('inf')


def structure_best_params(best_params_flat: dict) -> dict:
    """Splits Optuna's flat parameter names into the tuned-params file layout. The label (horizon, threshold) is NOT written:
    the tuner does not tune it (C3), and a copy of today's config values would silently override a later edit of
    `prediction_horizon` for every reader of the file while other code reads the config directly."""
    structured = {"features": {}, "models": {}}
    for key, value in best_params_flat.items():
        if key.startswith("feature_"):
            structured["features"][key.replace("feature_", "", 1)] = value
        elif key.startswith("model_"):
            parts = key.split('_')
            structured["models"].setdefault(parts[1], {})['_'.join(parts[2:])] = value
    return structured


def run_tuning_for_symbol(sym: str):
    logger.info(f"🔹 Starting combined feature and model tuning for {sym}...")

    # --- 1. Get all data and static features from the centralized pipeline ---
    # For tuning, we pass a default FeatureConfig and set build_dynamic=False.
    # The dynamic features will be built inside the objective function for each trial.
    static_features, _, df = get_training_data(  # Unpack X, discard y, get df
        cfg,
        sym,
        feature_cfg=cfg.features,  # the configured untuned features; the trials rebuild the dynamic ones
        source=cfg.data_source if hasattr(cfg, "data_source") else "csv",
        build_dynamic=False,  # Instruct the pipeline to return intermediate artifacts for tuner
        return_long_short_labels=False  # We will generate labels inside the objective
    )

    if df.empty:
        logger.error(f"[{sym}] No data returned from pipeline. Skipping tuning.")
        return

    # --- 2. Run Optuna Study ---
    objective_partial = partial(objective, df=df, static_features=static_features, symbol=sym)

    # One study per feature set, label and search space: an old study with the same name would keep adding trials scored on other
    # features (C3). The columns are the model's input columns for the default feature config.
    columns = list(model_matrix(add_relative_features(build_dynamic_features(df, static_features, cfg.features, sym), df)).columns)
    signature = study_signature((cfg.prediction_horizon, cfg.features.min_pct_change), yaml_cfg.get("features", {}),
                                yaml_cfg.get("models", []), columns, yaml_cfg.get("cv_samples_per_split", 300),
                                yaml_cfg.get("roc_lags_options"))
    study_name = study_name_for(sym, cfg.history_bars, signature)
    storage_path = f"sqlite:///{os.path.join(PARAMS_DIR, study_name)}.db"

    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="minimize", study_name=study_name, storage=storage_path, load_if_exists=True, pruner=pruner)

    n_trials = yaml_cfg.get("optuna_n_trials", 100)
    study.optimize(objective_partial, n_trials=n_trials)

    # --- 5. Process and Save Best Parameters ---
    best_params_structured = structure_best_params(study.best_params)

    param_file = save_optuna_params(sym, best_params_structured)

    logger.info(f"[{sym}] Best combined params saved to {param_file}")
    logger.debug(best_params_structured)

    # --- 6. Generate and Save Visualization Plots ---
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(os.path.join(PARAMS_DIR, f"{study_name}_optimization_history.png"))

        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image(os.path.join(PARAMS_DIR, f"{study_name}_param_importances.png"))
    except (ImportError, RuntimeError) as e:
        logger.warning(f"Could not generate plots. Make sure you have 'plotly' and 'kaleido' installed. Error: {e}")


# --- Main Execution Loop (Parallelized) ---
if __name__ == '__main__':
    n_jobs = yaml_cfg.get("n_jobs", -1)
    Parallel(n_jobs=n_jobs)(delayed(run_tuning_for_symbol)(sym) for sym in cfg.symbols)
