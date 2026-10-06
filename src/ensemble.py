# src/ensemble.py
from __future__ import annotations
import os
import pickle
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import Dict, Optional, List, Tuple
from .strategy_ml import MLStrategy
from .model_integrity import sign as sign_model_dir, verify as verify_model_dir
from .config import Cfg
from .costs import round_trip_pips
from sklearn.isotonic import IsotonicRegression  # type: ignore
from loguru import logger  # type: ignore
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score  # type: ignore
from sklearn.model_selection import TimeSeriesSplit  # type: ignore
from dataclasses import asdict
from .config import TradingCostsDefaultsCfg


def custom_pnl(
    y_true: pd.Series,
    y_pred: pd.Series,
    prices: pd.Series,
    cfg: "Cfg",
    model_type: str = "long",
    **trading_costs
) -> float:
    """Sum of the net fractional returns of the trades the 0/1 signals in `y_pred` would take (see `net_trade_returns`): the
    forward move over `cfg.prediction_horizon` bars for the model's side, one trade per holding period, one round-trip cost.
    A 0 is no trade. `y_true` is unused and kept for the metric call signature."""
    if prices is None or len(prices) != len(y_pred):
        raise ValueError(f"Prices and predictions must be same length: {len(prices)} vs {len(y_pred)}")
    horizon = cfg.prediction_horizon
    if not horizon:
        raise ValueError("cfg.prediction_horizon must be set to calculate custom_pnl")

    pnl = net_trade_returns(y_pred, prices, horizon, model_type, **trading_costs)
    total = float(pnl.sum())
    logger.debug(f"custom_pnl: total={total:.6f}, trades={len(pnl)}")
    return total


_warned_non_forex_pips = False


def infer_pip_size(prices: pd.Series) -> float:
    """Pip size guessed from the price level, because the threshold search does not know the symbol: below 20 (EURUSD-like,
    5 digits) 0.0001, 20 to 1000 (JPY-like, 3 digits) 0.01. Anything above looks like gold or an index, where an FX pip means
    nothing: warn once and use 0.0001. An explicit `pip_size` in the cost dict always wins."""
    global _warned_non_forex_pips
    level = float(pd.Series(prices).dropna().median())
    if level < 20:
        return 0.0001
    if level < 1000:
        return 0.01
    if not _warned_non_forex_pips:
        _warned_non_forex_pips = True
        logger.warning(f"Threshold search: prices around {level:.0f} do not look like forex, so FX pip costs are not meaningful "
                       f"(using a pip of 0.0001); pass pip_size in the trading costs for this symbol.")
    return 0.0001


def net_trade_returns(y_pred: pd.Series, prices: pd.Series, horizon: int, model_type: str = "long", **trading_costs) -> pd.Series:
    """Net fractional return of every trade the signals in `y_pred` (0/1) would take, holding `horizon` bars.

    Each trade is counted once: the next signal is taken only `horizon` or more bars after the last one taken (one position
    per symbol at a time). The cost is one round trip (spread plus slippage, in pips) as a fraction of the entry price, so it
    is in the same units as the return. Commission is money per lot and there is no lot size here, so it is left out.
    """
    future_prices = prices.shift(-horizon)

    # Correctly calculate forward returns based on model type
    if model_type == "long":
        forward_returns = (future_prices - prices) / prices
    else:  # short
        forward_returns = (prices - future_prices) / prices

    # y_pred is already the binary signal; bars without a known exit (the last `horizon`) cannot be traded
    signalled = (y_pred == 1).reindex(forward_returns.index, fill_value=False) & forward_returns.notna()
    taken, last = [], None
    for pos in np.flatnonzero(signalled.to_numpy()):
        if last is None or pos - last >= horizon:
            taken.append(pos)
            last = pos
    returns, entries = forward_returns.iloc[taken], prices.iloc[taken]

    pip_size = trading_costs.get("pip_size") or trading_costs.get("pip_value") or infer_pip_size(prices)
    cost_pips = round_trip_pips(trading_costs.get("spread_pips", 1.0), trading_costs.get("slippage_pips", 0.0),
                                bool(trading_costs.get("adaptive_slippage", False)),
                                trading_costs.get("adaptive_slippage_multiplier", 1.0))
    return returns - cost_pips * pip_size / entries


def calculate_sharpe_ratio(
    y_true: pd.Series,
    y_pred: pd.Series,  # These are binary 0/1 signals
    prices: pd.Series,
    cfg: "Cfg",
    model_type: str = "long",  # The crucial new parameter
    **trading_costs
) -> float:
    """Calculates annualized Sharpe ratio for a given set of predictions using forward returns."""
    if prices is None or len(prices) != len(y_pred):
        raise ValueError(f"Prices and predictions must be same length: {len(prices)} vs {len(y_pred)}")

    horizon = cfg.prediction_horizon
    if not horizon:
        raise ValueError("cfg.prediction_horizon must be set to calculate sharpe ratio")

    pnl = net_trade_returns(y_pred, prices, horizon, model_type, **trading_costs)

    if pnl.std() == 0 or pnl.empty:
        return 0.0

    # Annualization
    timeframe_minutes = cfg.timeframe_minutes()
    if timeframe_minutes is None:
        annualization_factor = 1.0
    else:
        bars_per_year = 252 * (24 * 60 / timeframe_minutes)
        annualization_factor = np.sqrt(bars_per_year / horizon)   # one return per trade of `horizon` bars, not per bar

    sharpe = (pnl.mean() / pnl.std()) * annualization_factor
    logger.debug(f"sharpe_ratio: calculated={sharpe:.4f}, returns={len(pnl)}")
    return sharpe if np.isfinite(sharpe) else 0.0


class DynamicWeightedEnsemble:
    def __init__(self, base_models: Dict[str, MLStrategy], decay: float = 0.9, min_weight: float = 0.05):
        self.base_models = base_models
        self.decay = float(decay)
        self.min_weight = float(min_weight)
        self.model_scores: Dict[str, float] = {name: 0.5 for name in base_models}
        self.weights: Dict[str, float] = {name: 1.0 / len(base_models) for name in base_models}

    def update_weights(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        new_scores: Dict[str, float] = {}
        for name, model in self.base_models.items():
            try:
                preds = model.predict_proba(X_val)
                if hasattr(preds, "ndim") and preds.ndim == 2 and preds.shape[1] >= 2:
                    p1 = preds[:, 1]
                else:
                    # fallback: if only single-dimension, treat as probability of up
                    p1 = np.array(preds).flatten()
                auc = roc_auc_score(y_val, p1)
            except Exception as e:
                logger.warning(f"DynamicWeightedEnsemble.update_weights: model {name} failed on validation: {e}")
                auc = 0.5
            # update with exponential decay
            previous = self.model_scores.get(name, 0.5)
            updated = self.decay * previous + (1.0 - self.decay) * auc
            self.model_scores[name] = updated
            new_scores[name] = updated

        # Normalize and apply min_weight flooring
        score_arr = np.array(list(new_scores.values()), dtype=float)
        # If sum very small, avoid division by zero
        sum_scores = score_arr.sum()
        if sum_scores <= 0:
            # fallback: equal weights
            logger.warning("DynamicWeightedEnsemble: sum of scores non-positive, falling back to equal weights.")
            self.weights = {name: 1.0 / len(new_scores) for name in new_scores}
            return self.weights

        # clip and renormalize
        weights = []
        for val in score_arr:
            w = max(val, self.min_weight)
            weights.append(w)
        weights = np.array(weights, dtype=float)
        weights /= weights.sum()

        self.weights = dict(zip(new_scores.keys(), weights))
        logger.debug(f"Updated dynamic weights: {self.weights}")
        return self.weights


class Ensemble:
    def __init__(self, cfg, model_params: Optional[Dict[str, Dict]] = None, n_jobs: int = -1):
        """
        cfg: configuration object with
          - cfg.models: list of dicts, each with "name" and optional "defaults" (single values)
          - cfg.ensemble: method, weights, etc.
          - cfg.cv_samples_per_split, etc.
        model_params: optional override from tuning; keys matching model names.
        n_jobs: threads each member may use (-1 = every core).
        """
        self.cfg = cfg
        self.members: Dict[str, MLStrategy] = {}
        self.failed_members: set[str] = set()
        use_gpu = getattr(cfg, "use_gpu", False)
        self.cv_samples = getattr(cfg, "cv_samples_per_split", None) or 300
        # A label looks `horizon` bars ahead: CV folds leave that many rows between training and validation (C2). A tuned horizon
        # wins over the config, as it does when the labels are built.
        self.purge_gap = max(0, int((model_params or {}).get("prediction_horizon", getattr(cfg, "prediction_horizon", 0)) or 0))

        for m in cfg.models:
            name = m.get("name")
            if not name:
                continue
            # Initialize params with tuned parameters if available
            current_model_params = model_params.get(name, {}) if model_params else {}

            # Fall back to the config's single-value defaults for anything the tuner did not set
            # (the "tune" ranges are only for tuner.py and are never used here).
            for k, v in m.get("defaults", {}).items():
                current_model_params.setdefault(k, v)

            # Ensure all parameters are single values, not lists
            cleaned_model_params = {}
            for k, v in current_model_params.items():
                cleaned_model_params[k] = v[0] if isinstance(v, list) else v

            # GPU device hint
            if use_gpu and name.lower() in ("lgbm", "xgb"):
                cleaned_model_params["device"] = cleaned_model_params.get("device", "gpu")

            try:
                self.members[name] = MLStrategy(model=name, calibrate=True, cv_samples_per_split=self.cv_samples, n_jobs=n_jobs, **cleaned_model_params)
                self.members[name].purge_gap = self.purge_gap
            except Exception as e:
                logger.error(f"Ensemble.__init__: failed to init member {name}: {e}")
                # do not include in members
                self.failed_members.add(name)

        if not self.members:
            raise ValueError("Ensemble requires at least one valid member model")

        self.method: str = self.cfg.ensemble.get("method", "soft_vote")
        self.weights: Dict[str, float] = self.cfg.ensemble.get("weights", {k: 1.0 / len(self.members) for k in self.members})
        self.meta: Dict = self.cfg.ensemble.get("meta", {"type": "logit", "C": 1.0})
        self.flat_mode: bool = bool(self.cfg.ensemble.get("flat_mode", False))
        self.threshold_metric: str = self.cfg.ensemble.get("threshold_metric", "custom_pnl")
        self.threshold_grid: str | float | List[float] = self.cfg.ensemble.get("threshold_grid", "auto")
        self.auto_threshold: bool = self.cfg.ensemble.get("auto_threshold", False)
        defaults_cfg = getattr(self.cfg.trading_costs, "defaults", TradingCostsDefaultsCfg())
        self.trading_costs = asdict(defaults_cfg)

        self._stacker = None
        self._meta_calibrator: Optional[IsotonicRegression] = None

        self.ensemble_cv_auc_: float = 0.5
        self.member_cv_aucs_: Dict[str, float] = {}
        self.dynamic_ensemble = DynamicWeightedEnsemble(self.members)

        # placeholder for threshold (after optimization)
        self.best_threshold_: Optional[float] = None
        self.promising_thresholds_: List[float] = []  # NEW: for Thompson Sampling grids

    def save(self, path: str):
        """Saves the entire ensemble to a directory."""
        os.makedirs(path, exist_ok=True)
        logger.info(f"Saving ensemble to {path}")

        # Save each member
        for name, member in self.members.items():
            member_path = os.path.join(path, name)
            try:
                member.save_model(member_path)
            except Exception as e:
                logger.error(f"Failed to save member {name}: {e}")

        # Save ensemble metadata
        meta_path = os.path.join(path, "ensemble_meta.pkl")
        metadata = {
            "ensemble_cv_auc_": self.ensemble_cv_auc_,
            "member_cv_aucs_": self.member_cv_aucs_,
            "best_threshold_": self.best_threshold_,
            "promising_thresholds_": self.promising_thresholds_,
            "dynamic_ensemble_scores": self.dynamic_ensemble.model_scores,
            "dynamic_ensemble_weights": self.dynamic_ensemble.weights,
            "failed_members": self.failed_members,
        }
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)

        # Save stacker if it exists
        if self._stacker:
            stacker_path = os.path.join(path, "stacker.pkl")
            with open(stacker_path, "wb") as f:
                pickle.dump(self._stacker, f)

        # Save meta-calibrator if it exists
        if self._meta_calibrator:
            calibrator_path = os.path.join(path, "meta_calibrator.pkl")
            with open(calibrator_path, "wb") as f:
                pickle.dump(self._meta_calibrator, f)

        sign_model_dir(path)  # must stay last: the manifest covers every file written above

    @classmethod
    def load(cls, path: str, cfg, model_params: Optional[Dict[str, Dict]] = None) -> "Ensemble":
        """Loads an entire ensemble from a directory."""
        logger.debug(f"Loading ensemble from {path}")
        verify_model_dir(path)  # refuse to unpickle anything that is not signed with our key

        # Create a new ensemble instance to populate
        ensemble = cls(cfg, model_params=model_params)

        # Load each member
        for name, member in ensemble.members.items():
            member_path = os.path.join(path, name)
            if os.path.isdir(member_path):
                try:
                    member.load_model(member_path)
                except Exception as e:
                    logger.error(f"Failed to load member {name}: {e}")
                    ensemble.failed_members.add(name)
            else:
                logger.warning(f"Directory for member {name} not found at {member_path}")
                ensemble.failed_members.add(name)

        # Remove failed members from the active list
        for name in list(ensemble.failed_members):
            if name in ensemble.members:
                del ensemble.members[name]

        # Load ensemble metadata
        meta_path = os.path.join(path, "ensemble_meta.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)
            ensemble.ensemble_cv_auc_ = metadata.get("ensemble_cv_auc_", 0.5)
            ensemble.member_cv_aucs_ = metadata.get("member_cv_aucs_", {})
            ensemble.best_threshold_ = metadata.get("best_threshold_")
            ensemble.promising_thresholds_ = metadata.get("promising_thresholds_", [])
            if hasattr(ensemble, "dynamic_ensemble"):
                ensemble.dynamic_ensemble.model_scores = metadata.get("dynamic_ensemble_scores", {k: 0.5 for k in ensemble.members})
                ensemble.dynamic_ensemble.weights = metadata.get("dynamic_ensemble_weights", {k: 1.0 / len(ensemble.members) for k in ensemble.members})

        # Load stacker if it exists
        stacker_path = os.path.join(path, "stacker.pkl")
        if os.path.exists(stacker_path):
            with open(stacker_path, "rb") as f:
                ensemble._stacker = pickle.load(f)

        # Load meta-calibrator if it exists
        calibrator_path = os.path.join(path, "meta_calibrator.pkl")
        if os.path.exists(calibrator_path):
            with open(calibrator_path, "rb") as f:
                ensemble._meta_calibrator = pickle.load(f)

        return ensemble

    def _perform_cross_validation(
        self,
        Xc: pd.DataFrame,
        yc: pd.Series,
        prices_c: Optional[pd.Series] = None,
        model_type: str = "long"
    ) -> None:
        """Performs time-series cross-validation to evaluate the ensemble and its members."""
        n_samples = len(Xc)
        cv_split = min(5, max(2, n_samples // self.cv_samples))
        gap = int(getattr(self, "purge_gap", 0))
        for member in self.members.values():
            member.purge_gap = gap   # the member's own split (it nests inside this one) leaves the same gap
        tscv = TimeSeriesSplit(n_splits=cv_split, gap=gap)
        oof_preds: List[pd.Series] = []
        oof_true: List[pd.Series] = []
        member_cv_raw: Dict[str, List[float]] = {name: [] for name in self.members}

        for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(Xc)):
            X_tr, X_val = Xc.iloc[tr_idx], Xc.iloc[val_idx]
            y_tr, y_val = yc.iloc[tr_idx], yc.iloc[val_idx]

            if len(X_tr) < 20 or len(X_val) < 20:
                logger.debug(f"Skip fold {fold_idx} due to too small split: {len(X_tr)}/{len(X_val)}")
                continue

            # Fit each member
            fold_base_preds: Dict[str, pd.Series] = {}
            for name, model in self.members.items():
                try:
                    if not model.fit(X_tr, y_tr):
                        logger.warning(f"Ensemble.fit: member {name} skipped fitting in fold {fold_idx} due to insufficient data.")
                        continue
                    proba = model.predict_proba(X_val)
                    # ensure proba is shaped correctly
                    if hasattr(proba, "ndim") and proba.ndim == 2:
                        p1 = proba[:, 1]
                    else:
                        p1 = np.array(proba).flatten()
                    fold_base_preds[name] = pd.Series(p1, index=y_val.index, name=name)
                    # record CV score
                    auc = roc_auc_score(y_val, p1)
                    member_cv_raw[name].append(auc)
                except Exception as e:
                    logger.warning(f"Ensemble.fit: member {name} failed in fold {fold_idx}: {e}")
                    self.failed_members.add(name)

            if not fold_base_preds:
                logger.warning(f"Ensemble.fit: no valid member predictions in fold {fold_idx}")
                continue

            # Aggregate OOF
            P_val_df = pd.concat(fold_base_preds.values(), axis=1)
            # Ensure P_val_df columns correspond to member names
            P_val_df.columns = list(fold_base_preds.keys())

            oof_preds.append(P_val_df)
            oof_true.append(y_val)

            # Update dynamic weights
            self.dynamic_ensemble.update_weights(X_val, y_val)

        if not oof_preds:
            # no valid folds
            logger.warning("Ensemble.fit: no valid CV folds; skipping threshold optimization.")
            self.ensemble_cv_auc_ = np.mean([np.mean(v) for v in member_cv_raw.values() if v]) if any(member_cv_raw.values()) else 0.5
        else:
            P_oof = pd.concat(oof_preds)
            y_oof = pd.concat(oof_true)
            # Log member CV AUC
            for name, auc_list in member_cv_raw.items():
                if auc_list:
                    self.member_cv_aucs_[name] = float(np.mean(auc_list))
                else:
                    self.member_cv_aucs_[name] = 0.5
                logger.info(f"[{name}] CV AUC: {self.member_cv_aucs_[name]:.4f}")

            if self.method == "stacking":
                from sklearn.linear_model import LogisticRegression  # type: ignore

                try:
                    self._stacker = LogisticRegression(C=self.meta.get("C", 1.0), max_iter=200, random_state=42)
                    self._stacker.fit(P_oof.values, y_oof.values)
                    proba_stacked = self._stacker.predict_proba(P_oof.values)[:, 1]
                    self.ensemble_cv_auc_ = roc_auc_score(y_oof, proba_stacked)
                except Exception as e:
                    logger.error(f"Ensemble.fit: stacking failed: {e}")
                    # fallback: average of member CVs
                    self.ensemble_cv_auc_ = float(np.mean(list(self.member_cv_aucs_.values()))
                                                  if self.member_cv_aucs_ else 0.5)
                    self._stacker = None
            else:
                # using soft vote or other methods
                self.ensemble_cv_auc_ = float(np.mean(list(self.member_cv_aucs_.values())))

            logger.info(f"Ensemble CV AUC: {self.ensemble_cv_auc_:.4f}")

            # threshold optimization if prices aligned and auto_threshold is enabled
            if self.auto_threshold and prices_c is not None:
                try:
                    # use last len(y_oof) prices
                    price_segment = prices_c.iloc[-len(y_oof):]
                    # prepare average base model proba across folds
                    # compute mean predictions across folds per sample
                    # simplest: use P_oof.mean(axis=1)
                    mean_proba = P_oof.mean(axis=1)
                    self.best_threshold_, self.promising_thresholds_ = self._optimize_threshold(y_oof, mean_proba, price_segment, model_type=model_type)  # MODIFIED
                except Exception as e:
                    logger.warning(f"Ensemble.fit: threshold optimization failed: {e}")

    def feature_names(self) -> Optional[List[str]]:
        """The columns the fitted members were trained on, or None while no member has been fitted (a skipped fit leaves it
        unfitted: it predicts 0.5 for every bar)."""
        for member in self.members.values():
            names = getattr(getattr(member, "_pipe", None), "feature_names_in_", None)
            if names is not None:
                return [str(n) for n in names]
        return None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        prices: Optional[pd.Series] = None,
        cv: bool = True,
        model_type: str = "long"  # NEW
    ) -> Ensemble:
        logger.info("Ensemble.fit: start")

        if X is None or y is None:
            raise ValueError("X and y must be provided to fit()")

        # clean data
        Xc = X.replace([np.inf, -np.inf], np.nan).ffill().dropna()
        yc = y.reindex(Xc.index)
        prices_c = None
        if prices is not None:
            prices_c = prices.reindex(Xc.index)
            if len(prices_c) != len(Xc):
                logger.warning("Prices length mismatches feature/label data in Ensemble.fit; dropping mismatched indices")
                # align
                common_idx = Xc.index.intersection(prices_c.index)
                Xc = Xc.loc[common_idx]
                yc = yc.loc[common_idx]
                prices_c = prices_c.loc[common_idx]

        n_samples = len(Xc)
        min_samples = getattr(self.cfg, "min_samples_for_ensemble", 200)
        if n_samples < min_samples:
            logger.warning(f"Not enough data to fit ensemble: {n_samples} samples < {min_samples}. Skipping fit.")
            # still attempt to fit members individually
            for name, model in self.members.items():
                try:
                    model.fit(Xc, yc)
                    # record individual cv_aucs_ if available
                    self.member_cv_aucs_[name] = getattr(model, "cv_auc_", 0.5)
                except Exception as e:
                    logger.warning(f"Ensemble.fit: member {name} failed to fit small data: {e}")
                    self.failed_members.add(name)
            self.ensemble_cv_auc_ = np.mean(list(self.member_cv_aucs_.values())) if self.member_cv_aucs_ else 0.5
            return self

        if cv:
            self._perform_cross_validation(Xc, yc, prices_c, model_type)

        # After CV, refit members on all data
        for name, model in self.members.items():
            try:
                model.fit(Xc, yc)
            except Exception as e:
                logger.warning(f"Ensemble.fit: member {name} failed full data fit: {e}")
                self.failed_members.add(name)

        return self

    def _optimize_threshold(self, y_true: pd.Series, y_pred_probs: pd.Series, prices: pd.Series, model_type: str = "long") -> Tuple[Optional[float], List[float]]:  # MODIFIED
        if prices is None or len(y_true) != len(y_pred_probs) or len(prices) != len(y_pred_probs):
            logger.warning("Threshold optimization: input lengths mismatch; skipping optimization.")
            return None, []

        if self.threshold_grid == "auto":
            thresholds = np.linspace(0.3, 0.7, 21)
        elif isinstance(self.threshold_grid, (list, np.ndarray)):
            thresholds = np.array(self.threshold_grid, dtype=float)
        elif isinstance(self.threshold_grid, (float, int)):
            thresholds = np.array([float(self.threshold_grid)])
        else:
            thresholds = np.linspace(0.0, 1.0, 101)

        best_thr = 0.5
        best_score = -np.inf
        all_threshold_scores: List[Tuple[float, float]] = []  # Store (threshold, score) pairs

        for thr in thresholds:
            preds = (y_pred_probs >= thr).astype(int)
            score: float
            if self.threshold_metric == "f1":
                score = f1_score(y_true, preds)
            elif self.threshold_metric == "precision":
                score = precision_score(y_true, preds)
            elif self.threshold_metric == "recall":
                score = recall_score(y_true, preds)
            elif self.threshold_metric == "custom_pnl":
                try:
                    score = custom_pnl(y_true, preds, prices, self.cfg, model_type=model_type, **self.trading_costs)
                except Exception as e:
                    logger.warning(f"Threshold evaluation custom_pnl failed at thr={thr}: {e}")
                    continue
            elif self.threshold_metric == "sharpe_ratio":
                try:
                    score = calculate_sharpe_ratio(y_true, preds, prices, self.cfg, model_type=model_type, **self.trading_costs)  # MODIFIED
                except Exception as e:
                    logger.warning(f"Threshold evaluation sharpe_ratio failed at thr={thr}: {e}")
                    continue
            else:
                score = f1_score(y_true, preds)

            all_threshold_scores.append((thr, score))  # Store all scores

            if score > best_score:
                best_score = score
                best_thr = thr

        self.best_threshold_ = best_thr
        logger.info(f"Optimized threshold: {best_thr:.3f} (metric={self.threshold_metric}, best score={best_score:.4f})")

        # Identify promising thresholds for TS grid
        promising_thresholds: List[float] = []
        if best_score > -np.inf:  # Ensure a valid best_score was found
            promising_factor = 0.9  # Thresholds with score >= 90% of best_score
            for thr, score in all_threshold_scores:
                if score >= best_score * promising_factor:
                    promising_thresholds.append(thr)
            # Ensure the best_thr is always included and sort them
            promising_thresholds = sorted(list(set(promising_thresholds + [best_thr])))

            # Limit the number of promising thresholds to a reasonable amount (e.g., 5-7)
            # to avoid excessively large grids for Thompson Sampling
            if len(promising_thresholds) > 7:
                # If too many, take a sample around the best_thr
                best_idx = promising_thresholds.index(best_thr)
                start_idx = max(0, best_idx - 3)
                end_idx = min(len(promising_thresholds), best_idx + 4)
                promising_thresholds = promising_thresholds[start_idx:end_idx]
                logger.debug(f"Reduced promising thresholds to {len(promising_thresholds)} around best_thr.")

        return best_thr, promising_thresholds

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        if X is None:
            raise ValueError("X must be provided to predict_proba()")

        # clean data
        Xc = X.replace([np.inf, -np.inf], np.nan).ffill()
        Xc = Xc.fillna(0)  # Fill any remaining NaNs with 0

        if Xc.empty:
            logger.warning("predict_proba: empty features after cleaning, returning default 0.5")
            return pd.Series(0.5, index=X.index)

        # Collect member probabilities
        member_probs: Dict[str, pd.Series] = {}
        for name, model in self.members.items():
            try:
                proba = model.predict_proba(Xc)
                if hasattr(proba, "ndim") and proba.ndim == 2:
                    p1 = proba[:, 1]
                else:
                    p1 = np.array(proba).flatten()
                member_probs[name] = pd.Series(p1, index=Xc.index, name=name)
            except Exception as e:
                logger.warning(f"Ensemble.predict_proba: member {name} failed proba: {e}")
                # fallback: constant 0.5
                member_probs[name] = pd.Series(0.5, index=Xc.index, name=name)

        P_df = pd.concat(member_probs.values(), axis=1)
        P_df.columns = list(member_probs.keys())

        if self.method == "soft_vote":
            w = np.array([self.dynamic_ensemble.weights.get(k, 1.0 / len(self.members)) for k in P_df.columns], dtype=float)
            if w.sum() == 0:
                w = np.ones_like(w) / len(w)
            else:
                w = w / w.sum()
            # weighted average of p_up
            p_final = (P_df.values * w).sum(axis=1)
        elif self.method == "stacking" and self._stacker is not None:
            # stacker expects 2-D array
            try:
                p_final = self._stacker.predict_proba(P_df.values)[:, 1]
            except Exception as e:
                logger.error(f"Ensemble.predict_proba: stacking predict failed: {e}")
                # fallback to soft vote
                p_final = P_df.mean(axis=1).values
        else:
            # fallback: simple average
            p_final = P_df.mean(axis=1).values

        return pd.Series(p_final, index=P_df.index, name="p_up")
