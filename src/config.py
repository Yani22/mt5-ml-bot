# src/config.py
from __future__ import annotations
import difflib
import yaml
from dataclasses import dataclass, field, fields
import typing
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Inputs of the contextual bandit: the vector `RiskController._context_vector` builds (a fixed list, not a function
# of the context_features flags). `thompson_sampling.context_dim` in the YAML is ignored.
CONTEXT_VECTOR_DIM = 9

# The bar sizes the data code and `timeframe_to_mt5_timeframe` know.
SUPPORTED_TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D1")


class ConfigError(ValueError):
    """config.yaml, or a file it points to (the tuned-params JSON), has something the code does not understand. The bot refuses to start on it."""


def _refuse_unknown_keys(cls, raw, name):
    _refuse_unknown_keys_in(raw, [f.name for f in fields(cls)], name)


def _refuse_unknown_keys_in(raw, known, name):
    known = list(known)
    unknown = [key for key in raw if key not in known]
    if unknown:
        hints = []
        for key in unknown:
            close = difflib.get_close_matches(str(key), known, n=1)
            hints.append(f"`{key}`" + (f" (did you mean `{close[0]}`?)" if close else ""))
        raise ConfigError(f"config.yaml: unknown key(s) in `{name}`: {', '.join(hints)}. Known keys: {', '.join(known)}")


def _type_name(hint):
    return getattr(hint, "__name__", None) or str(hint).replace("typing.", "")


def _type_ok(value, hint):
    """True when a YAML value fits the annotated type. A bool is never an int or a float; an int is a float."""
    origin = typing.get_origin(hint)
    if hint is Any:
        return True
    if origin is typing.Union:
        return any(_type_ok(value, arg) for arg in typing.get_args(hint))
    if hint is type(None):
        return value is None
    if origin in (list, List):
        args = typing.get_args(hint)
        return isinstance(value, list) and all(_type_ok(v, args[0]) for v in value) if args else isinstance(value, list)
    if origin in (dict, Dict):
        args = typing.get_args(hint)
        if not isinstance(value, dict):
            return False
        return all(_type_ok(k, args[0]) and _type_ok(v, args[1]) for k, v in value.items()) if args else True
    if hint is bool:
        return isinstance(value, bool)
    if hint is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if hint is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if hint is str:
        return isinstance(value, str)
    return True  # a nested dataclass or another type is checked where it is built


def _refuse_wrong_type(value, hint, where, source="config.yaml"):
    if not _type_ok(value, hint):
        if isinstance(value, dict):  # name the entry that is wrong, not the whole dict
            for arg in (typing.get_args(hint) if typing.get_origin(hint) is typing.Union else (hint,)):
                if typing.get_origin(arg) in (dict, Dict) and typing.get_args(arg):
                    for k, v in value.items():
                        _refuse_wrong_type(v, typing.get_args(arg)[1], f"{where}.{k}", source)
        note = ""
        if isinstance(value, str) and hint in (float, int, Optional[int]):
            try:
                float(value)
                note = " (quoted, or an exponent with no decimal point such as 1e-4: write 0.0001 or 1.0e-4)"
            except ValueError:
                pass
        raise ConfigError(f"{source}: `{where}` must be {_type_name(hint)}, got {value!r} ({type(value).__name__}){note}")


# the inner keys of the free-form dicts in RiskCfg get the type of their default; a key the default lacks is left as written
_INNER_DICTS = ("dynamic_risk", "dynamic_tp")


def _refuse_wrong_inner_types(cls, key, value, where):
    inner_types = {k: type(v) for k, v in cls().__dict__[key].items()}
    for inner_key, inner_value in value.items():
        if inner_key in inner_types:
            _refuse_wrong_type(inner_value, float if inner_types[inner_key] is float else inner_types[inner_key],
                               f"{where}.{inner_key}")


def _refuse_wrong_types(cls, raw, name):
    hints = typing.get_type_hints(cls)
    for key, value in raw.items():
        _refuse_wrong_type(value, hints[key], f"{name}.{key}")
        if cls is RiskCfg and key in _INNER_DICTS:
            _refuse_wrong_inner_types(RiskCfg, key, value, f"{name}.{key}")


# The settings the code reads per symbol (`Cfg.get_symbol_value` with a constant key), each with the type it must have.
# tests/test_config_types.py parses the source and checks this list both ways: a key read but missing here would stop a
# valid override, a key listed here that nothing reads would be an override that does nothing.
OVERRIDABLE = {
    "min_prob_long": float,
    "min_prob_short": float,
    "atr_multiplier_sl": float,
    "atr_multiplier_tp": float,
    "trailing_atr_mult": float,
    "breakeven_at_1R": bool,
    "min_ensemble_auc": float,
    "max_spread_atr": float,
    "risk_per_trade": float,
    "min_risk_reward_ratio": float,  # no config block has it; risk.py falls back to 1.2
    "dynamic_risk": Dict[str, Any],
    "dynamic_tp": Dict[str, Any],
    "atr_grid": List[float],
    "min_prob_grid_long": List[float],
    "min_prob_grid_short": List[float],
    "vol_threshold": float,
    "spread_pips": float,
    # an override may be one "HH:MM" string; the global setting is turned into a list by from_yaml
    "retrain_time_utc": typing.Union[str, List[str], None],
}


def _refuse_wrong_override_types(overrides):
    if not isinstance(overrides, dict):
        raise ConfigError(f"config.yaml: `symbol_overrides` must be a mapping of symbol to settings, got {type(overrides).__name__}")
    for symbol, settings in overrides.items():
        if not isinstance(settings, dict):
            raise ConfigError(f"config.yaml: `symbol_overrides.{symbol}` must be a mapping of settings, got {settings!r}")
        for key, value in settings.items():
            if key not in OVERRIDABLE:
                close = difflib.get_close_matches(str(key), list(OVERRIDABLE), n=1)
                raise ConfigError(f"config.yaml: `symbol_overrides.{symbol}` has `{key}`, which no code reads per symbol, so it would do "
                                  f"nothing" + (f" (did you mean `{close[0]}`?)" if close else "") +
                                  f". Settings that can be overridden: {', '.join(OVERRIDABLE)}")
            _refuse_wrong_type(value, OVERRIDABLE[key], f"symbol_overrides.{symbol}.{key}")
            if key in _INNER_DICTS:
                _refuse_wrong_inner_types(RiskCfg, key, value, f"symbol_overrides.{symbol}.{key}")


# The parameters `MLStrategy` takes from `models[].defaults` and the tuner's `models` output (`model_params.get(...)`), with their types.
# Any other key was ignored without a word. tests/test_config_blocks.py parses strategy_ml.py and checks this table against it.
MODEL_PARAMS = {
    "lgbm": {"n_estimators": int, "max_depth": int, "learning_rate": float, "subsample": float, "colsample_bytree": float,
             "min_child_samples": int},
    "xgb": {"n_estimators": int, "max_depth": int, "learning_rate": float, "subsample": float, "colsample_bytree": float},
    "rf": {"n_estimators": int, "max_depth": Optional[int], "min_samples_leaf": typing.Union[int, float]},
    "logreg": {},  # an SGD classifier with fixed settings: no parameter is read
    "sgd": {},
}


def _positive_finite(v):
    return 0 < v < float("inf")


def _fraction(v):
    return 0 < v <= 1


# What each library takes, probed against the installed lightgbm 4.7, xgboost 3.4 and scikit-learn (Linux and Wine Python had the same
# versions), with the values that a library accepts but that give a useless model (no trees, a constant model) tightened. Each check is
# written as `a < v <= b` so a NaN fails it. tests/test_config_limits.py fits each model just inside and just outside every limit.
# A parameter with no entry takes any value of its type (lightgbm's max_depth: 0 and below mean no limit).
MODEL_LIMITS = {
    ("lgbm", "n_estimators"): ("a whole number >= 1", lambda v: v >= 1),
    ("lgbm", "learning_rate"): ("a finite number > 0", _positive_finite),
    ("lgbm", "subsample"): ("a number > 0 and <= 1", _fraction),
    ("lgbm", "colsample_bytree"): ("a number > 0 and <= 1", _fraction),
    ("lgbm", "min_child_samples"): ("a whole number >= 0", lambda v: v >= 0),
    ("xgb", "n_estimators"): ("a whole number >= 1", lambda v: v >= 1),
    ("xgb", "max_depth"): ("a whole number >= 1", lambda v: v >= 1),
    ("xgb", "learning_rate"): ("a finite number > 0", _positive_finite),
    ("xgb", "subsample"): ("a number > 0 and <= 1", _fraction),
    ("xgb", "colsample_bytree"): ("a number > 0 and <= 1", _fraction),
    ("rf", "n_estimators"): ("a whole number >= 1", lambda v: v >= 1),
    ("rf", "max_depth"): ("null or a whole number >= 1", lambda v: v is None or v >= 1),
    ("rf", "min_samples_leaf"): ("a whole number >= 1 or a decimal > 0 and < 1",
                                 lambda v: (isinstance(v, int) and v >= 1) or (isinstance(v, float) and 0 < v < 1)),
}
_MODEL_ENTRY_KEYS = ("name", "defaults", "tune")
_ENSEMBLE_METHODS = ("soft_vote", "stacking")
_THRESHOLD_METRICS = ("f1", "precision", "recall", "custom_pnl", "sharpe_ratio")
_META_KEYS = ("type", "C")
_ENSEMBLE_KEYS = {
    "method": str, "weights": Dict[str, float], "meta": Dict[str, Any], "flat_mode": bool, "threshold_metric": str,
    "auto_threshold": bool, "min_ensemble_auc": float,
}
_LOGGING_KEYS = {"level": str, "to_file": bool, "rotate": typing.Union[str, int], "retention": typing.Union[str, int]}
_LOG_LEVELS = ("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL")


def _unknown_key_error(key, known, where, source="config.yaml"):
    close = difflib.get_close_matches(str(key), list(known), n=1)
    return ConfigError(f"{source}: unknown key `{key}` in `{where}`" + (f" (did you mean `{close[0]}`?)" if close else "") +
                       f". Known keys: {', '.join(known) or 'none (this model reads no parameter)'}")


def _scalar_kind(hint):
    """int, float or None (any number) for a parameter's type, with Optional unwrapped."""
    args = [a for a in typing.get_args(hint) if a is not type(None)] or [hint]
    return args[0] if len(args) == 1 else None


def _refuse_bad_range(value, hint, where, source="config.yaml"):
    kind = _scalar_kind(hint)
    ok = isinstance(value, list) and len(value) in (2, 3) and (len(value) == 2 or value[2] == "log")
    ok = ok and all(isinstance(b, (int, float)) and not isinstance(b, bool) for b in value[:2])
    if ok:
        lo, hi = value[:2]
        # the tuner takes suggest_int or suggest_float from the type of the bounds: a mixed range, or an int range for a
        # float setting (subsample [0, 1]), would silently search the wrong grid
        same = isinstance(lo, int) == isinstance(hi, int)
        fits = kind is None or (isinstance(lo, int) if kind is int else isinstance(lo, float))
        # tuner.suggest_params draws a decimal for every [lo, hi, log] range, so a whole-number setting would be saved as 287.3
        ok = same and fits and lo <= hi and not (len(value) == 3 and kind is int)
    if not ok:
        want = {int: "whole numbers", float: "decimals (write 0.6, not 0)", None: "numbers of one type"}[kind]
        log = "[low, high] (no log: the tuner draws a decimal for a log range)" if kind is int else "[low, high] or [low, high, log]"
        raise ConfigError(f"{source}: `{where}` must be {log} with {want}, low <= high; got {value!r}")


def _check_model_param(model, key, value, where, source="config.yaml", allow_range=False, allow_device=False):
    params = MODEL_PARAMS[model]
    if key == "device" and allow_device:
        return _refuse_wrong_type(value, str, f"{where}.{key}", source)
    if key not in params:
        raise _unknown_key_error(key, list(params) + (["device"] if allow_device else []), where, source)
    if allow_range and isinstance(value, list):
        _refuse_bad_range(value, params[key], f"{where}.{key}", source)
        values = value[:2]
    else:
        _refuse_wrong_type(value, params[key], f"{where}.{key}", source)
        values = [value]
    limit = MODEL_LIMITS.get((model, key))
    if limit and not all(limit[1](v) for v in values):
        raise ConfigError(f"{source}: `{where}.{key}` must be {limit[0]} for {model}, got {value!r}")


def _check_model_params(model, mapping, where, source="config.yaml", allow_range=False, allow_device=False):
    if not isinstance(mapping, dict):
        raise ConfigError(f"{source}: `{where}` must be a mapping of parameters, got {mapping!r}")
    for key, value in mapping.items():
        _check_model_param(model, key, value, where, source, allow_range, allow_device)


def _check_models(models):
    """`models` (a list of {name, defaults, tune}). Returns the list as written."""
    if models is None:
        return []
    if not isinstance(models, list):
        raise ConfigError(f"config.yaml: `models` must be a list of models, got {type(models).__name__}")
    seen = set()
    for i, entry in enumerate(models):
        where = f"models[{i}]"
        if not isinstance(entry, dict):
            raise ConfigError(f"config.yaml: `{where}` must be a mapping with `name`, got {entry!r}")
        for key in entry:
            if key not in _MODEL_ENTRY_KEYS:
                raise _unknown_key_error(key, _MODEL_ENTRY_KEYS, where)
        name = entry.get("name")
        if not isinstance(name, str) or name.lower() not in MODEL_PARAMS:
            raise ConfigError(f"config.yaml: `{where}.name` must be one of {', '.join(MODEL_PARAMS)}, got {name!r}")
        if name.lower() in seen:
            raise ConfigError(f"config.yaml: `{where}.name` {name!r} is listed twice")
        seen.add(name.lower())
        _check_model_params(name.lower(), entry.get("defaults") or {}, f"{where}.defaults", allow_device=True)
        _check_model_params(name.lower(), entry.get("tune") or {}, f"{where}.tune", allow_range=True)
    return models


def _check_ensemble(ensemble, model_names):
    if ensemble is None:
        return {}
    if not isinstance(ensemble, dict):
        raise ConfigError(f"config.yaml: `ensemble` must be a mapping of settings, got {type(ensemble).__name__}")
    for key, value in ensemble.items():
        if key == "threshold_grid":
            if not (value == "auto" or _type_ok(value, float) or _type_ok(value, List[float])):
                raise ConfigError(f"config.yaml: `ensemble.threshold_grid` must be \"auto\", a number or a list of numbers, got {value!r}")
            continue
        if key not in _ENSEMBLE_KEYS:
            raise _unknown_key_error(key, list(_ENSEMBLE_KEYS) + ["threshold_grid"], "ensemble")
        _refuse_wrong_type(value, _ENSEMBLE_KEYS[key], f"ensemble.{key}")
    meta = ensemble.get("meta") or {}
    for key, value in meta.items():
        if key not in _META_KEYS:
            raise _unknown_key_error(key, _META_KEYS, "ensemble.meta")
    if "type" in meta and meta["type"] != "logit":  # nothing reads `type`: the stacker is always a logistic regression
        raise ConfigError(f"config.yaml: `ensemble.meta.type` must be \"logit\" (the only stacker), got {meta['type']!r}")
    if "C" in meta:  # LogisticRegression refuses a bad C, and Ensemble.fit then swallows the error and averages the members instead
        _refuse_wrong_type(meta["C"], float, "ensemble.meta.C")
        if not _positive_finite(meta["C"]):
            raise ConfigError(f"config.yaml: `ensemble.meta.C` must be a finite number > 0, got {meta['C']!r}")
    if ensemble.get("method", "soft_vote") not in _ENSEMBLE_METHODS:
        raise ConfigError(f"config.yaml: `ensemble.method` must be one of {', '.join(_ENSEMBLE_METHODS)}, got {ensemble['method']!r}")
    if ensemble.get("threshold_metric", "f1") not in _THRESHOLD_METRICS:
        raise ConfigError(f"config.yaml: `ensemble.threshold_metric` must be one of {', '.join(_THRESHOLD_METRICS)}, "
                          f"got {ensemble['threshold_metric']!r}")
    for member in ensemble.get("weights") or {}:
        if member not in model_names:
            raise ConfigError(f"config.yaml: `ensemble.weights` names `{member}`, which is not a model in `models` ({', '.join(model_names)})")
    return ensemble


def _check_logging(logging_cfg):
    if logging_cfg is None:
        return {}
    if not isinstance(logging_cfg, dict):
        raise ConfigError(f"config.yaml: `logging` must be a mapping of settings, got {type(logging_cfg).__name__}")
    for key, value in logging_cfg.items():
        if key not in _LOGGING_KEYS:
            raise _unknown_key_error(key, list(_LOGGING_KEYS), "logging")
        _refuse_wrong_type(value, _LOGGING_KEYS[key], f"logging.{key}")
    if logging_cfg.get("level", "INFO") not in _LOG_LEVELS:  # case-sensitive: loguru raises on `info`
        raise ConfigError(f"config.yaml: `logging.level` must be one of {', '.join(_LOG_LEVELS)}, got {logging_cfg['level']!r}")
    return logging_cfg


_TUNED_TOP_LEVEL = {"models": Dict[str, Any], "features": Dict[str, Any], "prediction_horizon": int, "min_pct_change": float}


def check_tuned_params(params, source):
    """The tuned-params file (`optuna_params/<symbol>_best_params.json`): the keys and types the loaders read. Raises ConfigError naming `source`."""
    for key, value in params.items():
        if key not in _TUNED_TOP_LEVEL:
            raise _unknown_key_error(key, list(_TUNED_TOP_LEVEL), "the file", source)
        _refuse_wrong_type(value, _TUNED_TOP_LEVEL[key], key, source)
    hints = typing.get_type_hints(FeatureCfg)
    for key, value in (params.get("features") or {}).items():
        if key not in hints:
            raise _unknown_key_error(key, list(hints), "features", source)
        _refuse_wrong_type(value, hints[key], f"features.{key}", source)
    for name, tuned in (params.get("models") or {}).items():
        if name.lower() not in MODEL_PARAMS:
            raise ConfigError(f"{source}: `models.{name}` must be one of {', '.join(MODEL_PARAMS)}")
        _check_model_params(name.lower(), tuned, f"models.{name}", source, allow_device=True)


def _scalar(raw, key, kind, default):
    """`raw[key]` checked against `kind` (bool, int, float, str or Optional of one); `default` when absent. Replaces `bool()`/`int()`/`float()`,
    which turned "false" into True and 2.7 into 2."""
    value = raw.get(key, default)
    _refuse_wrong_type(value, kind, key)
    return value


def _block(cls, raw, name):
    """Builds the config dataclass `cls` from the YAML mapping `raw`; an absent or empty block gives the defaults.

    An unknown key raises ConfigError naming the block, the key and the closest known key. It used to log a warning and reset the
    whole block to defaults, so a typo such as `risk_per_trad` ran every risk setting on defaults without anyone noticing."""
    if not raw:
        return cls()
    if not isinstance(raw, dict):
        raise ConfigError(f"config.yaml: `{name}` must be a mapping of settings, got {type(raw).__name__}")
    _refuse_unknown_keys(cls, raw, name)
    _refuse_wrong_types(cls, raw, name)
    try:
        return cls(**raw)
    except Exception as e:
        raise ConfigError(f"config.yaml: invalid `{name}` block: {e}") from e


# Top-level keys that are not Cfg fields but are read in from_yaml
_EXTRA_TOP_LEVEL_KEYS = ("roc_lags_options", "ensemble_training")
_ENSEMBLE_TRAINING_KEYS = ("min_samples_for_ensemble",)


def _refuse_unknown_top_level_keys(raw):
    known = [f.name for f in fields(Cfg)] + list(_EXTRA_TOP_LEVEL_KEYS)
    unknown = [key for key in raw if key not in known]
    if unknown:
        hints = []
        for key in unknown:
            close = difflib.get_close_matches(str(key), known, n=1)
            hints.append(f"`{key}`" + (f" (did you mean `{close[0]}`?)" if close else ""))
        raise ConfigError(f"config.yaml: unknown top-level key(s): {', '.join(hints)}. Known keys: {', '.join(sorted(known))}")


def _same_setting(a, b, what):
    """A setting that can sit in two places must not hold two different values."""
    if a is not None and b is not None and a != b:
        raise ConfigError(f"config.yaml: {what} is set twice with different values ({a} and {b}); set it in one place")


@dataclass
class MtaCfg:
    enabled: bool = True
    timeframe: str = "H1"
    ema_period: int = 50
    rsi_period: int = 14


@dataclass
class InterMarketCfg:
    enabled: bool = True
    symbol: str = "DXY"
    roc_lags: List[int] = field(default_factory=lambda: [5, 21])


@dataclass
class BacktestingCfg:
    initial_equity: float = 10000.0
    enable_retraining: bool = True  # unused since the walk-forward backtester (kept: config.yaml sets it and an unknown key stops start-up)
    train_bars: int = 45000  # the first bars of a backtest only train; trading starts after them
    # Early stop: end a symbol's replay when its trades are demonstrably losing (see HybridBacktester._early_stop_reason). Off by default.
    early_stop: bool = False
    early_stop_min_trades: int = 100        # at least this many closed trades entered before `early_stop_until`
    early_stop_min_blocks: int = 5          # and at least this many retrain blocks reached
    early_stop_alpha: float = 0.001         # stop when the upper end of the (1 - alpha) two-sided t-interval of the mean R is below 0
    early_stop_until: Optional[str] = None  # "YYYY-MM-DD" (UTC, quoted): never check from this bar on and ignore trades entered after it; None = whole run


@dataclass
class PriceActionCfg:
    enabled: bool = True
    home_base_ma_period: int = 200
    swing_lookback: int = 50


@dataclass
class ContextFeaturesCfg:
    mta: MtaCfg = field(default_factory=MtaCfg)
    inter_market: InterMarketCfg = field(default_factory=InterMarketCfg)
    price_action: PriceActionCfg = field(default_factory=PriceActionCfg)


@dataclass
class FeatureCfg:
    rsi_period: int = 14
    ema_fast: int = 12
    ema_slow: int = 26
    window_vol: int = 20
    roc_lags: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    roc_lags_options: List[List[int]] = field(default_factory=list)
    adx_period: int = 14
    rsi_ob_level: int = 70
    rsi_os_level: int = 30
    adx_trend_thresh: int = 25
    timeframe_minutes: int = 5
    min_pct_change: float = 0.0001  # label dead zone: a long label needs the forward move above +this fraction of price (0.0001 = 0.01%, about 1.5 pips on USDJPY), a short label below -this


@dataclass
class RiskCfg:
    # default static risk values (can be overridden by YAML)
    risk_per_trade: float = 0.005
    max_positions: int = 3
    max_portfolio_risk: float = 0.03
    atr_multiplier_sl: float = 1.5
    atr_multiplier_tp: float = 2.5
    breakeven_at_1R: bool = True
    trailing_atr_mult: float = 1.0
    min_prob_long: float = 0.55
    min_prob_short: float = 0.55
    block_on_drawdown: float = 0.10
    max_spread_atr: float = 1.0  # skip entries when spread / decision-bar ATR is above this; 0 = off (symbol_overrides can set it)
    session_filter: Optional[Dict[str, str]] = None
    min_ensemble_auc: float = 0.55
    min_auc_improvement: float = 0.005  # unused since fix 65 (a retrained model replaces the old one whenever it fitted); kept so config.yaml loads
    max_drawdown_for_pruning: float = 0.70  # New: Max drawdown allowed before Optuna trial pruning
    dynamic_risk: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "base_risk": 0.005,
            "max_risk": 0.01,
            "auc_floor": 0.55,
            "auc_ceiling": 0.65,
        }
    )
    dynamic_tp: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "base_tp_mult": 2.0,
            "max_tp_mult": 3.5,
            "auc_floor": 0.55,
            "auc_ceiling": 0.65,
        }
    )


@dataclass
class WatchdogCfg:
    enabled: bool = True
    max_consecutive_losses: int = 5
    cooldown_hours: float = 1.0
    # additional optional thresholds
    daily_loss_limit: Optional[float] = None  # absolute or fraction of equity (if used)


@dataclass
class MonitoringCfg:
    lookback_days: int = 30
    monitor_state_file: str = "monitor_state.json"
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None


@dataclass
class TradingCostsDefaultsCfg:
    slippage_pips: float = 0.5
    spread_pips: float = 1.0   # backtester: one spread charged per round trip (bars are bid-only); symbol_overrides can set it per symbol
    commission_per_trade: float = 0.0
    adaptive_slippage: bool = True           # only multiplies slippage_pips by adaptive_slippage_multiplier; it does not read the spread
    retry_order_send: int = 3
    adaptive_slippage_multiplier: float = 1.0


@dataclass
class TradingCostsCfg:
    source: str = "static"
    defaults: TradingCostsDefaultsCfg = field(default_factory=TradingCostsDefaultsCfg)


@dataclass
class FetchCfg:
    initial_fetch_bars: int = 30000
    save_raw_data_locally: bool = True
    raw_data_dir: str = "data/historical_data"
    retrain_in_background: bool = True  # unused: retraining always runs in a child process (K14); kept so config.yaml still loads
    retrain_time_utc: Optional[List[str]] = None  # "HH:MM" format or None


@dataclass
class ThompsonSamplingCfg:
    enabled: bool = True
    atr_grid: List[float] = field(default_factory=lambda: [0.6, 0.8, 1.0, 1.25, 1.5])
    min_prob_grid_long: List[float] = field(default_factory=lambda: [0.51, 0.55, 0.60])
    min_prob_grid_short: List[float] = field(default_factory=lambda: [0.51, 0.55, 0.60])
    prior_mean: float = 0.0
    prior_var: float = 1.0
    obs_var: float = 1.0
    decay: float = 0.995
    reward_normalization_factor: float = 1000.0
    rule_rolling_window: int = 100
    vol_threshold: float = 0.0005
    dd_cut_multiplier: float = 2.0
    consec_loss_cut: float = 0.2
    state_file: str = "ts_risk_controller_state.json"

    # NEW fields
    contextual_enabled: bool = False           # Toggle contextual bandit
    context_dim: int = CONTEXT_VECTOR_DIM     # ignored: the size is fixed by the vector RiskController builds
    min_visits_for_exploration: int = 5        # number of visits before arm is considered "known"
    exploration_risk_mult: float = 0.5         # fraction of normal risk to use for exploratory arms
    warmstart_weight: float = 0.0              # how strongly to weight backtest priors when merging (0 = off, 1.0 = equal)

    # Adaptive Grid Configuration
    adaptive_grids_enabled: bool = False
    adaptation_interval_updates: int = 500
    adaptation_refinement_factor: float = 0.3
    min_grid_size: int = 5
    max_grid_size: int = 20

    # Bandit Reset Configuration
    bandit_reset_enabled: bool = False
    reset_on_drawdown_percent: float = 0.20
    reset_on_consecutive_losses: int = 10
    reset_on_low_ensemble_auc: float = 0.52
    reset_cooldown_hours: float = 24.0


@dataclass
class AsymmetricCompoundingCfg:
    enabled: bool = False
    win_streak_multiplier: float = 1.2  # Multiplier for risk after a winning trade
    loss_streak_divisor: float = 0.8   # Divisor for risk after a losing trade
    max_streak_effect: float = 2.0     # Max multiplier/divisor effect (e.g., 2.0 means risk can be 2x or 0.5x)
    reset_on_opposite_outcome: bool = True  # Reset streak counter if outcome changes
    lookback_trades: int = 5           # Number of recent trades to consider for streak


@dataclass
class Cfg:
    symbols: List[str] = field(default_factory=lambda: ["EURUSD#"])
    timeframe: str = "M5"
    history_bars: int = 2000
    retrain_every_bars: int = 250
    prediction_horizon: int = 6
    data_source: str = "csv"
    use_gpu: bool = False
    cv_samples_per_split: int = 300
    optuna_n_trials: int = 150
    optuna_pruning_interval: int = 100  # New: Interval for Optuna pruning checks
    n_jobs: int = -1  # Number of parallel jobs for tuning. -1 means all available CPU cores.
    initial_equity: float = 100.0  # New: Initial equity for backtesting
    features: FeatureCfg = field(default_factory=FeatureCfg)
    context_features: ContextFeaturesCfg = field(default_factory=ContextFeaturesCfg)
    models: List[Dict[str, Any]] = field(default_factory=list)
    ensemble: Dict[str, Any] = field(default_factory=dict)
    risk: RiskCfg = field(default_factory=RiskCfg)
    logging: Dict[str, Any] = field(default_factory=dict)
    watchdog: WatchdogCfg = field(default_factory=WatchdogCfg)
    monitoring: MonitoringCfg = field(default_factory=MonitoringCfg)
    thompson_sampling: ThompsonSamplingCfg = field(default_factory=ThompsonSamplingCfg)
    trading_costs: TradingCostsCfg = field(default_factory=TradingCostsCfg)
    fetch: FetchCfg = field(default_factory=FetchCfg)
    min_samples_for_ensemble: int = 1000
    force_retrain_on_startup: bool = False  # New: Force retraining of all models on bot startup
    retraining_window_bars: Optional[int] = None  # New: Number of recent bars for rolling window retraining
    startup_logging: bool = True
    magic_number: int = 424242
    symbol_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    backtesting: BacktestingCfg = field(default_factory=BacktestingCfg)
    asymmetric_compounding: AsymmetricCompoundingCfg = field(default_factory=AsymmetricCompoundingCfg)

    def __post_init__(self):
        # The contextual bandit's input size is what RiskController._context_vector builds; the YAML value is ignored
        self.thompson_sampling.context_dim = CONTEXT_VECTOR_DIM

    def timeframe_seconds(self) -> Optional[int]:
        """ Convert timeframe string like 'M5', 'H1', 'D1' to seconds.
        Returns None for unknown formats.
        """
        if not self.timeframe:
            return None
        tf = str(self.timeframe).upper().strip()
        try:
            unit = tf[0]
            value = int(tf[1:])
            if unit == "M":
                return int(value * 60)
            if unit == "H":
                return int(value * 3600)
            if unit == "D":
                return int(value * 86400)
        except Exception:
            logger.warning(f"Cfg: invalid timeframe format '{self.timeframe}'")
        return None

    def timeframe_minutes(self) -> Optional[int]:
        """ Convert timeframe string like 'M5', 'H1', 'D1' to minutes.
        Returns None for unknown formats.
        """
        seconds = self.timeframe_seconds()
        if seconds is not None:
            return seconds // 60
        return None

    def get_symbol_value(self, symbol: str, key: str, default: Any = None) -> Any:
        """
        Gets a configuration value for a symbol, checking for an override first.
        1. Looks in `symbol_overrides.<symbol>.<key>`
        2. Looks in top-level sections (risk, thompson_sampling)
        3. Returns the provided default.
        """
        # Check for a symbol-specific override first
        if symbol in self.symbol_overrides and key in self.symbol_overrides[symbol]:
            return self.symbol_overrides[symbol][key]

        # Fallback to global settings in 'risk'
        if hasattr(self.risk, key):
            return getattr(self.risk, key)

        # Fallback to global settings in 'thompson_sampling'
        if hasattr(self.thompson_sampling, key):
            return getattr(self.thompson_sampling, key)

        # Fallback to global settings in 'fetch'
        if hasattr(self.fetch, key):
            return getattr(self.fetch, key)

        # Return the default if not found anywhere
        return default

    @staticmethod
    def from_yaml(path: str) -> "Cfg":
        import platform
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        defaults = Cfg()   # the one place a missing key gets its value
        if "timeframe" in raw and raw["timeframe"] not in SUPPORTED_TIMEFRAMES:
            raise ConfigError(f"config.yaml: `timeframe` must be one of {', '.join(SUPPORTED_TIMEFRAMES)}, got {raw['timeframe']!r}")

        # Auto-switch data_source to csv on non-windows
        if platform.system() != "Windows" and raw.get("data_source") == "mt5":
            logger.warning("MT5 data source is only available on Windows. Falling back to 'csv'.")
            raw["data_source"] = "csv"

        _refuse_unknown_top_level_keys(raw)

        # features may contain lists (for tuning); pick sensible defaults
        raw_features = raw.get("features", {}) or {}
        if isinstance(raw_features, dict):
            feature_hints = typing.get_type_hints(FeatureCfg)
            for k, v in raw_features.items():  # a list is the tuner's [low, high] range: the config takes the first entry
                if isinstance(v, list) and k not in ("roc_lags", "roc_lags_options") and k in feature_hints:
                    _refuse_bad_range(v, feature_hints[k], f"features.{k}")
        cleaned_features: Dict[str, Any] = {}
        for k, v in raw_features.items():
            if isinstance(v, list) and k != "roc_lags":  # roc_lags is handled separately if it's a list of lists
                cleaned_features[k] = v[0]
            else:
                cleaned_features[k] = v

        # Handle roc_lags_options specifically
        roc_lags_options_from_yaml = raw.get("roc_lags_options", [])
        if roc_lags_options_from_yaml:
            cleaned_features["roc_lags_options"] = roc_lags_options_from_yaml
            # If roc_lags_options is present, ensure roc_lags itself is initialized, perhaps with the first option
            if "roc_lags" not in cleaned_features and roc_lags_options_from_yaml:
                cleaned_features["roc_lags"] = roc_lags_options_from_yaml[0]

        features_obj = _block(FeatureCfg, cleaned_features, "features")

        # Parse context features
        raw_context = raw.get("context_features", {}) or {}
        _refuse_unknown_keys(ContextFeaturesCfg, raw_context, "context_features")
        context_features_obj = ContextFeaturesCfg(
            mta=_block(MtaCfg, raw_context.get("mta"), "context_features.mta"),
            inter_market=_block(InterMarketCfg, raw_context.get("inter_market"), "context_features.inter_market"),
            price_action=_block(PriceActionCfg, raw_context.get("price_action"), "context_features.price_action"),
        )

        models = _check_models(raw.get("models"))
        ensemble_checked = _check_ensemble(raw.get("ensemble"), [m["name"] for m in models])
        logging_checked = _check_logging(raw.get("logging"))
        risk_obj = _block(RiskCfg, raw.get("risk"), "risk")
        # the AUC gate reads RiskCfg (get_symbol_value); `ensemble.min_ensemble_auc` in the YAML used to be ignored
        ensemble_raw = raw.get("ensemble") or {}
        if "min_ensemble_auc" in ensemble_raw:
            _same_setting(ensemble_raw["min_ensemble_auc"], (raw.get("risk") or {}).get("min_ensemble_auc"),
                          "`min_ensemble_auc` (in `ensemble` and `risk`)")
            _refuse_wrong_type(ensemble_raw["min_ensemble_auc"], float, "ensemble.min_ensemble_auc")
            risk_obj.min_ensemble_auc = float(ensemble_raw["min_ensemble_auc"])

        # `ensemble_training.min_samples_for_ensemble` was never read: the loader looked at the top level only
        ensemble_training_raw = raw.get("ensemble_training") or {}
        _refuse_unknown_keys_in(ensemble_training_raw, _ENSEMBLE_TRAINING_KEYS, "ensemble_training")
        _same_setting(ensemble_training_raw.get("min_samples_for_ensemble"), raw.get("min_samples_for_ensemble"),
                      "`min_samples_for_ensemble` (top level and `ensemble_training`)")
        min_samples_for_ensemble = _scalar(
            ensemble_training_raw, "min_samples_for_ensemble", int, _scalar(raw, "min_samples_for_ensemble", int, defaults.min_samples_for_ensemble))
        watchdog_obj = _block(WatchdogCfg, raw.get("watchdog"), "watchdog")
        mon_obj = _block(MonitoringCfg, raw.get("monitoring"), "monitoring")

        # parse fetch block if present (bootstrap + local caching)
        fetch_raw = raw.get("fetch", {}) or {}
        retrain_time = fetch_raw.get("retrain_time_utc")
        if isinstance(retrain_time, str):
            fetch_raw["retrain_time_utc"] = [retrain_time]
        if not fetch_raw.get("retrain_time_utc"):
            logger.warning("No `fetch.retrain_time_utc` in config.yaml: the live bot will never retrain (`retrain_every_bars` is only the "
                           "backtester's block length). Set a time such as \"23:55\" to retrain daily.")
        fetch_obj = _block(FetchCfg, fetch_raw, "fetch")

        ts_obj = _block(ThompsonSamplingCfg, raw.get("thompson_sampling"), "thompson_sampling")

        tc_raw = raw.get("trading_costs", {}) or {}
        _refuse_unknown_keys(TradingCostsCfg, tc_raw, "trading_costs")
        tc_obj = TradingCostsCfg(
            source=_scalar(tc_raw, "source", str, "static"),
            defaults=_block(TradingCostsDefaultsCfg, tc_raw.get("defaults"), "trading_costs.defaults"),
        )

        bt_obj = _block(BacktestingCfg, raw.get("backtesting"), "backtesting")
        ac_obj = _block(AsymmetricCompoundingCfg, raw.get("asymmetric_compounding"), "asymmetric_compounding")

        symbols = raw.get("symbols", defaults.symbols)
        _refuse_wrong_type(symbols, List[str], "symbols")
        symbol_overrides = raw.get("symbol_overrides") or {}
        _refuse_wrong_override_types(symbol_overrides)
        for key in symbol_overrides:
            if key not in symbols:
                logger.warning(f"config.yaml: `symbol_overrides` has `{key}`, which is not in `symbols` {list(symbols)}: "
                               f"its settings are not used (rename the key together with `symbols`, or delete the block).")

        return Cfg(
            symbols=symbols,
            timeframe=_scalar(raw, "timeframe", str, defaults.timeframe),
            history_bars=_scalar(raw, "history_bars", int, defaults.history_bars),
            retrain_every_bars=_scalar(raw, "retrain_every_bars", int, defaults.retrain_every_bars),
            prediction_horizon=_scalar(raw, "prediction_horizon", int, defaults.prediction_horizon),
            data_source=_scalar(raw, "data_source", str, defaults.data_source),
            use_gpu=_scalar(raw, "use_gpu", bool, defaults.use_gpu),
            cv_samples_per_split=_scalar(raw, "cv_samples_per_split", int, defaults.cv_samples_per_split),
            optuna_n_trials=_scalar(raw, "optuna_n_trials", int, defaults.optuna_n_trials),
            optuna_pruning_interval=_scalar(raw, "optuna_pruning_interval", int, defaults.optuna_pruning_interval),  # New
            n_jobs=_scalar(raw, "n_jobs", int, defaults.n_jobs),  # New
            initial_equity=float(bt_obj.initial_equity if "backtesting" in raw else _scalar(raw, "initial_equity", float, defaults.initial_equity)),
            features=features_obj,
            context_features=context_features_obj,
            models=models,
            ensemble=ensemble_checked,
            risk=risk_obj,
            logging=logging_checked,
            watchdog=watchdog_obj,
            monitoring=mon_obj,
            fetch=fetch_obj,
            thompson_sampling=ts_obj,
            trading_costs=tc_obj,
            min_samples_for_ensemble=min_samples_for_ensemble,
            force_retrain_on_startup=_scalar(raw, "force_retrain_on_startup", bool, defaults.force_retrain_on_startup),
            retraining_window_bars=_scalar(raw, "retraining_window_bars", Optional[int], defaults.retraining_window_bars),
            startup_logging=_scalar(raw, "startup_logging", bool, defaults.startup_logging),
            magic_number=_scalar(raw, "magic_number", int, defaults.magic_number),
            symbol_overrides=symbol_overrides,
            backtesting=bt_obj,
            asymmetric_compounding=ac_obj,
        )
