"""Settings that load without error but change nothing, found by reading the code (AUDIT 'Second-pass review'). `Cfg` still accepts
them: an unknown key stops start-up, so a key the code no longer reads has to stay accepted. The start-up summary lists them."""
from src.config import Cfg


def ignored_setting_warnings(cfg: Cfg) -> list:
    out = []
    if cfg.force_retrain_on_startup:
        out.append("force_retrain_on_startup is set but nothing reads it: models are retrained at `retrain_time_utc` only. "
                   "Run trainer.py to retrain now.")
    ensemble = cfg.ensemble or {}
    if ensemble.get("weights"):
        out.append("ensemble.weights is set but never used: the members are weighted by their running validation AUC.")
    if ensemble.get("flat_mode"):
        out.append("ensemble.flat_mode is set but never read.")
    dynamic_tp = cfg.risk.dynamic_tp or {}
    if dynamic_tp.get("enabled"):
        out.append("risk.dynamic_tp is enabled but never applied: the take-profit is always the stop multiple times "
                   "atr_multiplier_tp / atr_multiplier_sl, whatever the AUC.")
    defaults = Cfg()
    unread = (
        ("thompson_sampling.reward_normalization_factor", cfg.thompson_sampling.reward_normalization_factor,
         defaults.thompson_sampling.reward_normalization_factor),
        ("asymmetric_compounding.lookback_trades", cfg.asymmetric_compounding.lookback_trades,
         defaults.asymmetric_compounding.lookback_trades),
        ("optuna_pruning_interval", cfg.optuna_pruning_interval, defaults.optuna_pruning_interval),
        ("risk.max_drawdown_for_pruning", cfg.risk.max_drawdown_for_pruning, defaults.risk.max_drawdown_for_pruning),
        ("trading_costs.source", cfg.trading_costs.source, defaults.trading_costs.source),
    )
    for name, value, default in unread:
        if value != default:
            out.append(f"{name} is set to {value!r} but nothing reads it.")
    return out
