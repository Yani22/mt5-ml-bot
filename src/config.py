# src/config.py
from __future__ import annotations
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

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
    simulation_volume_min: float = 0.01

@dataclass
class TcaCfg:
    enabled: bool = True
    analysis_interval_hours: float = 24.0 # How often to run TCA analysis
    lookback_days: int = 7 # How many days of data to analyze
    slippage_threshold_pips_warning: float = -0.5 # Avg slippage (pips) below this triggers a warning
    slippage_threshold_pips_critical: float = -1.0 # Avg slippage (pips) below this triggers a critical alert
    total_cost_currency_warning: float = 100.0 # Total transaction cost (currency) above this triggers a warning

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
    min_pct_change: float = 0.0001 # New: Minimum percentage change for label generation

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
    session_filter: Optional[Dict[str, str]] = None
    min_ensemble_auc: float = 0.55
    min_auc_improvement: float = 0.005
    max_drawdown_for_pruning: float = 0.70 # New: Max drawdown allowed before Optuna trial pruning
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
    state_save_interval_minutes: float = 5.0 # How often to save bot state (monitor, risk, open positions)

@dataclass
class TradingCostsDefaultsCfg:
    spread_pips: float = 0.8
    slippage_pips: float = 0.5
    commission_per_trade: float = 0.0
    lot_size: float = 0.1
    pip_value: float = 0.0001
    adaptive_slippage: bool = True
    retry_order_send: int = 3
    dynamic_slippage_enabled: bool = False # Enable dynamic slippage model in backtesting
    slippage_atr_multiplier: float = 0.1 # Multiplier for ATR in dynamic slippage calculation
    slippage_spread_multiplier: float = 0.5 # Multiplier for spread in dynamic slippage calculation
    slippage_lot_multiplier: float = 0.01 # Multiplier for order lots (log-scaled) in dynamic slippage calculation

@dataclass
class TradingCostsCfg:
    source: str = "static"
    defaults: TradingCostsDefaultsCfg = field(default_factory=TradingCostsDefaultsCfg)

@dataclass
class FetchCfg:
    initial_fetch_bars: int = 30000
    save_raw_data_locally: bool = True
    raw_data_dir: str = "data/historical_data"
    retrain_in_background: bool = True
    retrain_time_utc: Optional[str] = None  # "HH:MM" format or None
    max_consecutive_data_issues: int = 5 # Max consecutive data issues before pausing trading
    min_valid_fetches_to_recover: int = 3 # Min consecutive valid fetches to resume trading

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
    context_dim: int = 9                      # dim of context vector if contextual_enabled
    min_visits_for_exploration: int = 5        # number of visits before arm is considered "known"
    exploration_risk_mult: float = 0.5         # fraction of normal risk to use for exploratory arms
    warmstart_weight: float = 1.0              # how strongly to weight backtest priors when merging (1.0 = equal)

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

    # Dynamic Risk/TP Grids for Contextual Bandit
    dynamic_risk_base_grid: List[float] = field(default_factory=lambda: [0.005, 0.0075, 0.01])
    dynamic_risk_max_grid: List[float] = field(default_factory=lambda: [0.01, 0.015, 0.02])
    dynamic_risk_auc_floor_grid: List[float] = field(default_factory=lambda: [0.53, 0.55, 0.57])
    dynamic_risk_auc_ceiling_grid: List[float] = field(default_factory=lambda: [0.60, 0.65, 0.70])

    dynamic_tp_base_mult_grid: List[float] = field(default_factory=lambda: [1.5, 2.0, 2.5])
    dynamic_tp_max_mult_grid: List[float] = field(default_factory=lambda: [3.0, 3.5, 4.0])
    dynamic_tp_auc_floor_grid: List[float] = field(default_factory=lambda: [0.53, 0.55, 0.57])
    dynamic_tp_auc_ceiling_grid: List[float] = field(default_factory=lambda: [0.60, 0.65, 0.70])

@dataclass
class AlertsCfg:
    enabled: bool = True
    min_notification_level: str = "WARNING" # INFO, WARNING, ERROR, CRITICAL
    alert_throttle_minutes: float = 5.0 # How often to send the same alert type
    mt5_reconnect_alert_threshold: int = 3 # Number of failed MT5 reconnects before critical alert
    order_failure_alert_threshold: int = 3 # Number of consecutive order failures before critical alert

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
    optuna_pruning_interval: int = 100 # New: Interval for Optuna pruning checks
    n_jobs: int = -1 # Number of parallel jobs for tuning. -1 means all available CPU cores.
    initial_equity: float = 100.0 # New: Initial equity for backtesting
    features: FeatureCfg = field(default_factory=FeatureCfg)
    context_features: ContextFeaturesCfg = field(default_factory=ContextFeaturesCfg)
    models: List[Dict[str, Any]] = field(default_factory=list)
    ensemble: Dict[str, Any] = field(default_factory=dict)
    risk: RiskCfg = field(default_factory=RiskCfg)
    logging: Dict[str, Any] = field(default_factory=dict)
    watchdog: WatchdogCfg = field(default_factory=WatchdogCfg)
    monitoring: MonitoringCfg = field(default_factory=MonitoringCfg)
    alerts: AlertsCfg = field(default_factory=AlertsCfg) # NEW: Alerts configuration
    thompson_sampling: ThompsonSamplingCfg = field(default_factory=ThompsonSamplingCfg)
    trading_costs: TradingCostsCfg = field(default_factory=TradingCostsCfg)
    fetch: FetchCfg = field(default_factory=FetchCfg)
    min_samples_for_ensemble: int = 1000
    force_retrain_on_startup: bool = False # New: Force retraining of all models on bot startup
    retraining_window_bars: Optional[int] = None # New: Number of recent bars for rolling window retraining
    startup_logging: bool = True
    magic_number: int = 424242
    symbol_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    backtesting: BacktestingCfg = field(default_factory=BacktestingCfg)
    tca: TcaCfg = field(default_factory=TcaCfg)

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
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}

        # features may contain lists (for tuning); pick sensible defaults
        raw_features = raw.get("features", {}) or {}
        cleaned_features: Dict[str, Any] = {}
        for k, v in raw_features.items():
            if isinstance(v, list) and k != "roc_lags": # roc_lags is handled separately if it's a list of lists
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

        try:
            features_obj = FeatureCfg(**cleaned_features)
        except Exception as e:
            logger.warning(f"Invalid feature config in YAML: {e}; using defaults.")
            features_obj = FeatureCfg()

        # Parse context features
        raw_context = raw.get("context_features", {}) or {}
        try:
            mta_cfg = MtaCfg(**(raw_context.get("mta", {})))
            inter_market_cfg = InterMarketCfg(**(raw_context.get("inter_market", {})))
            price_action_cfg = PriceActionCfg(**(raw_context.get("price_action", {})))
            context_features_obj = ContextFeaturesCfg(
                mta=mta_cfg,
                inter_market=inter_market_cfg,
                price_action=price_action_cfg
            )
        except Exception as e:
            logger.warning(f"Invalid context_features config in YAML: {e}; using defaults.")
            context_features_obj = ContextFeaturesCfg()

        try:
            risk_obj = RiskCfg(**(raw.get("risk", {}) or {}))
        except Exception as e:
            logger.warning(f"Invalid risk config in YAML: {e}; using defaults.")
            risk_obj = RiskCfg()

        # parse watchdog block if present
        try:
            wd_raw = raw.get("watchdog", {}) or {}
            watchdog_obj = WatchdogCfg(**wd_raw) if wd_raw else WatchdogCfg()
        except Exception as e:
            logger.warning(f"Invalid watchdog config in YAML: {e}; using defaults.")
            watchdog_obj = WatchdogCfg()

        # parse monitoring block if present
        try:
            mon_raw = raw.get("monitoring", {}) or {}
            mon_obj = MonitoringCfg(
                lookback_days=mon_raw.get("lookback_days", 30),
                monitor_state_file=mon_raw.get("monitor_state_file", "monitor_state.json"),
                telegram_bot_token=mon_raw.get("telegram_bot_token"),
                telegram_chat_id=mon_raw.get("telegram_chat_id"),
                state_save_interval_minutes=mon_raw.get("state_save_interval_minutes", 5.0),
            )
        except Exception as e:
            logger.warning(f"Invalid monitoring config in YAML: {e}; using defaults.")
            mon_obj = MonitoringCfg()

        # parse fetch block if present (bootstrap + local caching)
        try:
            fetch_raw = raw.get("fetch", {}) or {}
            if not fetch_raw.get("retrain_time_utc"):
                logger.warning("Could not find a valid `retrain_time_utc` in config.yaml; falling back to `retrain_every_bars`.")
            fetch_obj = FetchCfg(
                initial_fetch_bars=fetch_raw.get("initial_fetch_bars", 30000),
                save_raw_data_locally=fetch_raw.get("save_raw_data_locally", True),
                raw_data_dir=fetch_raw.get("raw_data_dir", "data/historical_data"),
                retrain_in_background=fetch_raw.get("retrain_in_background", True),
                retrain_time_utc=fetch_raw.get("retrain_time_utc"),
                max_consecutive_data_issues=fetch_raw.get("max_consecutive_data_issues", 5),
                min_valid_fetches_to_recover=fetch_raw.get("min_valid_fetches_to_recover", 3),
            )
        except Exception as e:
            logger.warning(f"Invalid fetch config in YAML: {e}; using defaults.")
            fetch_obj = FetchCfg()

        # parse thompson_sampling block if present
        try:
            ts_raw = raw.get("thompson_sampling", {}) or {}
            ts_obj = ThompsonSamplingCfg(
                enabled=ts_raw.get("enabled", True),
                atr_grid=ts_raw.get("atr_grid", [0.6, 0.8, 1.0, 1.25, 1.5]),
                min_prob_grid_long=ts_raw.get("min_prob_grid_long", [0.51, 0.55, 0.60]),
                min_prob_grid_short=ts_raw.get("min_prob_grid_short", [0.51, 0.55, 0.60]),
                prior_mean=ts_raw.get("prior_mean", 0.0),
                prior_var=ts_raw.get("prior_var", 1.0),
                obs_var=ts_raw.get("obs_var", 1.0),
                decay=ts_raw.get("decay", 0.995),
                reward_normalization_factor=ts_raw.get("reward_normalization_factor", 1000.0),
                rule_rolling_window=ts_raw.get("rule_rolling_window", 100),
                vol_threshold=ts_raw.get("vol_threshold", 0.0005),
                dd_cut_multiplier=ts_raw.get("dd_cut_multiplier", 2.0),
                consec_loss_cut=ts_raw.get("consec_loss_cut", 0.2),
                state_file=ts_raw.get("state_file", "ts_risk_controller_state.json"),
                contextual_enabled=ts_raw.get("contextual_enabled", False),
                context_dim=ts_raw.get("context_dim", 9),
                min_visits_for_exploration=ts_raw.get("min_visits_for_exploration", 5),
                exploration_risk_mult=ts_raw.get("exploration_risk_mult", 0.5),
                warmstart_weight=ts_raw.get("warmstart_weight", 1.0),
                adaptive_grids_enabled=ts_raw.get("adaptive_grids_enabled", False),
                adaptation_interval_updates=ts_raw.get("adaptation_interval_updates", 500),
                adaptation_refinement_factor=ts_raw.get("adaptation_refinement_factor", 0.3),
                min_grid_size=ts_raw.get("min_grid_size", 5),
                max_grid_size=ts_raw.get("max_grid_size", 20),
                bandit_reset_enabled=ts_raw.get("bandit_reset_enabled", False),
                reset_on_drawdown_percent=ts_raw.get("reset_on_drawdown_percent", 0.20),
                reset_on_consecutive_losses=ts_raw.get("reset_on_consecutive_losses", 10),
                reset_on_low_ensemble_auc=ts_raw.get("reset_on_low_ensemble_auc", 0.52),
                reset_cooldown_hours=ts_raw.get("reset_cooldown_hours", 24.0),
                dynamic_risk_base_grid=ts_raw.get("dynamic_risk_base_grid", [0.005, 0.0075, 0.01]),
                dynamic_risk_max_grid=ts_raw.get("dynamic_risk_max_grid", [0.01, 0.015, 0.02]),
                dynamic_risk_auc_floor_grid=ts_raw.get("dynamic_risk_auc_floor_grid", [0.53, 0.55, 0.57]),
                dynamic_risk_auc_ceiling_grid=ts_raw.get("dynamic_risk_auc_ceiling_grid", [0.60, 0.65, 0.70]),
                dynamic_tp_base_mult_grid=ts_raw.get("dynamic_tp_base_mult_grid", [1.5, 2.0, 2.5]),
                dynamic_tp_max_mult_grid=ts_raw.get("dynamic_tp_max_mult_grid", [3.0, 3.5, 4.0]),
                dynamic_tp_auc_floor_grid=ts_raw.get("dynamic_tp_auc_floor_grid", [0.53, 0.55, 0.57]),
                dynamic_tp_auc_ceiling_grid=ts_raw.get("dynamic_tp_auc_ceiling_grid", [0.60, 0.65, 0.70]),
            ) if ts_raw else ThompsonSamplingCfg()
        except Exception as e:
            logger.warning(f"Invalid thompson_sampling config in YAML: {e}; using defaults.")
            ts_obj = ThompsonSamplingCfg()

        # parse trading_costs block if present
        try:
            tc_raw = raw.get("trading_costs", {}) or {}
            defaults_raw = tc_raw.get("defaults", {}) or {}
            defaults_obj = TradingCostsDefaultsCfg(
                spread_pips=defaults_raw.get("spread_pips", 0.8),
                slippage_pips=defaults_raw.get("slippage_pips", 0.5),
                commission_per_trade=defaults_raw.get("commission_per_trade", 0.0),
                lot_size=defaults_raw.get("lot_size", 0.1),
                pip_value=defaults_raw.get("pip_value", 0.0001),
                adaptive_slippage=defaults_raw.get("adaptive_slippage", True),
                retry_order_send=defaults_raw.get("retry_order_send", 3),
                dynamic_slippage_enabled=defaults_raw.get("dynamic_slippage_enabled", False),
                slippage_atr_multiplier=defaults_raw.get("slippage_atr_multiplier", 0.1),
                slippage_spread_multiplier=defaults_raw.get("slippage_spread_multiplier", 0.5),
                slippage_lot_multiplier=defaults_raw.get("slippage_lot_multiplier", 0.01),
            )
            tc_obj = TradingCostsCfg(
                source=tc_raw.get("source", "static"),
                defaults=defaults_obj
            )
        except Exception as e:
            logger.warning(f"Invalid trading_costs config in YAML: {e}; using defaults.")
            tc_obj = TradingCostsCfg()

        # parse backtesting block if present
        try:
            bt_raw = raw.get("backtesting", {}) or {}
            bt_obj = BacktestingCfg(**bt_raw) if bt_raw else BacktestingCfg()
        except Exception as e:
            logger.warning(f"Invalid backtesting config in YAML: {e}; using defaults.")
            bt_obj = BacktestingCfg()

        # parse tca block if present
        try:
            tca_raw = raw.get("tca", {}) or {}
            tca_obj = TcaCfg(**tca_raw) if tca_raw else TcaCfg()
        except Exception as e:
            logger.warning(f"Invalid TCA config in YAML: {e}; using defaults.")
            tca_obj = TcaCfg()

        # parse alerts block if present
        try:
            alerts_raw = raw.get("alerts", {}) or {}
            alerts_obj = AlertsCfg(
                enabled=alerts_raw.get("enabled", True),
                min_notification_level=alerts_raw.get("min_notification_level", "WARNING"),
                alert_throttle_minutes=alerts_raw.get("alert_throttle_minutes", 5.0),
                mt5_reconnect_alert_threshold=alerts_raw.get("mt5_reconnect_alert_threshold", 3),
                order_failure_alert_threshold=alerts_raw.get("order_failure_alert_threshold", 3),
            ) if alerts_raw else AlertsCfg()
        except Exception as e:
            logger.warning(f"Invalid alerts config in YAML: {e}; using defaults.")
            alerts_obj = AlertsCfg()

        return Cfg(
            symbols=raw.get("symbols", ["EURUSD"]),
            timeframe=raw.get("timeframe", "M5"),
            history_bars=int(raw.get("history_bars", 2000)),
            retrain_every_bars=int(raw.get("retrain_every_bars", 250)),
            prediction_horizon=int(raw.get("prediction_horizon", 6)),
            data_source=raw.get("data_source", "csv"),
            use_gpu=bool(raw.get("use_gpu", False)),
            cv_samples_per_split=int(raw.get("cv_samples_per_split", 300)),
            optuna_n_trials=int(raw.get("optuna_n_trials", 100)),
            optuna_pruning_interval=int(raw.get("optuna_pruning_interval", 100)), # New
            n_jobs=int(raw.get("n_jobs", -1)), # New
            initial_equity=float(bt_obj.initial_equity if "backtesting" in raw else raw.get("initial_equity", 100.0)),
            features=features_obj,
            context_features=context_features_obj,
            models=raw.get("models", []),
            ensemble=raw.get("ensemble", {}),
            risk=risk_obj,
            logging=raw.get("logging", {}),
            watchdog=watchdog_obj,
            monitoring=mon_obj,
            alerts=alerts_obj, # NEW
            fetch=fetch_obj,
            thompson_sampling=ts_obj,
            trading_costs=tc_obj,
            min_samples_for_ensemble=int(raw.get("min_samples_for_ensemble", 1000)),
            force_retrain_on_startup=bool(raw.get("force_retrain_on_startup", False)),
            retraining_window_bars=raw.get("retraining_window_bars", None),
            startup_logging=bool(raw.get("startup_logging", True)),
            magic_number=int(raw.get("magic_number", 424242)),
            symbol_overrides=raw.get("symbol_overrides", {}),
            backtesting=bt_obj,
            tca=tca_obj,
        )
