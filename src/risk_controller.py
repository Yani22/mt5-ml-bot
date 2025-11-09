# src/risk_controller.py
import numpy as np  # type: ignore
from loguru import logger  # type: ignore
import datetime
from collections import deque
import json
import os
from typing import List, Dict, Tuple, Optional, Any

from src.config import Cfg
from src.trade import SimPosition # For reward normalization
from src.trade_types import ClosedTrade # Import ClosedTrade
from src.linear_thompson import LinearThompson  # new

def _json_serial(obj):
    """
    JSON serializer for objects not serializable by default json code.
    Handles datetime objects by converting them to ISO format strings.
    """
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


class ThompsonBandit:
    """
    Implements a multi-armed bandit using Thompson Sampling with a Gaussian prior
    and empirical-per-arm variance estimation. This bandit is used to select
    optimal discrete parameters (arms) based on observed rewards.
    """
    def __init__(self, num_arms: int, prior_mean: float, prior_var: float, min_var: float = 1e-6):
        self.num_arms = int(num_arms)
        self.prior_mean = float(prior_mean)
        self.prior_var = float(prior_var)
        self.min_var = float(min_var)

        # Sufficient statistics
        self.counts = np.zeros(self.num_arms, dtype=float)
        self.sum_rewards = np.zeros(self.num_arms, dtype=float)
        self.sum_squared_rewards = np.zeros(self.num_arms, dtype=float)

    def _emp_mean_var(self, i: int):
        n = self.counts[i]
        if n <= 0:
            return self.prior_mean, self.prior_var
        mean = self.sum_rewards[i] / n
        if n > 1:
            var = max(self.min_var, (self.sum_squared_rewards[i] - n * mean * mean) / (n - 1))
        else:
            var = max(self.min_var, self.prior_var)
        return float(mean), float(var)

    def sample(self) -> int:
        posterior_means = np.zeros(self.num_arms)
        posterior_vars = np.zeros(self.num_arms)

        for i in range(self.num_arms):
            mean_i, emp_var = self._emp_mean_var(i)
            denom = (1.0 / self.prior_var) + (self.counts[i] / emp_var if emp_var > 0 else 0.0)
            post_var = 1.0 / denom if denom > 0 else self.prior_var
            post_mean = (self.prior_mean / self.prior_var + (self.sum_rewards[i] / emp_var if emp_var > 0 else 0.0)) * post_var
            posterior_means[i] = post_mean
            posterior_vars[i] = max(post_var, 1e-12)

        samples = np.random.normal(posterior_means, np.sqrt(posterior_vars))
        return int(np.argmax(samples))

    def update(self, arm_index: int, reward: float, decay: float = 1.0):
        if decay != 1.0:
            self.counts *= decay
            self.sum_rewards *= decay
            self.sum_squared_rewards *= decay

        self.counts[arm_index] += 1.0
        self.sum_rewards[arm_index] += reward
        self.sum_squared_rewards[arm_index] += reward * reward

    def get_state(self):
        return {
            "num_arms": self.num_arms,
            "prior_mean": self.prior_mean,
            "prior_var": self.prior_var,
            "min_var": self.min_var,
            "counts": self.counts.tolist(),
            "sum_rewards": self.sum_rewards.tolist(),
            "sum_squared_rewards": self.sum_squared_rewards.tolist(),
        }

    @classmethod
    def from_state(cls, state):
        inst = cls(state["num_arms"], state.get("prior_mean", 0.0), state.get("prior_var", 1.0), state.get("min_var", 1e-6))
        inst.counts = np.array(state.get("counts", inst.counts))
        inst.sum_rewards = np.array(state.get("sum_rewards", inst.sum_rewards))
        inst.sum_squared_rewards = np.array(state.get("sum_squared_rewards", inst.sum_squared_rewards))
        return inst

class SymbolRiskState:
    """
    Manages the Thompson Sampling and adaptive grid state for a single trading symbol.
    This includes the bandits themselves, the current grid values, performance metrics
    like peak equity and consecutive losses, and counters for grid adaptation.
    """
    def __init__(self, cfg: Cfg, atr_grid: List[float], min_prob_grid_long: List[float], min_prob_grid_short: List[float]):
        self.cfg = cfg
        ts_cfg = cfg.thompson_sampling

        # Store current dynamic grid values from arguments
        self.atr_grid_values: List[float] = atr_grid
        self.min_prob_grid_long_values: List[float] = min_prob_grid_long
        self.min_prob_grid_short_values: List[float] = min_prob_grid_short

        self.atr_bandit = ThompsonBandit(
            num_arms=len(self.atr_grid_values),
            prior_mean=ts_cfg.prior_mean,
            prior_var=ts_cfg.prior_var,
            min_var=1e-6
        )
        self.min_prob_bandit_long = ThompsonBandit(
            num_arms=len(self.min_prob_grid_long_values),
            prior_mean=ts_cfg.prior_mean,
            prior_var=ts_cfg.prior_var,
            min_var=1e-6
        )
        self.min_prob_bandit_short = ThompsonBandit(
            num_arms=len(self.min_prob_grid_short_values),
            prior_mean=ts_cfg.prior_mean,
            prior_var=ts_cfg.prior_var,
            min_var=1e-6
        )
        # optional contextual bandit (for ATR choices)
        self.contextual_bandit = None
        if getattr(ts_cfg, "contextual_enabled", False):
            # small default context dimension; RiskController will define how to build the vector
            ctx_dim = int(getattr(ts_cfg, "context_dim", 5))
            self.contextual_bandit = LinearThompson(num_arms=len(self.atr_grid_values), dim=ctx_dim, lambda_prior=1.0, noise_var=float(ts_cfg.obs_var or 1.0))

        self.peak_equity: float = cfg.initial_equity
        self.current_equity: float = cfg.initial_equity
        self.consecutive_losses: int = 0
        self.recent_returns: deque[float] = deque(maxlen=ts_cfg.rule_rolling_window)
        self.last_atr: float = 0.0

        # Track updates for adaptive grids
        self.atr_updates_since_last_adaptation: int = 0
        self.min_prob_updates_since_last_adaptation: int = 0
        self.last_reset_time: Optional[datetime.datetime] = None # NEW: To track last reset for cooldown

    def _get_bandit_and_grid(self, param_type: str) -> Tuple[ThompsonBandit | LinearThompson, List[float]]:
        if param_type == "atr":
            if self.contextual_bandit is not None:
                return self.contextual_bandit, self.atr_grid_values
            return self.atr_bandit, self.atr_grid_values
        elif param_type == "min_prob":
            return self.min_prob_bandit, self.min_prob_grid_values
        raise ValueError(f"Unknown param_type: {param_type}")

    def get_state(self):
        d = {
            "atr_bandit": self.atr_bandit.get_state(),
            "min_prob_bandit_long": self.min_prob_bandit_long.get_state(),
            "min_prob_bandit_short": self.min_prob_bandit_short.get_state(),
            "atr_grid_values": self.atr_grid_values,
            "min_prob_grid_long_values": self.min_prob_grid_long_values,
            "min_prob_grid_short_values": self.min_prob_grid_short_values,
            "peak_equity": self.peak_equity,
            "current_equity": self.current_equity,
            "consecutive_losses": self.consecutive_losses,
            "recent_returns": list(self.recent_returns),
            "last_atr": self.last_atr,
            "atr_updates_since_last_adaptation": self.atr_updates_since_last_adaptation,
            "min_prob_updates_since_last_adaptation": self.min_prob_updates_since_last_adaptation,
            "last_reset_time": self.last_reset_time.isoformat() if self.last_reset_time else None,
        }
        if self.contextual_bandit is not None:
            d["contextual_bandit"] = self.contextual_bandit.get_state()
        return d

    @classmethod
    def from_state(cls, cfg: Cfg, state: dict) -> "SymbolRiskState":
        # When loading from state, the grids saved in the state file are the source of truth.
        # Fall back to config only if the state file is missing the grid values.
        atr_grid = state.get("atr_grid_values", list(cfg.thompson_sampling.atr_grid))
        min_prob_grid_long = state.get("min_prob_grid_long_values", list(cfg.thompson_sampling.min_prob_grid_long))
        min_prob_grid_short = state.get("min_prob_grid_short_values", list(cfg.thompson_sampling.min_prob_grid_short))

        # Initialize the instance with the grids from the state file.
        inst = cls(cfg, atr_grid, min_prob_grid_long, min_prob_grid_short)

        # Now, load the bandit statistics. The number of arms will now match perfectly.
        if "atr_bandit" in state:
            inst.atr_bandit = ThompsonBandit.from_state(state["atr_bandit"])
        if "min_prob_bandit_long" in state:
            inst.min_prob_bandit_long = ThompsonBandit.from_state(state["min_prob_bandit_long"])
        if "min_prob_bandit_short" in state:
            inst.min_prob_bandit_short = ThompsonBandit.from_state(state["min_prob_bandit_short"])

        # CRITICAL FIX: When loading from a warm-start state, these keys will be missing.
        # We must fall back to the initial equity from the main config, not the instance's default.
        inst.peak_equity = state.get("peak_equity", cfg.initial_equity)
        inst.current_equity = state.get("current_equity", cfg.initial_equity)
        
        inst.consecutive_losses = state.get("consecutive_losses", 0)
        inst.recent_returns = deque(state.get("recent_returns", []), maxlen=cfg.thompson_sampling.rule_rolling_window)
        inst.last_atr = state.get("last_atr", 0.0)
        inst.atr_updates_since_last_adaptation = state.get("atr_updates_since_last_adaptation", 0)
        inst.min_prob_updates_since_last_adaptation = state.get("min_prob_updates_since_last_adaptation", 0)
        last_reset_time_str = state.get("last_reset_time")
        inst.last_reset_time = datetime.datetime.fromisoformat(last_reset_time_str) if last_reset_time_str else None
        
        if "contextual_bandit" in state and getattr(cfg.thompson_sampling, "contextual_enabled", False):
            # Re-initialize contextual bandit with the correct number of arms from the loaded grid
            ctx_dim = int(getattr(cfg.thompson_sampling, "context_dim", 9))
            # Ensure the contextual bandit is created with the correct number of arms
            inst.contextual_bandit = LinearThompson(num_arms=len(inst.atr_grid_values), dim=ctx_dim, lambda_prior=1.0, noise_var=float(cfg.thompson_sampling.obs_var or 1.0))
            # Now load the state into the correctly sized bandit
            inst.contextual_bandit = LinearThompson.from_state(state["contextual_bandit"])
            
        return inst

class RiskController:
    """
    Manages Thompson Sampling bandits and rule-based scaling for multiple symbols.
    It orchestrates the selection of optimal risk parameters, handles dynamic
    grid adaptation, and triggers bandit resets based on performance metrics.
    """
    def __init__(self, cfg: Cfg, notifier=None):
        self.cfg = cfg
        self.notifier = notifier
        self.symbol_states: Dict[str, SymbolRiskState] = {}
        for sym in cfg.symbols:
            # Use the new helper to get symbol-specific grids, with fallbacks to global config
            atr_grid = cfg.get_symbol_value(sym, 'atr_grid', cfg.thompson_sampling.atr_grid)
            min_prob_grid_long = cfg.get_symbol_value(sym, 'min_prob_grid_long', cfg.thompson_sampling.min_prob_grid_long)
            min_prob_grid_short = cfg.get_symbol_value(sym, 'min_prob_grid_short', cfg.thompson_sampling.min_prob_grid_short)
            
            self.symbol_states[sym] = SymbolRiskState(cfg, atr_grid, min_prob_grid_long, min_prob_grid_short)
        
        self.state_file = cfg.thompson_sampling.state_file
        self.last_daily_retrain_date: Dict[str, Optional[datetime.date]] = {sym: None for sym in cfg.symbols}
        self.bar_counters: Dict[str, int] = {sym: 0 for sym in cfg.symbols}

    def update_last_daily_retrain_date(self, symbol: str, date: datetime.date):
        self.last_daily_retrain_date[symbol] = date

    def _calculate_rule_scale(self, symbol: str, context: Dict[str, Any]) -> float:
        """
        Computes a rule-based scaling factor (between 0 and 1, inclusive) based on
        various performance and market context variables such as volatility, drawdown,
        and consecutive losses. This scale is applied to certain risk parameters
        to dynamically adjust risk exposure.

        Args:
            symbol: The trading symbol for which to calculate the rule scale.
            context: A dictionary containing current market and performance context.

        Returns:
            A float representing the calculated rule scale, typically between 0.01 and 1.0.
        """
        sym_state = self.symbol_states[symbol]
        ts_cfg = self.cfg.thompson_sampling

        # Extract context variables
        vol = context.get("vol", sym_state.last_atr) # Use last_atr if current vol not provided
        equity = context.get("equity", sym_state.current_equity)
        peak_equity = context.get("peak_equity", sym_state.peak_equity)
        max_drawdown = 1.0 - (equity / peak_equity) if peak_equity is not None and peak_equity > 0 else 0.0
        consecutive_losses = sym_state.consecutive_losses

        rule_scale = 1.0

        # 1. Inverse Volatility Scale
        if vol > 0 and ts_cfg.vol_threshold > 0: # Avoid division by zero
            inverse_vol_scale = min(1.0, ts_cfg.vol_threshold / vol + 0.5) # Example scaling
            rule_scale *= inverse_vol_scale

        # 2. Drawdown Scale
        if max_drawdown > 0 and ts_cfg.dd_cut_multiplier > 0:
            drawdown_scale = max(0.1, 1.0 - ts_cfg.dd_cut_multiplier * max_drawdown)
            rule_scale *= drawdown_scale

        # 3. Consecutive Loss Scale
        if consecutive_losses > 0 and ts_cfg.consec_loss_cut > 0:
            consec_scale = max(0.1, 1.0 - ts_cfg.consec_loss_cut * consecutive_losses / 5.0) # Divide by 5 for example
            rule_scale *= consec_scale
        
        # Ensure rule_scale is within (0, 1]
        rule_scale = np.clip(rule_scale, 0.01, 1.0) # Min scale of 0.01 to avoid zeroing out

        logger.debug(f"[{symbol}] Rule Scale: {rule_scale:.2f} (Vol:{vol:.5f}, DD:{max_drawdown:.2%}, CL:{consecutive_losses})")
        return float(rule_scale)
        """
        Computes a rule-based scaling factor (between 0 and 1, inclusive) based on
        various performance and market context variables such as volatility, drawdown,
        and consecutive losses. This scale is applied to certain risk parameters
        to dynamically adjust risk exposure.

        Args:
            symbol: The trading symbol for which to calculate the rule scale.
            context: A dictionary containing current market and performance context.

        Returns:
            A float representing the calculated rule scale, typically between 0.01 and 1.0.
        """
        sym_state = self.symbol_states[symbol]
        ts_cfg = self.cfg.thompson_sampling

        # Extract context variables
        vol = context.get("vol", sym_state.last_atr) # Use last_atr if current vol not provided
        equity = context.get("equity", sym_state.current_equity)
        peak_equity = context.get("peak_equity", sym_state.peak_equity)
        max_drawdown = 1.0 - (equity / peak_equity) if peak_equity is not None and peak_equity > 0 else 0.0
        consecutive_losses = sym_state.consecutive_losses

        rule_scale = 1.0

        # 1. Inverse Volatility Scale
        if vol > 0 and ts_cfg.vol_threshold > 0: # Avoid division by zero
            inverse_vol_scale = min(1.0, ts_cfg.vol_threshold / vol + 0.5) # Example scaling
            rule_scale *= inverse_vol_scale

        # 2. Drawdown Scale
        if max_drawdown > 0 and ts_cfg.dd_cut_multiplier > 0:
            drawdown_scale = max(0.1, 1.0 - ts_cfg.dd_cut_multiplier * max_drawdown)
            rule_scale *= drawdown_scale

        # 3. Consecutive Loss Scale
        if consecutive_losses > 0 and ts_cfg.consec_loss_cut > 0:
            consec_scale = max(0.1, 1.0 - ts_cfg.consec_loss_cut * consecutive_losses / 5.0) # Divide by 5 for example
            rule_scale *= consec_scale
        
        # Ensure rule_scale is within (0, 1]
        rule_scale = np.clip(rule_scale, 0.01, 1.0) # Min scale of 0.01 to avoid zeroing out

        logger.debug(f"[{symbol}] Rule Scale: {rule_scale:.2f} (Vol:{vol:.5f}, DD:{max_drawdown:.2%}, CL:{consecutive_losses})")
        return float(rule_scale)

    def get_params(self, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Samples discrete choices via Thompson Sampling (or contextual bandit if enabled)
        and applies rule-based scaling to determine the final risk parameters.

        Args:
            symbol: The trading symbol for which to get parameters.
            context: A dictionary containing current market and performance context.

        Returns:
            A dictionary with chosen parameters (e.g., atr_multiplier_sl, min_prob_long)
            and their discrete indices, along with rule_scale and exploration metadata.
        """
        if not self.cfg.thompson_sampling.enabled:
            # If TS is disabled, return default risk parameters from cfg, respecting symbol overrides
            return {
                "atr_multiplier_sl": self.cfg.get_symbol_value(symbol, 'atr_multiplier_sl', 1.5),
                "atr_multiplier_tp": self.cfg.get_symbol_value(symbol, 'atr_multiplier_tp', 2.5),
                "trailing_atr_mult": self.cfg.get_symbol_value(symbol, 'trailing_atr_mult', 1.0),
                "min_prob_long": self.cfg.get_symbol_value(symbol, 'min_prob_long', 0.55),
                "min_prob_short": self.cfg.get_symbol_value(symbol, 'min_prob_short', 0.55),
                "atr_idx": -1, # Indicate no TS choice
                "min_prob_long_idx": -1,
                "min_prob_short_idx": -1,
            }

        sym_state = self.symbol_states[symbol]
        ts_cfg = self.cfg.thompson_sampling

        # Check and trigger reset if conditions are met
        self._check_and_trigger_reset(symbol, context, ensemble_auc=context.get("ensemble_auc", 0.5))

        # 1. Sample discrete choices (possibly contextual)
        atr_idx = None
        if getattr(self.cfg.thompson_sampling, "contextual_enabled", False) and sym_state.contextual_bandit is not None:
            # build a context vector from available context dict: vol, equity, peak_equity, ensemble_auc, adx, macd_diff, volatility_10, dist_from_ema_200
            # normalize vol by vol_threshold to keep scales reasonable
            vol = float(context.get("vol", sym_state.last_atr or 0.0))
            auc = float(context.get("ensemble_auc", 0.5))
            equity = float(context.get("equity", sym_state.current_equity or self.cfg.initial_equity))
            peak = float(context.get("peak_equity") or sym_state.peak_equity or self.cfg.initial_equity)
            drawdown = 1.0 - (equity / peak) if peak > 0 else 0.0
            # time-of-day features (hour sin/cos)
            now = datetime.datetime.utcnow()
            hour = now.hour + now.minute / 60.0
            hour_sin = np.sin(2 * np.pi * hour / 24.0)
            hour_cos = np.cos(2 * np.pi * hour / 24.0)
            vol_scale = float(self.cfg.thompson_sampling.vol_threshold or 1e-6)

            # New context features
            adx = float(context.get("adx", 0.0))
            macd_diff = float(context.get("macd_diff", 0.0))
            volatility_10 = float(context.get("volatility_10", 0.0))
            dist_from_ema_200 = float(context.get("dist_from_ema_200", 0.0))

            x = np.array([
                vol / max(vol_scale, 1e-9),
                auc,
                drawdown,
                hour_sin,
                hour_cos,
                adx / 100.0, # Normalize ADX (typically 0-100)
                macd_diff * 1000.0, # Scale macd_diff for better feature representation
                volatility_10 * 100.0, # Scale volatility
                dist_from_ema_200 * 100.0, # Scale distance
            ], dtype=float)
            # ensure dimension matches bandit's dimension; if not, pad/truncate
            ctx_dim = sym_state.contextual_bandit.dim
            if len(x) < ctx_dim:
                x = np.concatenate([x, np.zeros(ctx_dim - len(x))])
            elif len(x) > ctx_dim:
                x = x[:ctx_dim]
            atr_idx = sym_state.contextual_bandit.sample_arm(x)
        else:
            atr_idx = sym_state.atr_bandit.sample()

        min_prob_long_idx = sym_state.min_prob_bandit_long.sample()
        min_prob_short_idx = sym_state.min_prob_bandit_short.sample()

        # 2. Apply rule-based scaling
        rule_scale = self._calculate_rule_scale(symbol, context)

        # Apply rule_scale to ATR-related parameters
        atr_choice = sym_state.atr_grid_values[atr_idx] * rule_scale
        # For min_prob, scaling might be different or not applied directly
        min_prob_long_choice = sym_state.min_prob_grid_long_values[min_prob_long_idx]
        min_prob_short_choice = sym_state.min_prob_grid_short_values[min_prob_short_idx]

        # Use the symbol-specific value for the trailing stop multiplier
        base_trailing_mult = self.cfg.get_symbol_value(symbol, 'trailing_atr_mult', 1.0)
        trailing_atr_mult_choice = base_trailing_mult * rule_scale

        logger.debug(f"[{symbol}] TS Params: ATR={atr_choice:.2f} (idx:{atr_idx}), MinProbLong={min_prob_long_choice:.2f} (idx:{min_prob_long_idx}), MinProbShort={min_prob_short_choice:.2f} (idx:{min_prob_short_idx}), RuleScale={rule_scale:.2f}")

        # Exploration safety: if arm is under-visited, apply exploration risk multiplier
        ts_cfg = self.cfg.thompson_sampling
        is_exploratory = False
        exploration_risk_mult = 1.0
        if hasattr(sym_state.atr_bandit, "counts"):
            if sym_state.atr_bandit.counts[atr_idx] < ts_cfg.min_visits_for_exploration:
                is_exploratory = True
                exploration_risk_mult = float(ts_cfg.exploration_risk_mult)

        # Return chosen params plus exploratory metadata
        return {
            "atr_multiplier_sl": atr_choice,
            "atr_multiplier_tp": atr_choice,
            "trailing_atr_mult": trailing_atr_mult_choice,
            "min_prob_long": min_prob_long_choice,
            "min_prob_short": min_prob_short_choice,
            "atr_idx": atr_idx,
            "min_prob_long_idx": min_prob_long_idx,
            "min_prob_short_idx": min_prob_short_idx,
            "rule_scale": rule_scale,
            "is_exploratory": is_exploratory,
            "exploration_risk_mult": exploration_risk_mult,
            "context_vector": x.tolist() if 'x' in locals() else None,
        }

    def update(self, trade: ClosedTrade):
        if not self.cfg.thompson_sampling.enabled:
            return

        symbol = trade.symbol
        if symbol not in self.symbol_states:
            logger.warning(f"[{symbol}] Attempted to update non-existent symbol state. Skipping update.")
            return

        sym_state = self.symbol_states[symbol]
        reward = trade.pnl / self.cfg.thompson_sampling.reward_normalization_factor # Normalize PnL

        # Update ATR bandit
        if trade.atr_idx is not None and trade.atr_idx != -1: # -1 indicates no TS choice for this parameter
            if sym_state.contextual_bandit is not None and getattr(self.cfg.thompson_sampling, "contextual_enabled", False):
                if trade.context_vector is not None:
                    context_vector = np.array(trade.context_vector)
                    if context_vector.shape[0] == sym_state.contextual_bandit.dim:
                        sym_state.contextual_bandit.update(trade.atr_idx, context_vector, reward)
                    else:
                        logger.warning(f"[{symbol}] Context vector dimension mismatch. Bandit dim: {sym_state.contextual_bandit.dim}, trade context dim: {context_vector.shape[0]}.")
                else:
                    logger.warning(f"[{symbol}] Context vector not found in ClosedTrade. Cannot update contextual bandit.")
            else:
                if trade.atr_idx < sym_state.atr_bandit.num_arms:
                    sym_state.atr_bandit.update(trade.atr_idx, reward)
                else:
                    logger.warning(f"[{symbol}] Invalid atr_idx {trade.atr_idx} for atr_bandit with {sym_state.atr_bandit.num_arms} arms.")

        # Update min_prob_long bandit
        if trade.min_prob_long_idx is not None and trade.min_prob_long_idx != -1:
            if trade.min_prob_long_idx < sym_state.min_prob_bandit_long.num_arms:
                sym_state.min_prob_bandit_long.update(trade.min_prob_long_idx, reward)
            else:
                logger.warning(f"[{symbol}] Invalid min_prob_long_idx {trade.min_prob_long_idx} for min_prob_bandit_long with {sym_state.min_prob_bandit_long.num_arms} arms.")

        # Update min_prob_short bandit
        if trade.min_prob_short_idx is not None and trade.min_prob_short_idx != -1:
            if trade.min_prob_short_idx < sym_state.min_prob_bandit_short.num_arms:
                sym_state.min_prob_bandit_short.update(trade.min_prob_short_idx, reward)
            else:
                logger.warning(f"[{symbol}] Invalid min_prob_short_idx {trade.min_prob_short_idx} for min_prob_bandit_short with {sym_state.min_prob_bandit_short.num_arms} arms.")
                    
        # Increment adaptation counters
        if trade.atr_idx is not None and trade.atr_idx != -1:
            sym_state.atr_updates_since_last_adaptation += 1
        if trade.min_prob_long_idx is not None and trade.min_prob_long_idx != -1:
            sym_state.min_prob_updates_since_last_adaptation += 1
        if trade.min_prob_short_idx is not None and trade.min_prob_short_idx != -1:
            sym_state.min_prob_updates_since_last_adaptation += 1
        
        logger.debug(f"[{symbol}] Thompson Sampling bandits updated for trade {trade.ticket} with reward {reward:.4f}.")
                    
        # Check and trigger grid adaptation after updating bandits
        self._check_and_trigger_adaptation(symbol)        
        
    def save_state(self, open_positions_cache: dict | None = None):
        state_path = self.cfg.thompson_sampling.state_file
        try:
            symbol_states_data = {}
            for sym, sym_state in self.symbol_states.items():
                symbol_states_data[sym] = sym_state.get_state()

            state = {
                "symbol_states": symbol_states_data,
                "open_positions_cache": open_positions_cache or {},
                "last_daily_retrain_date": {sym: date.isoformat() if date else None for sym, date in self.last_daily_retrain_date.items()},
                "bar_counters": self.bar_counters
            }
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=4, default=_json_serial)
            logger.debug(f"RiskController state saved to {state_path}")
        except Exception as e:
            logger.error(f"Failed to save RiskController state: {e}")
            if self.notifier: self.notifier.send_message(f"<b>ERROR:</b> Failed to save RiskController state: {e}", level="ERROR")
    def load_state(self) -> dict:
        state_path = self.cfg.thompson_sampling.state_file
        if not os.path.exists(state_path):
            logger.info(f"No existing RiskController state file found at {state_path}. Starting fresh.")
            return {}

        try:
            with open(state_path, 'r') as f:
                state = json.load(f)

            symbol_states_data = state.get("symbol_states", {})
            for sym, sym_state_data in symbol_states_data.items():
                if sym in self.cfg.symbols: # Only load for active symbols
                    self.symbol_states[sym] = SymbolRiskState.from_state(self.cfg, sym_state_data)
                else:
                    logger.warning(f"State found for inactive symbol {sym}. Skipping load.")

            # Load last_daily_retrain_date and bar_counters
            loaded_dates = state.get("last_daily_retrain_date", {})
            for sym, date_str in loaded_dates.items():
                if sym in self.cfg.symbols:
                    self.last_daily_retrain_date[sym] = datetime.date.fromisoformat(date_str) if date_str else None
            self.bar_counters = state.get("bar_counters", {sym: 0 for sym in self.cfg.symbols})

            logger.info(f"RiskController state loaded from {state_path}")
            # Return the open positions cache if it exists
            return state.get("open_positions_cache", {})
        except Exception as e:
            logger.error(f"Failed to load RiskController state from {state_path}: {e}")
            if self.notifier: self.notifier.send_message(f"<b>ERROR:</b> Failed to load RiskController state from {state_path}: {e}", level="ERROR")
            return {}


    def diagnostics(self) -> Dict[str, Any]:
        """
        Returns a dictionary of diagnostic information about the RiskController's state.
        """
        all_diagnostics = {}
        for sym, sym_state in self.symbol_states.items():
            sym_diag = {
                "peak_equity": sym_state.peak_equity,
                "current_equity": sym_state.current_equity,
                "consecutive_losses": sym_state.consecutive_losses,
                "recent_returns_count": len(sym_state.recent_returns),
                "atr_bandit_counts": sym_state.atr_bandit.counts.tolist(),
                "atr_bandit_sum_rewards": sym_state.atr_bandit.sum_rewards.tolist(),
                "min_prob_bandit_long_counts": sym_state.min_prob_bandit_long.counts.tolist(),
                "min_prob_bandit_long_sum_rewards": sym_state.min_prob_bandit_long.sum_rewards.tolist(),
                "min_prob_bandit_short_counts": sym_state.min_prob_bandit_short.counts.tolist(),
                "min_prob_bandit_short_sum_rewards": sym_state.min_prob_bandit_short.sum_rewards.tolist(),
                "atr_grid_values": sym_state.atr_grid_values,
                "min_prob_grid_long_values": sym_state.min_prob_grid_long_values,
                "min_prob_grid_short_values": sym_state.min_prob_grid_short_values,
            }
            all_diagnostics[sym] = sym_diag
        return all_diagnostics

    def _reset_bandit_state(self, symbol: str, current_time: datetime.datetime):
        """
        Resets the bandit states and dynamic grids for a given symbol to their initial configurations.
        This is typically triggered due to significant performance degradation or market shifts.
        It clears all learned bandit statistics and resets the grids to the values defined in the config.

        Args:
            symbol: The trading symbol for which to reset the bandit state.
            current_time: The current UTC time, used to set the last_reset_time for cooldown.
        """
        sym_state = self.symbol_states[symbol]
        ts_cfg = self.cfg.thompson_sampling

        logger.warning(f"[{symbol}] Triggering bandit reset due to performance degradation or market shift.")
        if self.notifier: self.notifier.send_message(f"<b>RISK ALERT:</b> [{symbol}] Bandit reset triggered!", level="WARNING")
        # Reset ThompsonBandits to initial state
        sym_state.atr_bandit = ThompsonBandit(
            num_arms=len(ts_cfg.atr_grid),
            prior_mean=ts_cfg.prior_mean,
            prior_var=ts_cfg.prior_var,
            min_var=1e-6
        )
        sym_state.min_prob_bandit_long = ThompsonBandit(
            num_arms=len(ts_cfg.min_prob_grid_long),
            prior_mean=ts_cfg.prior_mean,
            prior_var=ts_cfg.prior_var,
            min_var=1e-6
        )
        sym_state.min_prob_bandit_short = ThompsonBandit(
            num_arms=len(ts_cfg.min_prob_grid_short),
            prior_mean=ts_cfg.prior_mean,
            prior_var=ts_cfg.prior_var,
            min_var=1e-6
        )
        logger.debug(f"[{symbol}] MinProb Long and Short bandits re-initialized.")

        # Reset contextual bandit if enabled
        if getattr(ts_cfg, "contextual_enabled", False) and sym_state.contextual_bandit is not None:
            ctx_dim = int(getattr(ts_cfg, "context_dim", 9))
            sym_state.contextual_bandit = LinearThompson(num_arms=len(ts_cfg.atr_grid), dim=ctx_dim, lambda_prior=1.0, noise_var=float(ts_cfg.obs_var or 1.0))

        # Reset dynamic grids to initial config values
        sym_state.atr_grid_values = list(ts_cfg.atr_grid)
        sym_state.min_prob_grid_long_values = list(ts_cfg.min_prob_grid_long)
        sym_state.min_prob_grid_short_values = list(ts_cfg.min_prob_grid_short)

        # Reset adaptation counters
        sym_state.atr_updates_since_last_adaptation = 0
        sym_state.min_prob_updates_since_last_adaptation = 0

        # Reset performance metrics for the symbol
        sym_state.peak_equity = sym_state.current_equity # Reset peak to current equity
        sym_state.consecutive_losses = 0
        sym_state.recent_returns.clear()

        # Set last reset time for cooldown
        sym_state.last_reset_time = current_time
        logger.info(f"[{symbol}] Bandit reset complete. Cooldown until {current_time + datetime.timedelta(hours=ts_cfg.reset_cooldown_hours)}")

    def _check_and_trigger_reset(self, symbol: str, context: Dict[str, Any], ensemble_auc: float):
        """
        Checks various conditions (drawdown, consecutive losses, low ensemble AUC) to determine
        if a bandit reset is necessary for a given symbol. If a reset is triggered and not
        in a cooldown period, it calls `_reset_bandit_state`.

        Args:
            symbol: The trading symbol to check for reset conditions.
            context: A dictionary containing current market and performance context.
            ensemble_auc: The current AUC score of the ensemble model for the symbol.
        """
        ts_cfg = self.cfg.thompson_sampling
        if not ts_cfg.bandit_reset_enabled:
            return

        sym_state = self.symbol_states[symbol]
        current_time = datetime.datetime.utcnow()

        # Check cooldown
        if sym_state.last_reset_time:
            cooldown_end_time = sym_state.last_reset_time + datetime.timedelta(hours=ts_cfg.reset_cooldown_hours)
            if current_time < cooldown_end_time:
                logger.debug(f"[{symbol}] Bandit reset in cooldown until {cooldown_end_time}")
                return

        # Check triggers
        reset_triggered = False
        trigger_reason = ""

        # 1. Drawdown trigger
        equity = context.get("equity", sym_state.current_equity)
        peak_equity = context.get("peak_equity", sym_state.peak_equity)
        if peak_equity is not None and peak_equity > 0:
            current_drawdown = 1.0 - (equity / peak_equity)
            if current_drawdown >= ts_cfg.reset_on_drawdown_percent:
                reset_triggered = True
                trigger_reason = f"Drawdown ({current_drawdown:.2%}) exceeded {ts_cfg.reset_on_drawdown_percent:.2%}"

        # 2. Consecutive losses trigger
        if not reset_triggered and sym_state.consecutive_losses >= ts_cfg.reset_on_consecutive_losses:
            reset_triggered = True
            trigger_reason = f"Consecutive losses ({sym_state.consecutive_losses}) exceeded {ts_cfg.reset_on_consecutive_losses}"

        # 3. Low ensemble AUC trigger
        if not reset_triggered and ensemble_auc < ts_cfg.reset_on_low_ensemble_auc:
            reset_triggered = True
            trigger_reason = f"Ensemble AUC ({ensemble_auc:.4f}) below {ts_cfg.reset_on_low_ensemble_auc:.4f}"
        
        if reset_triggered:
            logger.warning(f"[{symbol}] Bandit reset triggered: {trigger_reason}")
            self._reset_bandit_state(symbol, current_time)

    def _check_and_trigger_adaptation(self, symbol: str):
        """
        Checks if adaptive grid conditions are met for a given symbol and triggers
        grid refinement and bandit state transfer if necessary.
        """
        sym_state = self.symbol_states[symbol]
        ts_cfg = self.cfg.thompson_sampling

        if not ts_cfg.adaptive_grids_enabled:
            return

        # --- ATR Grid Adaptation ---
        if sym_state.atr_updates_since_last_adaptation >= ts_cfg.adaptation_interval_updates:
            logger.info(f"[{symbol}] Triggering ATR grid adaptation.")
            # Determine the best arm for ATR
            if sym_state.contextual_bandit is not None:
                # For contextual bandit, we need to find the arm with the highest estimated value
                # This is a simplification; a more robust approach might involve simulating contexts
                # For now, we'll use the arm with the highest mean reward from the underlying LinearThompson
                best_arm_index = np.argmax(sym_state.contextual_bandit.b[:, 0] / np.diag(sym_state.contextual_bandit.A[:, :, 0])) # Simplified
            else:
                best_arm_index = np.argmax(sym_state.atr_bandit.sum_rewards / (sym_state.atr_bandit.counts + 1e-6)) # Avoid div by zero

            old_atr_grid = sym_state.atr_grid_values
            new_atr_grid = self._refine_grid(
                current_grid=old_atr_grid,
                best_arm_index=int(best_arm_index),
                refinement_factor=ts_cfg.adaptation_refinement_factor,
                min_grid_size=ts_cfg.min_grid_size,
                max_grid_size=ts_cfg.max_grid_size
            )

            if new_atr_grid != old_atr_grid:
                logger.info(f"[{symbol}] ATR grid adapted. Old: {old_atr_grid}, New: {new_atr_grid}")
                if sym_state.contextual_bandit is not None:
                    old_bandit = sym_state.contextual_bandit
                    new_bandit = LinearThompson(
                        num_arms=len(new_atr_grid),
                        dim=old_bandit.dim,
                        lambda_prior=old_bandit.lambda_prior,
                        noise_var=old_bandit.noise_var
                    )
                    self._transfer_contextual_bandit_state(old_bandit, old_atr_grid, new_bandit, new_atr_grid)
                    sym_state.contextual_bandit = new_bandit
                else:
                    old_bandit = sym_state.atr_bandit
                    new_bandit = ThompsonBandit(
                        num_arms=len(new_atr_grid),
                        prior_mean=old_bandit.prior_mean,
                        prior_var=old_bandit.prior_var,
                        min_var=old_bandit.min_var
                    )
                    self._transfer_bandit_state(old_bandit, old_atr_grid, new_bandit, new_atr_grid)
                    sym_state.atr_bandit = new_bandit
                sym_state.atr_grid_values = new_atr_grid
                sym_state.atr_updates_since_last_adaptation = 0 # Reset counter
            else:
                logger.debug(f"[{symbol}] ATR grid adaptation resulted in no change.")
                sym_state.atr_updates_since_last_adaptation = 0 # Reset counter even if no change

        # --- Min Prob Long Grid Adaptation ---
        if sym_state.min_prob_updates_since_last_adaptation >= ts_cfg.adaptation_interval_updates:
            logger.info(f"[{symbol}] Triggering Min Prob Long grid adaptation.")
            best_arm_index = np.argmax(sym_state.min_prob_bandit_long.sum_rewards / (sym_state.min_prob_bandit_long.counts + 1e-6))

            old_min_prob_grid_long = sym_state.min_prob_grid_long_values
            new_min_prob_grid_long = self._refine_grid(
                current_grid=old_min_prob_grid_long,
                best_arm_index=int(best_arm_index),
                refinement_factor=ts_cfg.adaptation_refinement_factor,
                min_grid_size=ts_cfg.min_grid_size,
                max_grid_size=ts_cfg.max_grid_size
            )

            if new_min_prob_grid_long != old_min_prob_grid_long:
                logger.info(f"[{symbol}] Min Prob Long grid adapted. Old: {old_min_prob_grid_long}, New: {new_min_prob_grid_long}")
                old_bandit = sym_state.min_prob_bandit_long
                new_bandit = ThompsonBandit(
                    num_arms=len(new_min_prob_grid_long),
                    prior_mean=old_bandit.prior_mean,
                    prior_var=old_bandit.prior_var,
                    min_var=old_bandit.min_var
                )
                self._transfer_bandit_state(old_bandit, old_min_prob_grid_long, new_bandit, new_min_prob_grid_long)
                sym_state.min_prob_bandit_long = new_bandit
                sym_state.min_prob_grid_long_values = new_min_prob_grid_long
                sym_state.min_prob_updates_since_last_adaptation = 0
            else:
                logger.debug(f"[{symbol}] Min Prob Long grid adaptation resulted in no change.")
                sym_state.min_prob_updates_since_last_adaptation = 0

        # --- Min Prob Short Grid Adaptation ---
        if sym_state.min_prob_updates_since_last_adaptation >= ts_cfg.adaptation_interval_updates:
            logger.info(f"[{symbol}] Triggering Min Prob Short grid adaptation.")
            best_arm_index = np.argmax(sym_state.min_prob_bandit_short.sum_rewards / (sym_state.min_prob_bandit_short.counts + 1e-6))

            old_min_prob_grid_short = sym_state.min_prob_grid_short_values
            new_min_prob_grid_short = self._refine_grid(
                current_grid=old_min_prob_grid_short,
                best_arm_index=int(best_arm_index),
                refinement_factor=ts_cfg.adaptation_refinement_factor,
                min_grid_size=ts_cfg.min_grid_size,
                max_grid_size=ts_cfg.max_grid_size
            )

            if new_min_prob_grid_short != old_min_prob_grid_short:
                logger.info(f"[{symbol}] Min Prob Short grid adapted. Old: {old_min_prob_grid_short}, New: {new_min_prob_grid_short}")
                old_bandit = sym_state.min_prob_bandit_short
                new_bandit = ThompsonBandit(
                    num_arms=len(new_min_prob_grid_short),
                    prior_mean=old_bandit.prior_mean,
                    prior_var=old_bandit.prior_var,
                    min_var=old_bandit.min_var
                )
                self._transfer_bandit_state(old_bandit, old_min_prob_grid_short, new_bandit, new_min_prob_grid_short)
                sym_state.min_prob_bandit_short = new_bandit
                sym_state.min_prob_grid_short_values = new_min_prob_grid_short
                sym_state.min_prob_updates_since_last_adaptation = 0
            else:
                logger.debug(f"[{symbol}] Min Prob Short grid adaptation resulted in no change.")
                sym_state.min_prob_updates_since_last_adaptation = 0

    @staticmethod
    def _transfer_bandit_state(old_bandit: ThompsonBandit, old_grid: List[float], new_bandit: ThompsonBandit, new_grid: List[float]):
        """
        Transfers learned statistics from an old bandit to a new bandit with a refined grid.
        This is crucial when the grid changes (e.g., during adaptation) to preserve learning.
        It maps the statistics of old arm values to the closest corresponding new arm values.

        Args:
            old_bandit: The ThompsonBandit instance with the old grid.
            old_grid: The list of values for the old grid.
            new_bandit: The newly initialized ThompsonBandit instance with the new grid.
            new_grid: The list of values for the new grid.
        """
        if not old_grid or not new_grid:
            return

        # For each old arm, find the closest new arm and transfer its statistics
        for old_idx, old_val in enumerate(old_grid):
            if old_bandit.counts[old_idx] > 0:
                # Find the index of the closest value in the new grid
                new_idx = int(np.argmin(np.abs(np.array(new_grid) - old_val)))

                # Transfer statistics. If multiple old arms map to the same new arm,
                # their statistics will be summed up. This is a reasonable heuristic.
                new_bandit.counts[new_idx] += old_bandit.counts[old_idx]
                new_bandit.sum_rewards[new_idx] += old_bandit.sum_rewards[old_idx]
                new_bandit.sum_squared_rewards[new_idx] += old_bandit.sum_squared_rewards[old_idx]

        logger.debug(f"Transferred bandit state from {len(old_grid)} to {len(new_grid)} arms.")

    @staticmethod
    def _transfer_contextual_bandit_state(old_bandit: LinearThompson, old_grid: List[float], new_bandit: LinearThompson, new_grid: List[float]):
        """
        Transfers learned statistics (A and b matrices) from an old LinearThompson bandit
        to a new LinearThompson bandit with a refined grid.
        Maps old arm values to the closest new arm values and aggregates their statistics.
        """
        if not old_grid or not new_grid:
            return

        # For each old arm, find the closest new arm and transfer its statistics
        for old_idx, old_val in enumerate(old_grid):
            # Find the index of the closest value in the new grid
            new_idx = int(np.argmin(np.abs(np.array(new_grid) - old_val)))
            # Transfer A and b matrices. If multiple old arms map to the same new arm,
            # their statistics will be summed up.
            new_bandit.A[new_idx] += old_bandit.A[old_idx]
            new_bandit.b[new_idx] += old_bandit.b[old_idx]
        
        logger.debug(f"Transferred contextual bandit state from {len(old_grid)} to {len(new_grid)} arms.")

    @staticmethod
    def _refine_grid(current_grid: List[float], best_arm_index: int, refinement_factor: float, min_grid_size: int, max_grid_size: int) -> List[float]:
        """
        Refines a given grid by narrowing the range around the best-performing arm.
        The new grid will be centered around the best_val and its range will be
        `refinement_factor` times the original range around the best_val.

        Args:
            current_grid: The current list of grid values.
            best_arm_index: The index of the best-performing arm in the current_grid.
            refinement_factor: A float between 0 and 1, indicating how much to narrow the grid.
            min_grid_size: The minimum allowed size for the refined grid.
            max_grid_size: The maximum allowed size for the refined grid.

        Returns:
            A new, refined list of grid values.
        """
        if not (0 < refinement_factor < 1):
            logger.warning(f"Invalid refinement_factor: {refinement_factor}. Must be between 0 and 1. Using 0.5.")
            refinement_factor = 0.5

        # Handle cases where refinement is not possible or meaningful
        if not current_grid or len(current_grid) <= 1 or len(current_grid) < min_grid_size:
            # If the grid is empty, has only one element, or is already smaller than min_grid_size,
            # return it as is, as refinement is not applicable or would violate constraints.
            return current_grid

        best_val = current_grid[best_arm_index]

        # Determine the interval around the best value
        # If best_arm_index is 0, use the interval to the next arm
        # If best_arm_index is last, use the interval to the previous arm
        # Otherwise, use the smaller of the two adjacent intervals
        lower_bound = current_grid[best_arm_index - 1] if best_arm_index > 0 else best_val - (current_grid[1] - current_grid[0])
        upper_bound = current_grid[best_arm_index + 1] if best_arm_index < len(current_grid) - 1 else best_val + (current_grid[-1] - current_grid[-2])

        # Ensure bounds are sensible if at edges
        if best_arm_index == 0:
            lower_bound = best_val - (current_grid[1] - best_val) * 2 # Extend a bit below
        if best_arm_index == len(current_grid) - 1:
            upper_bound = best_val + (best_val - current_grid[-2]) * 2 # Extend a bit above

        # Calculate the new, narrower range
        current_range = upper_bound - lower_bound
        new_range = current_range * refinement_factor
        
        # Center the new range around the best_val
        new_lower = best_val - new_range / 2
        new_upper = best_val + new_range / 2

        # Ensure new_lower and new_upper don't go beyond the original min/max of the full grid
        original_min = min(current_grid)
        original_max = max(current_grid)
        new_lower = max(new_lower, original_min)
        new_upper = min(new_upper, original_max)

        # If the new range is too small or invalid, return current grid
        if new_upper <= new_lower:
            logger.debug(f"Refinement resulted in invalid range [{new_lower}, {new_upper}]. Returning current grid.")
            return current_grid

        # Generate new grid points
        num_new_points = min(max_grid_size, max(min_grid_size, len(current_grid) + 2)) # Add a few points, but respect max_grid_size
        new_grid = np.linspace(new_lower, new_upper, num_new_points).tolist()
        new_grid = sorted(list(set(new_grid + [best_val]))) # Ensure best_val is always in the new grid
        
        # Ensure grid size constraints are met
        if len(new_grid) < min_grid_size:
            # If after refinement, grid is too small, try to expand it slightly or just return original
            logger.debug(f"Refined grid size {len(new_grid)} is less than min_grid_size {min_grid_size}. Returning current grid.")
            return current_grid
        if len(new_grid) > max_grid_size:
            # If too large, resample to max_grid_size
            new_grid = np.linspace(min(new_grid), max(new_grid), max_grid_size).tolist()

        logger.info(f"Grid refined from {len(current_grid)} to {len(new_grid)} arms. Old best: {best_val:.4f}. New range: [{min(new_grid):.4f}, {max(new_grid):.4f}]")
        return new_grid
