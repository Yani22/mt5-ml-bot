# backtester.py
from __future__ import annotations
import pandas as pd  # type: ignore
from loguru import logger  # type: ignore
import os
import datetime
import quantstats as qs  # type: ignore
import optuna  # type: ignore
import numpy as np  # type: ignore

from src.config import Cfg
from src.features import FeatureCfg
from src.risk import RiskManager
from src.costs import round_trip_pips
from src.decision import choose_direction
from src.utils import setup_logging, load_optuna_params, log_symbol_specific_configs
from src.backtest_data import load_backtest_frame
from src.backtest_fills import bar_spread, entry_price, exit_hit, next_stop
from src.backtest_models import WalkForwardModels, make_fit_fn
from src.backtest_symbols import BacktestSymbolClient
from src.trade import SimPosition
from src.risk_controller import RiskController
from src.trade_types import ClosedTrade  # NEW: Import ClosedTrade for backtester
import threading


class HybridBacktester:
    """Adaptive hybrid backtester mirroring main_hybrid_adaptive.py logic."""

    def __init__(self, cfg: Cfg, broker_client=None):
        self.logged_low_confidence = set()
        self.logged_skips = set()
        self.cfg = cfg
        log_symbol_specific_configs(self.cfg)  # NEW
        self.equity = cfg.backtesting.initial_equity
        self.initial_equity = cfg.backtesting.initial_equity  # Store initial equity for drawdown pruning
        self.signals = 0  # trade signals that reached position sizing
        self.skipped_for_size = 0  # ...of which sizing rejected (lot below broker minimum or risk caps)
        self.consecutive_losses = 0  # NEW: Efficiently track consecutive losses
        self.pending = {}  # symbol -> the order decided on the previous bar, filled at this bar's open
        self.blocked_by_auc = 0  # signals the `min_ensemble_auc` gate refused (a run with 0 trades says why)
        self.skipped_for_spread = 0  # entries refused because the fill bar's spread was above `max_spread_atr` x ATR (live does the same)
        self.positions: list[SimPosition] = []
        self.equity_curve = []
        lock = threading.Lock()
        if broker_client is None:
            broker_client = BacktestSymbolClient()  # no terminal: derive the symbol facts (FX with USD only)
        self.risk_manager = RiskManager(cfg, broker_client, lock)
        self.bar_counters = {sym: 0 for sym in cfg.symbols}
        self.risk_controller = RiskController(cfg)  # Instantiate RiskController
        # no `load_state()`: a saved state already knows which arms won on these bars, so a backtest starts with empty bandits
        self.ts_param_history = []  # To store Thompson Sampling parameter evolution
        self.save_state_every_bars = getattr(cfg, "save_ts_state_every_bars", 500)
        ts_ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        symbol_str = self.cfg.symbols[0].replace('#', '')  # Use the exact symbol string from config, sanitized
        self.backtest_ts_state_file = f"results/ts_risk_controller_state_backtest_{symbol_str}_{ts_ts}.json"
        self.ts_history_csv = f"results/ts_param_evolution_backtest_{symbol_str}_{ts_ts}.csv"
        os.makedirs("results", exist_ok=True)

        logger.info(f"Initializing backtester with starting equity: {self.equity}")

    def _spread_price(self, sym: str, raw_points) -> float:
        """A bar's spread in price units: the file's points x the symbol's point, or the `trading_costs` spread (pips) when the bar
        has none (0, NaN, no file)."""
        info = self.risk_manager.mt5_client.symbol_info(sym)
        points_per_pip = 10 if getattr(info, "digits", None) in (3, 5) else 1
        costs = self.cfg.trading_costs.defaults
        fallback = float(self.cfg.get_symbol_value(sym, "spread_pips", costs.spread_pips)) * points_per_pip * info.point
        try:
            raw_price = float(raw_points) * info.point
        except (TypeError, ValueError):
            raw_price = None
        return bar_spread(raw_price, fallback)

    def _manage_stops(self, sym: str, row: pd.Series, atr: float):
        """Breakeven at +1R and the ATR trail on the closed bar, by the rule live uses (`next_stop`). Called after the bar's exits, so
        a moved stop first applies on the next bar."""
        breakeven = bool(self.cfg.get_symbol_value(sym, "breakeven_at_1R", True))
        trailing = float(self.cfg.get_symbol_value(sym, "trailing_atr_mult", 0.0) or 0.0)
        if not (breakeven or trailing > 0):
            return
        spread = self._spread_price(sym, row.get("spread"))
        for pos in [p for p in self.positions if p.symbol == sym and p.status == "open"]:
            placed_mult = abs(pos.entry_price - pos.initial_sl) / pos.atr if pos.atr else None
            pos.sl = next_stop(pos.direction, pos.entry_price, pos.sl, pos.atr, placed_mult, row["close"], spread, atr, breakeven, trailing)

    def _open_pending(self, sym: str, row: pd.Series):
        """Fills the order queued on the previous bar at this bar's open (a long at the ask), sizing it on the real stop distance."""
        order = self.pending.pop(sym, None)
        if not order:
            return
        direction, atr, auc, params = order["direction"], order["atr"], order["auc"], order["params"]
        spread = self._spread_price(sym, row.get("spread"))
        cap = float(self.cfg.get_symbol_value(sym, "max_spread_atr", 1.0) or 0.0)    # the live guard (fix 20): no entry in a wide spread
        if cap > 0 and not spread <= cap * atr + 1e-12:
            self.skipped_for_spread += 1
            logger.info(f"[{sym}][{row.name}] Entry skipped: spread {spread:.6f} is above {cap:.2f} x the decision ATR {atr:.6f}.")
            return
        price = entry_price(direction, row["open"], spread)
        sl, tp = self.risk_manager.stop_targets(price, atr, direction, auc, sym, sl_mult=params["atr_multiplier_sl"], tp_mult=params["atr_multiplier_tp"])
        self.signals += 1
        if sl <= 0 or tp <= 0:
            self.skipped_for_size += 1
            return
        lots, effective_risk = self.risk_manager.position_size(
            self.equity, atr, auc, total_open_risk=self._total_open_risk(), symbol=sym,
            exploration_mult=params.get("exploration_risk_mult", 1.0), ac_multiplier=params.get("ac_multiplier", 1.0),
            sl_distance=abs(price - sl))
        if lots <= 0:
            self.skipped_for_size += 1
            logger.info(f"[{sym}] Trade skipped due to risk limits or position size zero.")
            return
        self.positions.append(SimPosition(
            sym, direction, lots, price, sl, tp, row.name, atr, auc, effective_risk, entry_equity=self.equity,
            atr_idx=params["atr_idx"], min_prob_long_idx=params.get("min_prob_long_idx", -1),
            min_prob_short_idx=params.get("min_prob_short_idx", -1), trade_context=params.get("context_vector")))
        logger.info(f"[{sym}][{row.name}] Opened {direction} at {price:.5f}. Lots: {lots:.2f}, SL: {sl:.5f}, TP: {tp:.5f}, AUC: {auc:.4f}")

    def _open_stop_money(self, pos) -> float:
        """Money (account currency) at the stop as it was placed, the live convention (fixes 8 and 54): breakeven and trailing
        move `pos.sl` but not what the trade risked."""
        return self.risk_manager.move_value(pos.symbol, abs(pos.entry_price - pos.initial_sl), pos.lots)

    def _total_open_risk(self) -> float:
        """Sum of `_open_stop_money` over the open positions, what the portfolio cap counts."""
        return sum(self._open_stop_money(p) for p in self.positions if p.status == "open")

    def _close_costs(self, sym: str, lots: float) -> float:
        """Money (account currency) charged when the position closes: slippage (pips) and commission. The spread is not here: a long
        is filled at the ask and a short is bought back at the ask, so each bar's spread is already in the fill prices. There is no
        `risk.transaction_cost_pips`."""
        costs = self.cfg.trading_costs.defaults
        pips = round_trip_pips(0.0, costs.slippage_pips, costs.adaptive_slippage, costs.adaptive_slippage_multiplier)
        return self.risk_manager.pips_to_money(sym, pips, lots) + costs.commission_per_trade * lots

    def _update_positions(self, sym, row):
        """Check open positions for SL/TP, calculate PnL, and update equity using sequential reconstruction."""
        closed_trades_this_cycle = []

        # This loop identifies trades that close on the current bar
        for pos in [p for p in self.positions if p.symbol == sym and p.status == "open"]:
            hit = exit_hit(pos.direction, pos.sl, pos.tp, row["open"], row["high"], row["low"], self._spread_price(sym, row.get("spread")))
            exit_reason = None
            if hit:
                price, exit_reason = hit

            if exit_reason:
                gross_pnl = self.risk_manager.move_value(sym, (price - pos.entry_price) if pos.direction == "long" else (pos.entry_price - price), pos.lots)

                transaction_cost = self._close_costs(sym, pos.lots)
                net_pnl = gross_pnl - transaction_cost

                # Determine the exit equity for this specific trade
                exit_equity = self.equity + net_pnl

                # Create ClosedTrade object from SimPosition
                closed_trade = ClosedTrade(
                    ticket=pos.ticket,  # SimPosition already has a unique ticket-like ID
                    symbol=pos.symbol,
                    direction=pos.direction,
                    lots=pos.lots,
                    entry_price=pos.entry_price,
                    exit_price=price,
                    entry_time=pos.entry_time,
                    exit_time=row.name,  # Use bar timestamp as exit_time
                    pnl=net_pnl,
                    risk_fraction=pos.risk_fraction,
                    atr=pos.atr,  # ATR at entry
                    atr_idx=pos.atr_idx,
                    min_prob_long_idx=pos.min_prob_long_idx,
                    min_prob_short_idx=pos.min_prob_short_idx,
                    entry_auc=pos.entry_auc,
                    entry_equity=pos.entry_equity,
                    exit_equity=exit_equity,  # Pass the correct exit equity
                    adx=getattr(pos, 'adx', 0.0),
                    macd_diff=getattr(pos, 'macd_diff', 0.0),
                    volatility_10=getattr(pos, 'volatility_10', 0.0),
                    dist_from_ema_200=getattr(pos, 'dist_from_ema_200', 0.0),
                    context_vector=getattr(pos, 'trade_context', None),  # NEW: Pass stored context vector
                    risk_amount=self._open_stop_money(pos),
                    sl_atr_mult=(abs(pos.entry_price - pos.initial_sl) / pos.atr) if pos.atr else None
                )

                # Update the bandit and the symbol's state with the ClosedTrade object
                self.risk_controller.update(closed_trade)

                # Now, update the backtester's main equity
                self.equity += net_pnl

                # Close the position object (SimPosition)
                pos.close(price, row.name.to_pydatetime().replace(tzinfo=datetime.timezone.utc), net_pnl, self.equity)

                logger.info(
                    f"[{sym}] Closed {pos.direction} position at {pos.exit_price:.5f}. "
                    f"Entry: {pos.entry_price:.5f}, PnL: {pos.pnl:.2f}, Final Equity: {self.equity:.2f}"
                )

    def _force_close_open_positions(self, sym: str, bars: pd.DataFrame):
        """Close any positions left open for `sym` at the last bar: a long at the bid (the close), a short at the ask."""
        logger.info(f"Closing any remaining open positions for {sym}...")
        for pos in [p for p in self.positions if p.symbol == sym and p.status == "open"]:
            last_row = bars.iloc[-1]
            last_price = last_row["close"] + (self._spread_price(sym, last_row.get("spread")) if pos.direction == "short" else 0.0)
            gross_pnl = self.risk_manager.move_value(sym, (last_price - pos.entry_price) if pos.direction == "long" else (pos.entry_price - last_price), pos.lots)
            transaction_cost = self._close_costs(sym, pos.lots)
            net_pnl = gross_pnl - transaction_cost

            pos.close(last_price, last_row.name, net_pnl, self.equity + net_pnl)
            self.equity += net_pnl
            logger.info(
                f"[{pos.symbol}] Force-closed open {pos.direction} position at final price {last_price:.5f}. "
                f"PnL: {net_pnl:.2f}, Final Equity: {self.equity:.2f}"
            )

    def _check_and_prune(self, trial: optuna.Trial, i: int):
        """Checks if the trial should be pruned."""
        current_returns = pd.Series([eq for _, eq in self.equity_curve]).pct_change().dropna()
        if not current_returns.empty:
            intermediate_sharpe = 0.0
            if current_returns.std() != 0:
                timeframe_minutes = self.cfg.timeframe_minutes()
                if timeframe_minutes is not None:
                    annualization_factor = np.sqrt(252 * (24 * 60 / timeframe_minutes))
                    intermediate_sharpe = current_returns.mean() / current_returns.std() * annualization_factor
            trial.report(intermediate_sharpe, i)
            if trial.should_prune():
                raise optuna.TrialPruned()

    def _process_bar(self, sym: str, frame, models, trial: optuna.Trial | None = None, pruning_interval: int = 0):
        """Walk-forward replay of one symbol. A decision on bar i (features known at its close) queues an order that fills at the
        open of bar i + 1; exits are checked on every bar's high and low; the models for bar i were fitted on past bars only."""
        bars, X = frame.bars, frame.X
        risk_mgr = self.risk_manager
        client = risk_mgr.mt5_client

        logger.info(f"Processing {len(X)} bars for {sym}...")
        for i in range(models.start, len(X)):
            bar_time = X.index[i]
            current_row = bars.iloc[i]
            self.bar_counters[sym] += 1
            atr = X["atr_14"].iloc[i]
            last_features = X.iloc[[i]]
            if hasattr(client, "update_price"):
                client.update_price(sym, current_row["close"])  # the tick value of a USD-base pair moves with the price

            # An order decided on the previous bar fills at this bar's open; this bar's range can already stop it out
            self._open_pending(sym, current_row)
            self._update_positions(sym, current_row)
            self._manage_stops(sym, current_row, atr)  # a moved stop applies from the next bar
            if i == len(X) - 1:
                break  # no next bar to fill a decision on

            # --- Drawdown and Cooldown Check ---
            # Clear a cooldown that has just expired BEFORE the checks below (live `should_trade` does the same), so a drawdown or
            # a loss streak that still holds starts a new cooldown on this bar instead of letting this bar decide.
            risk_mgr.cooldown_active(now=bar_time.to_pydatetime().replace(tzinfo=datetime.timezone.utc))
            risk_mgr._update_equity_peak(self.equity)
            if risk_mgr._drawdown_exceeded(self.equity):
                if risk_mgr.cooldown_until is None:  # Only trigger if not already in cooldown
                    logger.warning(f"[{sym}][{bar_time}] Drawdown threshold exceeded. Triggering cooldown.")
                    now_utc = bar_time.to_pydatetime().replace(tzinfo=datetime.timezone.utc)
                    risk_mgr._trigger_cooldown(now=now_utc)

            # --- Consecutive Loss Check ---
            if risk_mgr.watchdog_cfg.enabled:
                max_losses = getattr(risk_mgr.watchdog_cfg, "max_consecutive_losses", None)
                if max_losses is not None and max_losses > 0:
                    # Get consecutive losses from the symbol's state within RiskController
                    consecutive_losses = self.risk_controller.symbol_states[sym].consecutive_losses
                    if consecutive_losses >= max_losses:
                        if risk_mgr.cooldown_until is None:
                            logger.warning(f"[{sym}][{bar_time}] Watchdog: consecutive losses {consecutive_losses} >= threshold {max_losses}. Triggering cooldown.")
                            now_utc = bar_time.to_pydatetime().replace(tzinfo=datetime.timezone.utc)
                            risk_mgr._trigger_cooldown(now=now_utc)

            now_utc = bar_time.to_pydatetime().replace(tzinfo=datetime.timezone.utc)
            if risk_mgr.cooldown_active(now=now_utc):
                logger.info(f"[{sym}][{bar_time}] Trading blocked: watchdog cooldown active.")
                self.equity_curve.append((bar_time, self.equity))
                # --- Pruning Check (if in tuning mode) ---
                if trial and pruning_interval > 0 and (i % pruning_interval == 0) and self.cfg.symbols.index(sym) == 0:
                    current_returns = pd.Series([eq for _, eq in self.equity_curve]).pct_change().dropna()
                    if not current_returns.empty:
                        intermediate_sharpe = 0.0
                        if current_returns.std() != 0:
                            timeframe_minutes = self.cfg.timeframe_minutes()
                            if timeframe_minutes is not None:
                                annualization_factor = np.sqrt(252 * (24 * 60 / timeframe_minutes))
                                intermediate_sharpe = current_returns.mean() / current_returns.std() * annualization_factor
                        trial.report(intermediate_sharpe, i)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                continue

            prob_long, prob_short, auc_long, auc_short = models.probs(i)

            # Get dynamic risk parameters from RiskController
            context = {
                "vol": atr,
                "price": float(current_row["close"]),
                "bar_time": bar_time,
                "equity": self.equity,
                "peak_equity": self.risk_manager.equity_peak,
                "ensemble_auc": (auc_long + auc_short) / 2,  # Pass current model confidence
                "adx": float(last_features["adx"].iloc[0]) if "adx" in last_features.columns else 0.0,
                "macd_diff": float(last_features["macd_diff"].iloc[0]) if "macd_diff" in last_features.columns else 0.0,
                "volatility_10": float(last_features["volatility_10"].iloc[0]) if "volatility_10" in last_features.columns else 0.0,
                "dist_from_ema_200": float(last_features["dist_from_ema_200"].iloc[0]) if "dist_from_ema_200" in last_features.columns else 0.0,
            }
            dynamic_risk_params = self.risk_controller.get_params(sym, context)

            atr_multiplier_sl = dynamic_risk_params["atr_multiplier_sl"]
            atr_multiplier_tp = dynamic_risk_params["atr_multiplier_tp"]
            trailing_atr_mult = dynamic_risk_params["trailing_atr_mult"]
            min_prob_long = dynamic_risk_params["min_prob_long"]
            min_prob_short = dynamic_risk_params["min_prob_short"]
            min_ensemble_auc = risk_mgr.cfg.get_symbol_value(sym, 'min_ensemble_auc', 0.55)
            atr_idx = dynamic_risk_params["atr_idx"]
            min_prob_long_idx = dynamic_risk_params.get("min_prob_long_idx", -1)
            min_prob_short_idx = dynamic_risk_params.get("min_prob_short_idx", -1)

            direction, auc_score, conflict = choose_direction(
                prob_long, prob_short, min_prob_long, min_prob_short,
                auc_long, auc_short, min_ensemble_auc)
            if conflict:
                logger.info(f"[{sym}] Conflicting signals skipped: prob_long={prob_long:.3f} and prob_short={prob_short:.3f}.")
            elif direction is None:
                blocked = False
                if prob_long >= min_prob_long and auc_long < min_ensemble_auc:
                    logger.info(f"[{sym}] Long trade blocked due to low ensemble confidence (AUC={auc_long:.4f} < {min_ensemble_auc:.4f}).")
                    blocked = True
                if prob_short >= min_prob_short and auc_short < min_ensemble_auc:
                    logger.info(f"[{sym}] Short trade blocked due to low ensemble confidence (AUC={auc_short:.4f} < {min_ensemble_auc:.4f}).")
                    blocked = True
                if blocked:
                    self.blocked_by_auc += 1

            if direction:
                if sym in self.pending or any(p.symbol == sym and p.status == "open" for p in self.positions):
                    logger.info(f"[{sym}] Signal skipped: a position is already open on this symbol.")
                else:
                    self.pending[sym] = dict(direction=direction, auc=auc_score, atr=atr, params=dynamic_risk_params)
            else:
                logger.info(f"[{sym}] No trade signal. Probs: (Long: {prob_long:.3f}, Short: {prob_short:.3f}) ")

            self.equity_curve.append((bar_time, self.equity))

            # Periodic persistence of Thompson state + CSV (avoid overwriting prod state)
            total_bars = sum(self.bar_counters.values())
            if total_bars % self.save_state_every_bars == 0:
                # write to the backtest-specific file so you don't overwrite live/prod state
                self._persist_bandit_state(force_path=self.backtest_ts_state_file)

            # --- Pruning Check (if in tuning mode) ---
            if trial and pruning_interval > 0 and (i % pruning_interval == 0) and self.cfg.symbols.index(sym) == 0:
                self._check_and_prune(trial, i)

    def _generate_results(self):
        """Generates and saves the backtesting results."""
        eq_df = pd.DataFrame(self.equity_curve, columns=["time", "equity"]).set_index("time")
        trades_df = pd.DataFrame([p.__dict__ for p in self.positions])

        os.makedirs("results", exist_ok=True)
        symbol_str = "_".join([s.replace('#', '') for s in self.cfg.symbols])

        eq_df.to_csv(f"results/equity_curve_{symbol_str}_hybrid_adaptive.csv")
        trades_df.to_csv(f"results/trades_{symbol_str}_hybrid_adaptive.csv")

        # Generate quantstats report
        try:
            trades = [p for p in self.positions]
            long_trades = [t for t in trades if t.direction == "long"]
            short_trades = [t for t in trades if t.direction == "short"]
            logger.info(f"Number of long trades: {len(long_trades)}")
            logger.info(f"Number of short trades: {len(short_trades)}")

            returns = pd.Series([t.pnl for t in trades if t.pnl is not None], index=[t.exit_time for t in trades if t.pnl is not None])
            returns.index = pd.to_datetime(returns.index)

            if not eq_df.empty:
                try:
                    report_path = f"results/report_{symbol_str}_hybrid_adaptive.html"
                    qs.reports.html(eq_df["equity"], output=report_path, title=f"{symbol_str} Hybrid Adaptive Strategy")
                    logger.info(f"QuantStats report saved to {report_path}")
                except Exception as e:
                    logger.warning(f"Failed to generate QuantStats report: {e}")
            else:
                logger.warning("Skipping QuantStats report generation: No returns data available.")
        except Exception as e:
            logger.exception(f"Failed to generate QuantStats report: {e}")

        logger.info(f"=== Hybrid Adaptive Backtest Complete. Final Equity: {self.equity:.2f} === ")
        logger.info(f"Trades: {len(self.positions)}; signals refused by the min_ensemble_auc gate: {self.blocked_by_auc}; "
                    f"entries skipped for a wide spread: {self.skipped_for_spread}.")
        if self.signals and self.skipped_for_size / self.signals > 0.2:
            logger.warning(
                f"{self.skipped_for_size}/{self.signals} signals were skipped at sizing (lot below the broker minimum or risk caps). "
                f"initial_equity={self.initial_equity} is probably too small for this symbol; results will not match a real account."
            )
        logger.info("Results saved to 'results/' directory.")

        # NEW: Generate Thompson Sampling parameter evolution report
        if self.ts_param_history:
            ts_df = pd.DataFrame(self.ts_param_history)
            ts_report_path = f"results/ts_param_evolution_{symbol_str}_hybrid_adaptive.csv"
            ts_df.to_csv(ts_report_path, index=False)
            logger.info(f"Thompson Sampling parameter evolution report saved to {ts_report_path}")

        return trades_df, eq_df

    def _persist_bandit_state(self, force_path: str | None = None):
        """
        Persist RiskController state and ts_param_history CSV.
        If force_path supplied, temporarily write the TS JSON to that path.
        """
        try:
            # optionally override cfg path just for this save
            original_state_file = None
            if force_path:
                original_state_file = self.cfg.thompson_sampling.state_file
                self.cfg.thompson_sampling.state_file = force_path

            # Save RiskController internal JSON (uses RiskController.save_state)
            try:
                self.risk_controller.save_state()
            except Exception:
                logger.exception("Failed to save RiskController state with risk_controller.save_state()")

            # Save human-readable CSV of ts history for offline analysis
            if self.ts_param_history:
                import pandas as pd  # type: ignore
                df = pd.DataFrame(self.ts_param_history)
                df.to_csv(self.ts_history_csv, index=False)

            logger.info(f"Persisted bandit state to {self.cfg.thompson_sampling.state_file} and CSV to {self.ts_history_csv}")
        except Exception as e:
            logger.exception(f"_persist_bandit_state failed: {e}")
        finally:
            if force_path and original_state_file is not None:
                # restore original
                self.cfg.thompson_sampling.state_file = original_state_file

    def run(self, trial: optuna.Trial | None = None, pruning_interval: int = 0):
        logger.info("=== Starting Hybrid Adaptive Backtest ===")
        try:
            for sym in self.cfg.symbols:
                logger.info(f"--- Backtesting Symbol: {sym} ---")

                # Load best feature params from optuna study
                optuna_params = load_optuna_params(sym, self.cfg)
                feature_params = optuna_params.get('features', {}) if optuna_params else {}
                feature_cfg = FeatureCfg(**feature_params)

                # NEW: Get tuned prediction_horizon and min_pct_change, falling back to global defaults
                tuned = optuna_params or {}  # None when there is no tuned-params file
                tuned_prediction_horizon = tuned.get('prediction_horizon', self.cfg.prediction_horizon)
                tuned_min_pct_change = tuned.get('min_pct_change', self.cfg.features.min_pct_change)

                frame = load_backtest_frame(self.cfg, sym, feature_cfg, tuned_prediction_horizon, tuned_min_pct_change)
                if frame.bars.empty:
                    logger.warning(f"No data for {sym}, skipping.")
                    continue
                models = WalkForwardModels(frame.X, frame.y_long, frame.y_short, frame.bars["close"], self.cfg.backtesting.train_bars,
                                           self.cfg.retrain_every_bars, tuned_prediction_horizon,
                                           make_fit_fn(self.cfg, sym, optuna_params))
                if len(frame.X) <= models.start + 1:
                    logger.warning(f"{sym}: {len(frame.X)} bars do not go past backtesting.train_bars={models.start}; nothing to trade.")
                    continue

                self._process_bar(sym, frame, models, trial, pruning_interval)

                logger.info(f"--- Completed Backtest for Symbol: {sym} ---")

                self._force_close_open_positions(sym, frame.bars)
        except KeyboardInterrupt:
            logger.warning("Backtest interrupted by user. Generating results for completed portion...")

        return self._generate_results()


if __name__ == "__main__":
    import numpy as np  # type: ignore
    import random
    import sys

    np.random.seed(42)
    random.seed(42)
    cfg = Cfg.from_yaml("config.yaml")
    setup_logging(level=cfg.logging["level"], to_file=cfg.logging["to_file"], rotate=cfg.logging["rotate"], retention=cfg.logging["retention"])

    mt5_client = None
    if cfg.data_source == "mt5":
        from src.mt5_client import MT5Client
        mt5_client = MT5Client(
            login=os.getenv("MT5_LOGIN"),
            password=os.getenv("MT5_PASSWORD"),
            server=os.getenv("MT5_SERVER"),
            path=os.getenv("MT5_PATH"),
        )
        if not mt5_client.connect():
            logger.error("Failed to connect to MT5, falling back to csv data source.")
            cfg.data_source = "csv"
            mt5_client = None

    cfg.thompson_sampling.state_file = f"ts_risk_controller_state_backtest_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"

    bt = HybridBacktester(cfg, mt5_client)
    try:
        trades_df, eq_df = bt.run()
        print("\n--- Trades Summary ---")
        print(trades_df.tail())
        print("\n--- Equity Curve ---")
        print(eq_df.tail())
    finally:
        try:
            bt._persist_bandit_state(force_path=bt.backtest_ts_state_file)
        except Exception:
            logger.exception("Final persist of bandit state failed.")
        if mt5_client:
            mt5_client.shutdown()
            from src.mt5_client import teardown_connection
            teardown_connection()  # close the process's MT5 connection (a client's shutdown only releases it)
