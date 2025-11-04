# src/tca_analyzer.py
import pandas as pd
import numpy as np
import datetime
import os
from typing import Dict, Any, List, Tuple
from loguru import logger

class TcaAnalyzer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tca_file = "results/tca_metrics.csv"
        self.alert_manager = None # Will be set by main.py

    def set_alert_manager(self, alert_manager):
        self.alert_manager = alert_manager

    def _load_tca_data(self, lookback_days: int) -> pd.DataFrame:
        if not os.path.exists(self.tca_file):
            return pd.DataFrame()
        
        df = pd.read_csv(self.tca_file)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()

        if lookback_days > 0:
            start_date = datetime.datetime.now(datetime.timezone.utc) - pd.Timedelta(days=lookback_days)
            df = df[df.index >= start_date]
        
        return df

    def analyze(self, lookback_days: int = 7) -> Tuple[str, Dict[str, Any]]:
        tca_df = self._load_tca_data(lookback_days)

        if tca_df.empty:
            return "No TCA data available for analysis.", {}

        report_lines = []
        suggestions = {}
        
        report_lines.append(f"--- TCA Analysis Report (Last {lookback_days} Days) ---")
        report_lines.append(f"Total Trades Analyzed: {len(tca_df)}")
        report_lines.append(f"Average Slippage (pips): {tca_df['slippage_pips'].mean():.3f}")
        report_lines.append(f"Median Slippage (pips): {tca_df['slippage_pips'].median():.3f}")
        report_lines.append(f"Total Slippage Cost (currency): {tca_df['slippage_currency'].sum():.2f}")
        report_lines.append(f"Total Commission Cost (currency): {tca_df['commission_per_trade'].sum():.2f}")
        report_lines.append(f"Total Transaction Cost (currency): {tca_df['total_transaction_cost_currency'].sum():.2f}")

        # Slippage Distribution
        positive_slippage_count = (tca_df['slippage_pips'] > 0.01).sum() # > 0.01 to account for floating point
        negative_slippage_count = (tca_df['slippage_pips'] < -0.01).sum()
        neutral_slippage_count = len(tca_df) - positive_slippage_count - negative_slippage_count
        
        report_lines.append(f"Slippage Distribution: Positive={positive_slippage_count}, Negative={negative_slippage_count}, Neutral={neutral_slippage_count}")

        # Per-Symbol Analysis
        report_lines.append("\n--- Per-Symbol Analysis ---")
        for symbol, sym_df in tca_df.groupby('symbol'):
            avg_slippage_sym = sym_df['slippage_pips'].mean()
            report_lines.append(f"  {symbol}: Avg Slippage={avg_slippage_sym:.3f} pips, Trades={len(sym_df)}")
            
            # Suggestion: High slippage for a specific symbol
            if avg_slippage_sym < self.cfg.tca.slippage_threshold_pips_warning:
                suggestions[f"{symbol}_slippage"] = f"Slippage for {symbol} ({avg_slippage_sym:.3f} pips) is worse than warning threshold ({self.cfg.tca.slippage_threshold_pips_warning}). Consider reviewing its trading parameters or liquidity."

        # Overall Suggestions
        if tca_df['slippage_pips'].mean() < self.cfg.tca.slippage_threshold_pips_critical:
            suggestions["overall_slippage"] = f"Overall average slippage ({tca_df['slippage_pips'].mean():.3f} pips) is worse than critical threshold ({self.cfg.tca.slippage_threshold_pips_critical}). Consider increasing `deviation` in config.yaml or reviewing broker execution."
        
        # Example: If total transaction cost is too high relative to PnL (requires PnL data, which we don't have in TCA yet)
        # For now, we'll just warn if total transaction cost is high in absolute terms
        if tca_df['total_transaction_cost_currency'].sum() > self.cfg.tca.total_cost_currency_warning:
             suggestions["overall_cost"] = f"Total transaction cost ({tca_df['total_transaction_cost_currency'].sum():.2f}) is high. Review spread and commission settings."

        final_report = "\n".join(report_lines)
        if suggestions:
            final_report += "\n\n--- Suggestions ---\n" + "\n".join([f"- {s}" for s in suggestions.values()])
        
        return final_report, suggestions

    def run_analysis_and_notify(self, lookback_days: int = 7):
        report, suggestions = self.analyze(lookback_days)
        logger.info(f"TCA Analysis Report:\n{report}")
        if self.alert_manager:
            message = f"<b>TCA Analysis Report (Last {lookback_days} Days):</b>\n{report}"
            if suggestions:
                message += "\n\n<b>Suggestions:</b>\n" + "\n".join([f"- {s}" for s in suggestions.values()])
            self.alert_manager.send_alert(message, level="INFO", category="TCA_ANALYSIS")
