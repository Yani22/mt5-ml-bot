"""quantstats annualises with `periods` (252 by default, meant for daily bars). The backtester writes one equity point per bar, so on M5
data the Sharpe, volatility and Sortino came out about 17 times too small and CAGR almost zero (+10% a year printed as 0.03%)."""
import numpy as np
import pandas as pd
import pytest

import analyzer
from src.periods import periods_per_year


def m5_year():
    return pd.date_range("2024-01-01", periods=288 * 365, freq="5min")


def test_a_year_of_m5_bars_is_about_105_thousand_periods():
    assert periods_per_year(m5_year()) == pytest.approx(105_120, rel=0.01)


def test_daily_bars_give_about_252_to_365():
    idx = pd.bdate_range("2024-01-01", periods=252)
    assert 240 <= periods_per_year(idx) <= 270


def test_a_series_too_short_to_judge_falls_back_to_252():
    assert periods_per_year(pd.date_range("2024-01-01", periods=1, freq="5min")) == 252
    assert periods_per_year(pd.date_range("2024-01-01", periods=50, freq="5min")) == 252


def test_the_analyzer_reports_the_real_annual_growth_on_bar_data(capsys):
    idx = m5_year()
    growth = np.linspace(100.0, 110.0, len(idx))
    equity = pd.DataFrame({"time": idx.astype(str), "equity": growth})
    trades = pd.DataFrame({"entry_time": [idx[0]], "exit_time": [idx[10]], "pnl": [1.0], "direction": ["long"],
                           "exit_price": [1.0], "sl": [0.5], "tp": [1.0]})
    analyzer.analyze_trades(trades, "t", equity)
    out = capsys.readouterr().out
    cagr_line = next(line for line in out.splitlines() if line.startswith("Compound Annual Growth Rate"))
    assert float(cagr_line.split(":")[1].strip().rstrip("%")) == pytest.approx(10.0, abs=0.6)


def test_the_backtest_report_is_given_the_periods_per_year():
    source = open("backtester.py").read()
    assert "periods_per_year=periods_per_year(eq_df.index)" in source
