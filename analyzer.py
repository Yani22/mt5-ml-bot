# analyzer.py
import pandas as pd  # type: ignore
import sys
import os
import quantstats as qs # type: ignore

def analyze_trades(df: pd.DataFrame, name: str, equity_df: pd.DataFrame | None = None):
    """Analyzes a dataframe of trades and prints a summary."""
    
    if df.empty:
        print(f"--- No trades to analyze for: {name} ---")
        return

    # Convert time columns to datetime objects FIRST
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])

    # Calculate trade duration in minutes
    df['duration_minutes'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 60

    total_trades = len(df)
    winning_trades = df[df['pnl'] > 0]
    losing_trades = df[df['pnl'] <= 0]

    num_winning_trades = len(winning_trades)
    num_losing_trades = len(losing_trades)
    win_rate = (num_winning_trades / total_trades) * 100 if total_trades > 0 else 0

    avg_pnl = df['pnl'].mean()
    avg_pnl_winning = winning_trades['pnl'].mean() if num_winning_trades > 0 else 0
    avg_pnl_losing = losing_trades['pnl'].mean() if num_losing_trades > 0 else 0

    max_win = winning_trades['pnl'].max() if num_winning_trades > 0 else 0
    max_loss = losing_trades['pnl'].min() if num_losing_trades > 0 else 0

    avg_duration_all = df['duration_minutes'].mean()
    avg_duration_winning = winning_trades['duration_minutes'].mean() if num_winning_trades > 0 else 0
    avg_duration_losing = losing_trades['duration_minutes'].mean() if num_losing_trades > 0 else 0

    # Analyze SL/TP hits
    sl_hits = 0
    tp_hits = 0
    other_exits = 0
    tolerance = 0.00001

    for index, row in df.iterrows():
        if row.get('direction') == 'long':
            if row['pnl'] <= 0 and (row['exit_price'] <= row['sl'] + tolerance):
                sl_hits += 1
            elif row['pnl'] > 0 and (row['exit_price'] >= row['tp'] - tolerance):
                tp_hits += 1
            else:
                other_exits += 1
        elif row.get('direction') == 'short':
            if row['pnl'] <= 0 and (row['exit_price'] >= row['sl'] - tolerance):
                sl_hits += 1
            elif row['pnl'] > 0 and (row['exit_price'] <= row['tp'] + tolerance):
                tp_hits += 1
            else:
                other_exits += 1

    print(f"--- Trade Analysis Summary for: {name} ---")
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {num_winning_trades} ({win_rate:.2f}%)")
    print(f"Losing Trades: {num_losing_trades} ({100 - win_rate:.2f}%)")
    print(f"Average PnL per Trade: {avg_pnl:.2f}")
    print(f"Average PnL for Winning Trades: {avg_pnl_winning:.2f}")
    print(f"Average PnL for Losing Trades: {avg_pnl_losing:.2f}")
    print(f"Maximum Winning Trade PnL: {max_win:.2f}")
    print(f"Maximum Losing Trade PnL: {max_loss:.2f}")
    print(f"Average Trade Duration (minutes): {avg_duration_all:.2f}")
    print(f"  - Winning Trades Avg Duration: {avg_duration_winning:.2f}")
    print(f"  - Losing Trades Avg Duration: {avg_duration_losing:.2f}")
    print(f"Trades Closed by SL: {sl_hits}")
    print(f"Trades Closed by TP: {tp_hits}")
    print(f"Trades Closed by Other Means: {other_exits}")

    if equity_df is not None and not equity_df.empty:
        # Ensure equity_df is sorted by time and index is datetime
        equity_df['time'] = pd.to_datetime(equity_df['time'])
        equity_df = equity_df.set_index('time').sort_index()
        
        # Calculate daily returns
        returns = equity_df['equity'].pct_change().dropna()

        if not returns.empty:
            # Annualization factor for daily returns (assuming 252 trading days)
            annualization_factor = 252

            # Compound Annual Growth Rate (CAGR)
            cagr = qs.stats.cagr(returns)

            # Annualized Volatility
            annual_volatility = qs.stats.volatility(returns, annualize=True)

            # Sharpe Ratio
            sharpe_ratio = qs.stats.sharpe(returns, annualize=True) if returns.std() > 0 else 0.0
            
            # Max Drawdown
            max_drawdown = qs.stats.max_drawdown(returns)

            # Calmar Ratio
            calmar_ratio = qs.stats.calmar(returns, periods=252) if max_drawdown != 0 else 0.0

            # Sortino Ratio
            sortino_ratio = qs.stats.sortino(returns, annualize=True) if returns[returns < 0].std() > 0 else 0.0

            # Value at Risk (VaR) - 95% confidence
            var_95 = qs.stats.var(returns)

            print(f"\n--- Portfolio Metrics for: {name} ---")
            print(f"Final Equity: {equity_df['equity'].iloc[-1]:.2f}")
            print(f"Initial Equity: {equity_df['equity'].iloc[0]:.2f}")
            print(f"Compound Annual Growth Rate (CAGR): {cagr:.2%}")
            print("  * Explanation: The average annual rate at which an investment has grown over a specified period, assuming profits are reinvested.")
            print(f"Annualized Volatility: {annual_volatility:.2%}")
            print("  * Explanation: Measures the degree of variation of a trading strategy's returns over a year. Higher volatility means higher risk.")
            print(f"Sharpe Ratio (Ann.): {sharpe_ratio:.2f}")
            print("  * Explanation: Measures risk-adjusted return. Higher is better, indicating more return per unit of risk.")
            print(f"Max Drawdown: {max_drawdown:.2%}")
            print("  * Explanation: The largest peak-to-trough decline in the equity curve, representing the greatest loss from a peak.")
            print(f"Calmar Ratio (Ann.): {calmar_ratio:.2f}")
            print("  * Explanation: Measures risk-adjusted return by dividing the CAGR by the absolute value of the maximum drawdown. Higher is better.")
            print(f"Sortino Ratio (Ann.): {sortino_ratio:.2f}")
            print("  * Explanation: Similar to Sharpe, but only penalizes downside volatility (bad volatility). Higher is better.")
            print(f"Value at Risk (VaR 95%): {var_95:.2%}")
            print("  * Explanation: Estimates the maximum expected loss over a given period with a 95% confidence level. E.g., a 5% VaR of -2% means there's a 5% chance of losing more than 2% over the period.")
        else:
            print(f"\n--- Portfolio Metrics for: {name} ---")
            print("Not enough data to calculate portfolio metrics.")
    print("\n")


if __name__ == "__main__":
    trade_files = []
    equity_files = []
    results_dir = 'results'

    if len(sys.argv) < 2:
        print("No file path provided. Analyzing all 'trades_*.csv' and 'equity_curve_*.csv' files in 'results/' directory.")
        try:
            all_files = os.listdir(results_dir)
            trade_files = sorted([
                os.path.join(results_dir, f) 
                for f in all_files
                if f.startswith('trades_') and f.endswith('.csv')
            ])
            equity_files = sorted([
                os.path.join(results_dir, f) 
                for f in all_files
                if f.startswith('equity_curve_') and f.endswith('.csv')
            ])

            if not trade_files:
                print("No trade files found in 'results/'.")
                sys.exit(1)
        except FileNotFoundError:
            print("Error: Could not find the 'results/' directory.")
            sys.exit(1)
    else:
        # If specific files are provided, assume they are trade files and try to find corresponding equity files
        trade_files = sys.argv[1:]
        for tf in trade_files:
            base_name = os.path.basename(tf).replace('trades_', 'equity_curve_')
            eq_path = os.path.join(os.path.dirname(tf), base_name)
            if os.path.exists(eq_path):
                equity_files.append(eq_path)
            else:
                equity_files.append(None) # Append None if no matching equity file

    # Create a mapping from trade file base name to equity file path
    trade_to_equity_map = {}
    for tf in trade_files:
        base_name = os.path.basename(tf).replace('trades_', '')
        matching_equity_file = next((ef for ef in equity_files if ef and os.path.basename(ef).replace('equity_curve_', '') == base_name), None)
        trade_to_equity_map[tf] = matching_equity_file

    for filepath in trade_files:
        print(f"============== Analyzing file: {filepath} ==============")
        try:
            main_df = pd.read_csv(filepath)
            equity_filepath = trade_to_equity_map.get(filepath)
            equity_df = pd.read_csv(equity_filepath) if equity_filepath else None

        except FileNotFoundError:
            print(f"Error: The file '{filepath}' or its corresponding equity file was not found.\n")
            continue
        except Exception as e:
            print(f"Error loading files for {filepath}: {e}\n")
            continue

        # --- Overall Analysis ---
        analyze_trades(main_df.copy(), f"Overall Portfolio ({os.path.basename(filepath)})", equity_df.copy() if equity_df is not None else None)

        # --- Per-Symbol Analysis (only trade data for now, equity curve is portfolio-wide) ---
        symbols = main_df['symbol'].unique()
        if len(symbols) > 1:
            for sym in symbols:
                symbol_df = main_df[main_df['symbol'] == sym]
                analyze_trades(symbol_df.copy(), f"Symbol: {sym} ({os.path.basename(filepath)})")