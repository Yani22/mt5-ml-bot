
import pandas as pd  # type: ignore

# trades_df = pd.read_csv("results/trades_GOLDm_hybrid_adaptive.csv")
trades_df = pd.read_csv("results/trades_EURUSDm_hybrid_adaptive.csv")
# trades_df = pd.read_csv("results/trades_GBPUSDm_hybrid_adaptive.csv")
# trades_df = pd.read_csv("results/trades_USDJPYm_hybrid_adaptive.csv")

# Basic metrics
num_trades = len(trades_df)
winning_trades = trades_df[trades_df["pnl"] > 0]
num_winning_trades = len(winning_trades)
num_losing_trades = num_trades - num_winning_trades
win_rate = (num_winning_trades / num_trades) * 100 if num_trades > 0 else 0

# Profit analysis
gross_profit = winning_trades["pnl"].sum()
average_profit = gross_profit / num_winning_trades if num_winning_trades > 0 else 0

losing_trades = trades_df[trades_df["pnl"] <= 0]
gross_loss = abs(losing_trades["pnl"].sum())
average_loss = gross_loss / num_losing_trades if num_losing_trades > 0 else 0

profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

print("--- Backtest Analysis ---")
print(f"Total Trades: {num_trades}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Average Profit: {average_profit:.2f}")
print(f"Average Loss: {average_loss:.2f}")
print(f"Profit Factor: {profit_factor:.2f}")
