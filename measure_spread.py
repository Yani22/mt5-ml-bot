import MetaTrader5 as mt5
import os
import time
import numpy as np
from dotenv import load_dotenv
from loguru import logger
from src.mt5_client import MT5Client

# --- Configuration ---
SYMBOL_TO_MEASURE = "GOLDm#"
DURATION_MINUTES = 15
CHECK_INTERVAL_SECONDS = 5
# -------------------

load_dotenv()

def measure_spread():
    """Connects to MT5, measures the spread for a symbol over a duration, and reports statistics."""
    
    logger.info(f"Starting spread measurement for {SYMBOL_TO_MEASURE} over {DURATION_MINUTES} minutes.")
    
    # --- Connection ---
    mt5c = MT5Client(
        login=os.getenv("MT5_LOGIN"),
        password=os.getenv("MT5_PASSWORD"),
        server=os.getenv("MT5_SERVER"),
        path=os.getenv("MT5_PATH"),
    )
    if not mt5c.connect():
        logger.error("Failed to connect to MT5.")
        return

    # --- Get Symbol Info for Pip Size ---
    symbol_info = mt5.symbol_info(SYMBOL_TO_MEASURE)
    if not symbol_info:
        logger.error(f"Could not retrieve info for symbol {SYMBOL_TO_MEASURE}.")
        mt5c.shutdown()
        return
    
    pip_size = symbol_info.point
    logger.info(f"Detected Pip Size (Point) for {SYMBOL_TO_MEASURE}: {pip_size}")

    # --- Measurement Loop ---
    spreads = []
    start_time = time.time()
    end_time = start_time + DURATION_MINUTES * 60

    print("Measuring spread... Press Ctrl+C to stop early.")
    try:
        while time.time() < end_time:
            tick = mt5.symbol_info_tick(SYMBOL_TO_MEASURE)
            if tick:
                spread = tick.ask - tick.bid
                spreads.append(spread)
                # Use \r to update the line in place
                print(f"  Current Spread: {spread:.5f}", end='\r')
            
            time.sleep(CHECK_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        logger.info("\nMeasurement stopped by user.")

    print("\n") # Newline after the loop finishes

    # --- Reporting ---
    if not spreads:
        logger.warning("No spread data was collected. Is the market open for this symbol?")
        mt5c.shutdown()
        return

    spreads_arr = np.array(spreads)
    avg_spread = np.mean(spreads_arr)
    median_spread = np.median(spreads_arr)
    max_spread = np.max(spreads_arr)
    min_spread = np.min(spreads_arr)

    # --- Convert to Pips ---
    avg_spread_pips = avg_spread / pip_size

    logger.info("--- Spread Measurement Results ---")
    logger.info(f"Symbol: {SYMBOL_TO_MEASURE}")
    logger.info(f"Duration: {DURATION_MINUTES} minutes")
    logger.info(f"Samples collected: {len(spreads)}")
    logger.info("---")
    logger.info(f"Average Spread: {avg_spread:.5f} (Price) / {avg_spread_pips:.2f} (Pips)")
    logger.info(f"Median Spread: {median_spread:.5f} (Price)")
    logger.info(f"Min/Max Spread: {min_spread:.5f} / {max_spread:.5f} (Price)")
    logger.info("----------------------------------")
    logger.warning(f"RECOMMENDATION: To be conservative, consider using a `spread_pips` value in your config that is slightly higher than the average, e.g., {avg_spread_pips + 0.5:.1f} or more.")

    mt5c.shutdown()

if __name__ == "__main__":
    measure_spread()
