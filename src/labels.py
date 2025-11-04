import pandas as pd  # type: ignore

def generate_labels(df: pd.DataFrame, horizon: int, min_pct_change: float = 0.0) -> pd.Series:
    """
    Generates binary labels based on forward percentage change, with a minimum threshold.
    1: Price moves up by at least min_pct_change
    0: Price does not move up by that amount
    """
    fwd = df["close"].pct_change(horizon).shift(-horizon)
    
    y = (fwd > min_pct_change).astype(int)

    return y

def generate_long_short_labels(df: pd.DataFrame, horizon: int, min_pct_change: float = 0.0) -> tuple[pd.Series, pd.Series]:
    """
    Generates separate long and short labels based on forward percentage change, with a minimum threshold.
    """
    # Correctly calculate forward returns
    future_close = df["close"].shift(-horizon)
    fwd_pct_change = (future_close - df["close"]) / df["close"]
    
    y_long = (fwd_pct_change > min_pct_change).astype(int)
    y_short = (fwd_pct_change < -min_pct_change).astype(int)

    return y_long, y_short