# src/data.py
from __future__ import annotations
import pandas as pd  # type: ignore
from loguru import logger  # type: ignore


def merge_features_labels(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Merge features (X) and labels (y) with raw df. Returns a DataFrame with 'y', 'close', 'high', 'low', 'volume'.
    Drops rows with NaNs and returns empty DataFrame on failure.
    """
    try:
        out = X.copy()
        out["y"] = y.reindex(out.index)
        if "close" in df.columns:
            out["close"] = df["close"].reindex(out.index)
        if "high" in df.columns:
            out["high"] = df["high"].reindex(out.index)
        if "low" in df.columns:
            out["low"] = df["low"].reindex(out.index)
        out["volume"] = df.get("volume", pd.Series(dtype="float")).reindex(out.index)
        out = out.dropna()
        logger.debug(f"Merged features & labels. Final shape: {out.shape}")
        return out
    except Exception:
        logger.exception("Error merging features and labels")
        return pd.DataFrame()
