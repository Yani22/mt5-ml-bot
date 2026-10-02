"""Exploration 2: cross-asset features for an H1 target symbol (risk sentiment, dollar, gold, crypto).

For each reference symbol R: pct change of R's close over 1/4/24 H1 bars, aligned to the target's bar index by
as-of (forward fill of PAST values only: bars share open time and close time, so no look-ahead). Reference symbols
that trade fewer hours (indices) carry their last value through closed hours.
"""
import pandas as pd

REFS = ["US500Cash", "US30Cash", "GOLD", "EURUSD", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "BTCUSD"]


def cross_features(target_index: pd.DatetimeIndex, target: str, data: str, refs=None) -> pd.DataFrame:
    tz = target_index.tz
    naive = target_index.tz_localize(None) if tz is not None else target_index  # compare wall-clock labels only
    out = pd.DataFrame(index=naive)
    for ref in (refs or REFS):
        if ref == target:
            continue
        r = pd.read_csv(f"{data}/{ref}_H1.csv", index_col=0, parse_dates=True)["close"]
        r.index = r.index.tz_localize(None) if r.index.tz is not None else r.index
        aligned = r.reindex(r.index.union(naive)).ffill().reindex(naive)
        for n in (1, 4, 24):
            out[f"x_{ref}_ret{n}"] = aligned.pct_change(n)
    out.index = target_index
    return out
