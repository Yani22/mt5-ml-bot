"""Bars per year of a time series, for annualising returns (quantstats' `periods` is 252, which is right only for daily bars)."""
import pandas as pd

SECONDS_PER_YEAR = 365.25 * 24 * 3600


def periods_per_year(index) -> int:
    """How many observations a year the index holds: its length over its calendar span. 252 when there is too little to judge (under
    a day of data or fewer than 100 points)."""
    index = pd.DatetimeIndex(index)
    if len(index) < 100:
        return 252
    span = (index.max() - index.min()).total_seconds()
    if span < 24 * 3600:
        return 252
    return max(1, round(len(index) / (span / SECONDS_PER_YEAR)))
