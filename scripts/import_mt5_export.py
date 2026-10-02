"""Convert MT5 'Export Bars' CSVs (tab separated) to the bot's cache format.

Usage: python scripts/import_mt5_export.py [src_dir] [dst_dir]
Writes <SYMBOL>_<TF>.csv (no '#') with a UTC 'time' index and open/high/low/close/volume,
plus <SYMBOL>_<TF>_spread.csv (spread in points) for cost modelling.
"""
import glob
import os
import sys

import pandas as pd


def convert(path: str, dst_dir: str) -> None:
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip("<>").lower() for c in df.columns]
    df["time"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M:%S", utc=True)
    df = df.set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.rename(columns={"tickvol": "volume"})
    base = os.path.basename(path).replace("#", "")[:-4]
    df[["open", "high", "low", "close", "volume"]].to_csv(os.path.join(dst_dir, base + ".csv"))
    df[["spread"]].to_csv(os.path.join(dst_dir, base + "_spread.csv"))
    print(f"{base}: {len(df)} bars {df.index[0]} -> {df.index[-1]}, median spread {df['spread'].median():.0f} pts")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/raw_export"
    dst = sys.argv[2] if len(sys.argv) > 2 else "data/historical_data"
    os.makedirs(dst, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(src, "*.csv"))):
        convert(p, dst)
