"""Contrarian trades on extreme speculator positioning (CFTC Traders in Financial Futures, weekly).

PRE-REGISTERED (written before the first run, 2026-10-02). Nothing below is fitted; every number is a conventional
default fixed in advance, so the whole 2010-2026 sample is out-of-sample for this rule.
  * Positioning: Leveraged Funds net = (long - short) / open interest, for the CME futures of AUD, GBP, CAD, EUR,
    JPY, CHF, NZD (contract codes below). Percentile = rank of the latest value within the trailing 156 weeks
    (>= 104 weeks required), using data up to and including that report only.
  * Timing: reports are dated Tuesday and published Friday 15:30 ET. The trade enters at the first daily open on or
    after report date + 6 days (the Monday), so nothing is used before it was public.
  * Rule (CONTRARIAN, the only variant that can PASS): percentile >= 0.90 (crowded long) -> short that currency vs
    USD; percentile <= 0.10 -> long it. Pairs: AUDUSD GBPUSD EURUSD NZDUSD (same direction as the future) and
    USDJPY USDCAD USDCHF (inverted). Exit: 20 trading days, or a 3 x ATR14(daily) hard stop, whichever comes first.
    One position per pair at a time.
  * Costs: median 2024+ H1 recorded spread per pair, charged once per round trip (long pays at entry, short pays at
    exit, shorts trigger the stop on the ask) + 0.1 x spread slippage per side. Swap: a flat 0.02 ATR per calendar
    day held (conservative: roughly 1-1.5 pips/day on EURUSD, more than typical broker swaps).
  * PASS if pooled over all pairs: mean net R > 0 AND year-bootstrap 95% CI low > 0 AND >= 60% of years positive
    AND the swap-charged mean > 0 AND >= 4 of 7 pairs individually have mean net R > 0.
  * CONTROL (reported only, cannot pass): the momentum direction (follow the crowd), to show which way any effect runs.
Data note (added before the first run, after fixing a loader bug that had silently dropped 2010-2012): the TFF
reports start 2010-07-20, so the first signals appear mid-2012 and the tested window is ~2012-2026.
Weekly signals give roughly 15-25 trades per pair per year; if pooled trades < 300 the verdict is INCONCLUSIVE.

Usage: python scripts/cot_contrarian.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import ta  # type: ignore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resad_recipe as rr  # noqa: E402

CODES = {"AUD": "232741", "GBP": "096742", "CAD": "090741", "EUR": "099741", "JPY": "097741", "CHF": "092741",
         "NZD": "112741"}
PAIRS = {"AUDUSD": ("AUD", 1), "GBPUSD": ("GBP", 1), "EURUSD": ("EUR", 1), "NZDUSD": ("NZD", 1),
         "USDJPY": ("JPY", -1), "USDCAD": ("CAD", -1), "USDCHF": ("CHF", -1)}
WINDOW, MIN_WEEKS, HI, LO = 156, 104, 0.90, 0.10
CAP, STOP_ATR, SWAP_ATR_DAY, SLIP_FRAC = 20, 3.0, 0.02, 0.1


def load_cot(d="data/cot"):
    frames = []
    for f in sorted(glob.glob(f"{d}/txt_*/*.txt")):
        df = pd.read_csv(f, low_memory=False)
        if "Report_Date_as_YYYY-MM-DD" in df.columns:  # 2013+ layout
            df["date"] = pd.to_datetime(df["Report_Date_as_YYYY-MM-DD"])
        else:  # 2010-2012 layout (column name differs)
            df["date"] = pd.to_datetime(df["Report_Date_as_MM_DD_YYYY"], format="mixed")
        frames.append(df)
    c = pd.concat(frames)
    assert c["date"].notna().all(), "unparsed COT report dates"
    c["code"] = c["CFTC_Contract_Market_Code"].astype(str).str.strip().str.zfill(6)
    c["net"] = (c["Lev_Money_Positions_Long_All"] - c["Lev_Money_Positions_Short_All"]) / c["Open_Interest_All"]
    return c


def percentile(net: pd.Series) -> pd.Series:
    """Rank of each value within the trailing WINDOW values (itself included, nothing after it)."""
    return net.rolling(WINDOW, min_periods=MIN_WEEKS).apply(lambda w: (w <= w[-1]).mean(), raw=True)


def pair_side(pct: float, sign: int, contrarian: bool = True) -> int:
    """+1 buy / -1 sell the pair. Crowded long the foreign currency (pct high) -> contrarian sells it."""
    if pct >= HI:
        foreign = -1
    elif pct <= LO:
        foreign = 1
    else:
        return 0
    return (foreign if contrarian else -foreign) * sign


def entry_date(report_date: pd.Timestamp) -> pd.Timestamp:
    return report_date + pd.Timedelta(days=6)  # Tuesday report -> Monday after the Friday release


def load_daily(pair):
    raw = pd.read_csv(f"data/raw_export/old/{pair}#_Daily.csv", sep="\t")
    raw.columns = [c.strip("<>").lower() for c in raw.columns]
    digits = len(str(open(f"data/raw_export/old/{pair}#_Daily.csv").readlines()[1].split("\t")[4]).split(".")[1])
    raw["date"] = pd.to_datetime(raw["date"], format="%Y.%m.%d")
    bars = raw.set_index("date").sort_index()[["open", "high", "low", "close"]]
    sp_pts = pd.read_csv(f"data/historical_data/{pair}_H1_spread.csv", index_col=0)["spread"].median()
    return bars, float(sp_pts) * 10.0 ** -digits


def trades(pair, pct: pd.Series, bars, spread_px, contrarian=True):
    ccy, sign = PAIRS[pair]
    o, h, l_ = bars["open"].to_numpy(), bars["high"].to_numpy(), bars["low"].to_numpy()
    atr = ta.volatility.AverageTrueRange(bars.high, bars.low, bars.close, window=14).average_true_range().to_numpy()
    sp = np.full(len(bars), spread_px)
    slip = SLIP_FRAC * spread_px
    out, free = [], 0
    for rdate, p in pct.dropna().items():
        side = pair_side(p, sign, contrarian)
        i = bars.index.searchsorted(entry_date(rdate))
        if side == 0 or i < max(free, 15) or i + CAP + 1 >= len(bars):
            continue
        entry = o[i] + (spread_px if side == 1 else 0.0)
        risk = STOP_ATR * atr[i - 1]  # ATR known before the entry bar opens
        r = rr.first_touch(side, entry, entry + side * 1e9, entry - side * risk, o, h, l_, sp, i, CAP)
        if r is None:
            continue
        exit_px, held, outcome = r
        net = side * (exit_px - entry) - 2 * slip
        out.append((pair, bars.index[i], side, net / risk, held, outcome, net, bars.index[i].year))
        free = i + held
    return pd.DataFrame(out, columns=["pair", "entry", "side", "net_r", "held", "outcome", "net_pts", "fold"])


def run(contrarian=True):
    cot = load_cot()
    allt = []
    for pair, (ccy, _) in PAIRS.items():
        s = cot[cot.code == CODES[ccy]].drop_duplicates("date").set_index("date").sort_index()["net"]
        assert s.index.min() <= pd.Timestamp("2010-12-31") and len(s) > 800, f"{ccy}: COT coverage {s.index.min()} n={len(s)}"
        bars, spx = load_daily(pair)
        allt.append(trades(pair, percentile(s), bars, spx, contrarian))
    return pd.concat(allt, ignore_index=True)


if __name__ == "__main__":
    swap_r_per_bar = SWAP_ATR_DAY * 7 / 5 / STOP_ATR  # per trading day held, in R
    for name, contrarian in (("CONTRARIAN", True), ("CONTROL momentum", False)):
        t = run(contrarian)
        j = rr.judge(t, swap_r_per_bar)
        per = t.groupby("pair").net_r.agg(["size", "mean"]).round(3)
        n_pos = int((per["mean"] > 0).sum())
        ok = j["trades"] >= 300 and j["mean_r"] > 0 and j["ci_lo"] > 0 and j["folds_pos"] >= 60 \
            and j["mean_r_swap"] > 0 and n_pos >= 4
        verdict = "INCONCLUSIVE" if j["trades"] < 300 else ("PASS" if ok else "FAIL")
        print(f"\n{name}: {verdict}")
        print({k: round(float(v), 3) for k, v in j.items()}, f"pairs positive {n_pos}/7")
        print(per.T.to_string())
        if contrarian:
            t.to_csv("results/cot_contrarian_trades.csv", index=False)
