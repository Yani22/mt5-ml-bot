"""Leak-free test of the "Resad recipe": LightGBM long/short, enter when raw p > 0.40 (higher of the two),
take-profit 4 x ATR14, stop-loss 2 x ATR14, no time exit in the original bot (here capped at --cap bars).

PRE-REGISTERED (written before the first run, 2026-10-02):
  * Confirmatory cell (the only one that can PASS): USDJPY H1, data 2010-01 .. 2023-12 (data/historical_old),
    which the design never saw. Two label variants, both pre-registered:
      dir12  - the old bot's label: close[t+12] > close[t] (long), < (short)
      tb     - barrier label: TP (4 ATR) touched before SL (2 ATR) within the cap, per side
    PASS if, for a variant, mean net R/trade > 0 AND fold-bootstrap 95% CI low > 0 AND >= 60% folds positive
    AND the swap-charged mean is also > 0. Two variants -> one may pass by chance; report both.
  * Everything else (GBPJPY, EURUSD, GBPUSD, GOLD, BTCUSD on 2024-26 H1, and M5) is EXPLORATORY only: that window
    was already used by >300 tests today. It can only fail, or justify exporting pre-2024 data for a real holdout.
Simulator rules: signal at close of bar t, fill at open t+1 (long at ask = bid + spread, short at bid). Bars are bid.
Long exits at bid; short exits at ask, so short TP/SL are checked against low+spread / high+spread of that bar.
TP and SL inside the same bar -> SL first. A bar opening beyond a barrier exits at its open. Cap reached -> exit at
the open of bar t+1+cap. Train labels end `cap` bars before the test fold (purge = the longest look-ahead).
Old bars have spread 0, so the spread is the median recorded spread for that hour of day in 2024+.
Net is reported in R (net / SL distance) and in points; slippage = --slip-frac x median spread per side.

Usage: python scripts/resad_recipe.py --mode holdout
       python scripts/resad_recipe.py --mode explore --tf H1 --symbols GBPJPY EURUSD GBPUSD GOLD BTCUSD USDJPY
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.h1_features import build_h1  # noqa: E402

TP_ATR, SL_ATR, TH = 4.0, 2.0, 0.40
DIR_HZ = 12
PARAMS = dict(n_estimators=200, learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              num_leaves=15, min_child_samples=50, n_jobs=4, verbose=-1, random_state=0)


def first_touch(side, entry, tp, sl, o, h, l_, sp, start, cap):
    """Walk bars start..start+cap-1 (start = entry bar). Returns (exit_px, bars_held, outcome) with outcome
    'tp', 'sl' or 'cap'. Long checks bid h/l; short checks ask = h/l + spread. SL wins a same-bar tie."""
    for k in range(cap):
        b = start + k
        if b >= len(o):
            return None
        if side == 1:
            hi, lo, op = h[b], l_[b], o[b]
            if k > 0 and op <= sl:
                return op, k, "sl"
            if k > 0 and op >= tp:
                return op, k, "tp"
            if lo <= sl:
                return sl, k + 1, "sl"
            if hi >= tp:
                return tp, k + 1, "tp"
        else:
            hi, lo, op = h[b] + sp[b], l_[b] + sp[b], o[b] + sp[b]
            if k > 0 and op >= sl:
                return op, k, "sl"
            if k > 0 and op <= tp:
                return op, k, "tp"
            if hi >= sl:
                return sl, k + 1, "sl"
            if lo <= tp:
                return tp, k + 1, "tp"
    b = start + cap
    if b >= len(o):
        return None
    return (o[b] if side == 1 else o[b] + sp[b]), cap, "cap"


def barrier_labels(o, h, l_, c, sp, atr, cap):
    """tb label per bar t: 1 if a trade entered at open t+1 hits TP before SL within cap bars. NaN when the
    forward window is incomplete (those rows are dropped)."""
    n = len(c)
    yl, ys = np.full(n, np.nan), np.full(n, np.nan)
    for t in range(n):
        if t + 1 + cap >= n or not np.isfinite(atr[t]):
            continue
        for side, y in ((1, yl), (-1, ys)):
            entry = o[t + 1] + (sp[t + 1] if side == 1 else 0.0)
            tp, sl = entry + side * TP_ATR * atr[t], entry - side * SL_ATR * atr[t]
            r = first_touch(side, entry, tp, sl, o, h, l_, sp, t + 1, cap)
            y[t] = 1.0 if r is not None and r[2] == "tp" else 0.0
    return yl, ys


def dir_labels(c, hz):
    n = len(c)
    fwd = np.full(n, np.nan)
    fwd[: n - hz] = c[hz:] - c[: n - hz]
    yl = np.where(np.isnan(fwd), np.nan, (fwd > 0).astype(float))
    ys = np.where(np.isnan(fwd), np.nan, (fwd < 0).astype(float))
    return yl, ys


def simulate(pl, ps, rows, o, h, l_, sp, atr, cap, slip):
    out, i = [], 0
    while i < len(rows):
        t = rows[i]
        side = 1 if (pl[i] > TH and pl[i] >= ps[i]) else -1 if (ps[i] > TH and ps[i] > pl[i]) else 0
        if side == 0 or t + 1 + cap >= len(o):
            i += 1
            continue
        entry = o[t + 1] + (sp[t + 1] if side == 1 else 0.0)
        risk = SL_ATR * atr[t]
        tp, sl = entry + side * TP_ATR * atr[t], entry - side * risk
        r = first_touch(side, entry, tp, sl, o, h, l_, sp, t + 1, cap)
        if r is None:
            break
        exit_px, held, outcome = r
        net = side * (exit_px - entry) - 2 * slip
        out.append((t + 1, side, net, net / risk, held, outcome))
        while i < len(rows) and rows[i] < t + 1 + held:  # one position at a time
            i += 1
    return out


def hour_spread(sp_series, recent_from="2024-01-01"):
    """Recorded spread where present; old zero-spread bars get the 2024+ median for that hour of day."""
    recent = sp_series[sp_series.index >= pd.Timestamp(recent_from, tz=sp_series.index.tz)]
    prof = recent.groupby(recent.index.hour).median()
    filled = pd.Series(sp_series.index.hour.map(prof).to_numpy(), index=sp_series.index)
    return sp_series.where(sp_series > 0, filled)


def point_size(symbol, raw_dirs):
    name = {"GOLD": "GOLD#"}.get(symbol, symbol + "#")
    for d in raw_dirs:
        for tf in ("M5", "H1"):
            p = f"{d}/{name}_{tf}.csv"
            if os.path.exists(p):
                row = open(p).readlines()[1].split("\t")[5]
                return 10.0 ** -len(row.split(".")[1]) if "." in row else 1.0
    raise FileNotFoundError(name)


def run(symbol, tf, data_dir, end, cap, train, test, slip_frac, label):
    bars = pd.read_csv(f"{data_dir}/{symbol}_{tf}.csv", index_col=0, parse_dates=True)
    sps = pd.read_csv(f"{data_dir}/{symbol}_{tf}_spread.csv", index_col=0, parse_dates=True)["spread"]
    sps = hour_spread(sps.reindex(bars.index).ffill().fillna(0))
    if end:
        keep = bars.index <= pd.Timestamp(end, tz=bars.index.tz) + pd.Timedelta("23h59min")
        bars, sps = bars[keep], sps[keep]
    pt = point_size(symbol, ["data/raw_export", "data/raw_export/old"])
    sp = sps.to_numpy() * pt
    slip = slip_frac * float(np.median(sp))
    X = build_h1(bars)  # same feature builder for M5 and H1 (H4/D1 context resampled from the bars)
    o, h, l_, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
    atr = X["atr_14"].reindex(bars.index).to_numpy()
    if label == "tb":
        yl, ys = barrier_labels(o, h, l_, c, sp, atr, cap)
    else:
        yl, ys = dir_labels(c, DIR_HZ)
    pos = bars.index.get_indexer(X.index)
    ok = ~np.isnan(yl[pos]) & np.isfinite(atr[pos])
    X, pos = X[ok], pos[ok]
    yl, ys = yl[pos].astype(int), ys[pos].astype(int)
    purge = max(cap, DIR_HZ) + 1
    out, start, fold = [], train, 0
    while start + test <= len(X):
        tr, te = slice(start - train, start - purge), slice(start, start + test)
        pl = LGBMClassifier(**PARAMS).fit(X.iloc[tr], yl[tr]).predict_proba(X.iloc[te])[:, 1]
        ps = LGBMClassifier(**PARAMS).fit(X.iloc[tr], ys[tr]).predict_proba(X.iloc[te])[:, 1]
        for eb, side, net, r, held, outcome in simulate(pl, ps, pos[te], o, h, l_, sp, atr, cap, slip):
            out.append((symbol, tf, label, fold, bars.index[eb], side, net / pt, r, held, outcome,
                        SL_ATR * atr[eb - 1] / pt))
        start += test
        fold += 1
    return pd.DataFrame(out, columns=["symbol", "tf", "label", "fold", "entry", "side", "net_pts", "net_r", "held",
                                      "outcome", "risk_pts"])


def judge(t, swap_r_per_bar=0.0):
    """Mean net R, fold-bootstrap 95% CI, % folds positive. swap_r_per_bar: rough swap charge per held bar."""
    if t.empty:
        return dict(trades=0)
    t = t.assign(net_sw=t.net_r - swap_r_per_bar * t.held)
    g = t.groupby("fold").agg(s=("net_r", "sum"), n=("net_r", "size"))
    rng = np.random.default_rng(0)
    boots = [g.iloc[rng.integers(0, len(g), len(g))].pipe(lambda s: s.s.sum() / s.n.sum()) for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(trades=len(t), folds=len(g), mean_r=t.net_r.mean(), ci_lo=lo, ci_hi=hi,
                folds_pos=(g.s > 0).mean() * 100, mean_r_swap=t.net_sw.mean(), win=(t.net_r > 0).mean() * 100,
                tp=(t.outcome == "tp").mean() * 100, cap_hit=(t.outcome == "cap").mean() * 100,
                held=t.held.mean(), pts=t.net_pts.mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["holdout", "explore"], default="holdout")
    ap.add_argument("--tf", default="H1")
    ap.add_argument("--symbols", nargs="+", default=["USDJPY"])
    ap.add_argument("--cap", type=int, default=None, help="max hold in bars (default H1 120, M5 288)")
    ap.add_argument("--slip-frac", type=float, default=0.1)
    ap.add_argument("--swap-r-per-day", type=float, default=0.02, help="rough swap cost in R per day held")
    a = ap.parse_args()
    cap = a.cap or (120 if a.tf == "H1" else 288)
    bars_per_day = 24 if a.tf == "H1" else 288
    if a.mode == "holdout":
        cells = [("USDJPY", "H1", "data/historical_old", "2023-12-31", 6000, 1000)]
    else:
        tr, te = (6000, 1000) if a.tf == "H1" else (60000, 10000)
        cells = [(s, a.tf, "data/historical_data", None, tr, te) for s in a.symbols]
    res, trades = [], []
    for sym, tf, d, end, tr, te in cells:
        for label in ("dir12", "tb"):
            t = run(sym, tf, d, end, cap if tf == a.tf else cap, tr, te, a.slip_frac, label)
            trades.append(t)
            j = judge(t, a.swap_r_per_day / bars_per_day)
            ok = j.get("trades", 0) > 0 and j["mean_r"] > 0 and j["ci_lo"] > 0 and j["folds_pos"] >= 60 \
                and j["mean_r_swap"] > 0
            res.append(dict(symbol=sym, tf=tf, label=label, verdict="PASS" if ok else "FAIL", **j))
            print(res[-1], flush=True)
    tag = f"{a.mode}_{a.tf}"
    pd.concat(trades).to_csv(f"results/resad_recipe_{tag}_trades.csv", index=False)
    summ = pd.DataFrame(res)
    summ.to_csv(f"results/resad_recipe_{tag}_summary.csv", index=False)
    print("\n" + summ.round(3).to_string(index=False))
    if a.mode == "explore":
        print("EXPLORATORY: 2024-26 data was already used by earlier tests; a PASS here is not evidence.")
