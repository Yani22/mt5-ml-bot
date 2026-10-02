"""USDJPY H1 24-bar strategy: the live decision rule, frozen to what scripts/walkforward_h1*.py backtested.

Spec (do not tune here; any change means the forward test is no longer the backtested strategy):
  * features: src.h1_features.build_h1 on the fixed-origin H1 history (never a sliding window: OBV/EMA depend on it)
  * models: LightGBM long ("up after 24 bars") and short ("down after 24 bars"), TRAIN_ROWS labelled rows, the last
    HORIZON rows purged, retrained at fixed fold boundaries every RETRAIN_EVERY rows, random_state=SEED
  * signal at the close of bar t: long if pl - bl + 0.5 > 0.5 + OFFSET and pl > ps (short mirrored), bl/bs = train
    base rates; entry at the next open; hard stop STOP_ATR * ATR14(t) from the fill; exit after HORIZON bars
Models are retrained from the cache on start-up (deterministic) instead of unpickled.
"""
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from lightgbm import LGBMClassifier

from src.h1_features import build_h1

HORIZON = 24
OFFSET = 0.06
STOP_ATR = 2.0
TRAIN_ROWS = 6000
RETRAIN_EVERY = 1000
SEED = 0
LGBM_PARAMS = dict(n_estimators=200, learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                   num_leaves=15, min_child_samples=50, n_jobs=4, verbose=-1, random_state=SEED)


@dataclass
class Decision:
    bar_time: pd.Timestamp  # open time of the last closed bar the decision is based on
    side: int  # 1 long, -1 short, 0 none
    p_long: float
    p_short: float
    atr: float
    stop_distance: float  # STOP_ATR * atr, in price units
    reason: str = ""


def fold_start(row: int) -> int:
    """First test row of the walk-forward fold that row `row` belongs to (rows count from the first feature row)."""
    return TRAIN_ROWS + RETRAIN_EVERY * ((row - TRAIN_ROWS) // RETRAIN_EVERY)


def signal_side(pl: float, ps: float, bl: float, bs: float, offset: float = OFFSET) -> int:
    ql, qs = pl - bl + 0.5, ps - bs + 0.5
    th = 0.5 + offset
    if ql > th and ql > qs:
        return 1
    if qs > th and qs > ql:
        return -1
    return 0


class H1Strategy:
    def __init__(self):
        self._fold: Optional[int] = None
        self._ml = self._ms = None
        self._bl = self._bs = 0.5

    def _fit_fold(self, h1: pd.DataFrame, X: pd.DataFrame, s: int) -> None:
        fwd = (h1["close"].shift(-HORIZON) - h1["close"]).dropna()
        common = X.index.intersection(fwd.index)
        Xc, fwd = X.loc[common], fwd.loc[common]
        tr = slice(s - TRAIN_ROWS, s - HORIZON)  # purge: labels of the last HORIZON rows look past the fold start
        yl, ys = (fwd > 0).astype(int), (fwd < 0).astype(int)
        self._ml = LGBMClassifier(**LGBM_PARAMS).fit(Xc.iloc[tr], yl.iloc[tr])
        self._ms = LGBMClassifier(**LGBM_PARAMS).fit(Xc.iloc[tr], ys.iloc[tr])
        self._bl, self._bs = float(yl.iloc[tr].mean()), float(ys.iloc[tr].mean())
        self._fold = s

    def decide(self, h1: pd.DataFrame) -> Decision:
        """Decision for the LAST bar of `h1` (which must be a fully closed bar and the history must start at the
        fixed origin)."""
        X = build_h1(h1)
        r = len(X) - 1
        t = X.index[r]
        if r < TRAIN_ROWS:
            return Decision(t, 0, float("nan"), float("nan"), float("nan"), float("nan"), "not enough history")
        s = fold_start(r)
        if s != self._fold:
            self._fit_fold(h1, X, s)
        row = X.iloc[[r]]
        pl = float(self._ml.predict_proba(row)[0, 1])
        ps = float(self._ms.predict_proba(row)[0, 1])
        atr = float(row["atr_14"].iloc[0])
        side = signal_side(pl, ps, self._bl, self._bs)
        return Decision(t, side, pl, ps, atr, STOP_ATR * atr, "")
