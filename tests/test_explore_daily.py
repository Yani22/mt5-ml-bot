"""Look-ahead and cost checks for the daily trend-following exploration (synthetic data only)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import explore_daily as ed  # noqa: E402


def _daily(n=900, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n)
    c = 1.2 + np.cumsum(rng.normal(0, 0.004, n))
    return pd.DataFrame({"open": c - rng.normal(0, 0.001, n), "high": c + 0.003, "low": c - 0.003, "close": c}, index=idx)


def test_positions_do_not_use_future_days():
    d = _daily()
    full_t = ed.tsmom_positions(d.close, 6)
    full_d = ed.donchian_positions(d.high, d.low, d.close, 50)
    for cut in range(300, 880, 41):
        part = d.iloc[:cut]
        assert np.array_equal(full_t.iloc[cut - 3:cut].to_numpy(), ed.tsmom_positions(part.close, 6).iloc[cut - 3:cut].to_numpy())
        assert np.array_equal(full_d.iloc[cut - 3:cut].to_numpy(),
                              ed.donchian_positions(part.high, part.low, part.close, 50).iloc[cut - 3:cut].to_numpy())


def test_tsmom_only_changes_at_month_start():
    d = _daily()
    pos = ed.tsmom_positions(d.close, 3)
    changes = pos.index[pos.diff().abs().fillna(0) > 0]
    assert len(changes) > 5
    # a change can only appear on the first trading day after a month-end close
    assert all(d.index[d.index.get_loc(c) - 1].month != c.month for c in changes)


def test_cost_charged_per_position_change():
    idx = pd.bdate_range("2020-01-01", periods=6)
    o = pd.Series(1.0, index=idx)  # flat prices: gross pnl is zero, only costs show
    pos = pd.Series([0, 1, 1, 0, -1, -1], index=idx, dtype=float)
    pnl = ed.daily_pnl_pct(o, pos, cost_per_change=0.001)
    # position changes: 0->1 (1), 1->0 (1), 0->-1 (1) => three legs, each half the round-trip cost
    assert np.isclose(pnl.sum(), -3 * 0.0005)
    assert np.isclose(ed.pair_cost_pct(0.0001, 1.0, 0.0001), (2 * 0.0001 + 2 * 0.1 * 0.0001) / 1.0)
