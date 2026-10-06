import pytest

from src.backtest_fills import bar_spread, entry_price, exit_hit, next_stop


@pytest.mark.parametrize("raw", [0, 0.0, -1.0, float("nan"), None, "x"])
def test_a_missing_spread_uses_the_fallback(raw):
    assert bar_spread(raw, 0.0002) == 0.0002


def test_a_real_spread_is_kept():
    assert bar_spread(0.0001, 0.0002) == 0.0001


def test_a_long_pays_the_ask_and_a_short_the_bid():
    assert entry_price("long", 1.1000, 0.0002) == pytest.approx(1.1002)
    assert entry_price("short", 1.1000, 0.0002) == 1.1000


def test_a_long_stop_and_target_use_the_bid_range():
    assert exit_hit("long", 1.0990, 1.1020, 1.1005, 1.1010, 1.0995, 0.0002) is None
    assert exit_hit("long", 1.0990, 1.1020, 1.1005, 1.1010, 1.0989, 0.0002) == (1.0990, "Stop Loss")
    assert exit_hit("long", 1.0990, 1.1020, 1.1005, 1.1021, 1.0995, 0.0002) == (1.1020, "Take Profit")


def test_both_touched_exits_at_the_stop():
    assert exit_hit("long", 1.0990, 1.1020, 1.1005, 1.1030, 1.0980, 0.0002) == (1.0990, "Stop Loss")
    assert exit_hit("short", 1.1010, 1.0980, 1.1000, 1.1020, 1.0970, 0.0002)[1] == "Stop Loss"


def test_a_gap_past_the_stop_fills_at_the_open():
    assert exit_hit("long", 1.0990, 1.1020, 1.0950, 1.0960, 1.0940, 0.0002) == (1.0950, "Stop Loss")
    price, reason = exit_hit("short", 1.1010, 1.0980, 1.1050, 1.1060, 1.1040, 0.0002)
    assert reason == "Stop Loss" and price == pytest.approx(1.1052)


def test_a_short_stop_triggers_on_the_ask_so_the_exit_bars_spread_decides():
    # high 1.1008 + spread 0.0004 = 1.1012 >= stop 1.1010: stopped, filled at the stop
    assert exit_hit("short", 1.1010, 1.0980, 1.1000, 1.1008, 1.0995, 0.0004) == (1.1010, "Stop Loss")
    # the same bar with a 0.0001 spread does not reach the stop
    assert exit_hit("short", 1.1010, 1.0980, 1.1000, 1.1008, 1.0995, 0.0001) is None


def test_a_short_target_triggers_on_low_plus_spread():
    assert exit_hit("short", 1.1010, 1.0980, 1.0990, 1.0995, 1.0977, 0.0002) == (1.0980, "Take Profit")
    assert exit_hit("short", 1.1010, 1.0980, 1.0990, 1.0995, 1.0979, 0.0002) is None  # 1.0979 + 0.0002 is above the target


def test_breakeven_needs_one_r_and_moves_the_stop_to_entry():
    kw = dict(direction="long", entry=1.1000, sl=1.0990, atr_entry=0.0010, placed_mult=1.0, spread=0.0002, atr_now=0.0010,
              breakeven=True, trailing_mult=0.0)
    assert next_stop(close=1.1005, **kw) == 1.0990
    assert next_stop(close=1.1010, **kw) == 1.1000


def test_trailing_follows_the_close_and_only_tightens():
    kw = dict(direction="long", entry=1.1000, sl=1.1000, atr_entry=0.0010, placed_mult=1.0, spread=0.0002, atr_now=0.0010,
              breakeven=True, trailing_mult=1.0)
    assert next_stop(close=1.1030, **kw) == pytest.approx(1.1020)
    assert next_stop(close=1.1005, **kw) == 1.1000


def test_a_short_uses_close_plus_spread_as_its_exit_price():
    kw = dict(direction="short", entry=1.1000, sl=1.1010, atr_entry=0.0010, placed_mult=1.0, atr_now=0.0010, breakeven=True,
              trailing_mult=0.0, spread=0.0002)
    assert next_stop(close=1.0991, **kw) == 1.1010   # ask 1.0993: 7 pips in profit, not yet 1R (10)
    assert next_stop(close=1.0987, **kw) == 1.1000   # ask 1.0989: 11 pips, 1R reached


def test_no_placed_multiple_means_no_breakeven_and_no_trail_start():
    assert next_stop("long", 1.1, 1.099, 0.001, None, 1.2, 0.0, 0.001, True, 1.0) == 1.099
