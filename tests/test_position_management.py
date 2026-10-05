"""B2b: live breakeven and trailing stop management.

Breakeven at +1R, where 1R is the stop actually placed (`sl_atr_mult` x entry ATR, fix 8), then a trailing stop that only
starts once the trade has reached +1R or the stop is already at entry. Never in dry-run, never on simulated entries,
and every SLTP request names the symbol."""
import threading
from types import SimpleNamespace as NS

import MetaTrader5 as mt5
import pandas as pd
import pytest

from src.config import Cfg
from src.risk import RiskManager
from src.symbol_processor import SymbolProcessor

SYMBOL = "EURUSD#"
ENTRY, ATR = 1.1000, 0.0010


class FakeClient:
    def __init__(self, bid, ask=None, retcode=None):
        self.bid = bid
        self.ask = bid + 0.0001 if ask is None else ask
        self.retcode = mt5.TRADE_RETCODE_DONE if retcode is None else retcode
        self.requests = []

    def symbol_info_tick(self, symbol):
        return NS(bid=self.bid, ask=self.ask)

    def symbol_info(self, symbol):
        return NS(digits=5, point=1e-5, trade_contract_size=100_000)

    def order_send(self, request):
        self.requests.append(request)
        return NS(retcode=self.retcode, comment="")


def make_rm(client, **risk):
    cfg = Cfg()
    cfg.data_source = "mt5"
    cfg.risk.breakeven_at_1R = True
    cfg.risk.trailing_atr_mult = 1.0
    for key, value in risk.items():
        setattr(cfg.risk, key, value)
    return RiskManager(cfg, client, threading.Lock())


def long_pos(ticket=42, sl=1.0990, sl_atr_mult=1.0, **extra):
    pos = {"ticket": ticket, "symbol": SYMBOL, "direction": "long", "entry_price": ENTRY, "atr": ATR,
           "sl": sl, "tp": 1.1012, "sl_atr_mult": sl_atr_mult, "risk": 1.0}
    pos.update(extra)
    return pos


def short_pos(ticket=43, sl=1.1010, sl_atr_mult=1.0, **extra):
    pos = long_pos(ticket, sl, sl_atr_mult, direction="short", tp=1.0988)
    pos.update(extra)
    return pos


def manage(rm, *positions, atr=ATR):
    for pos in positions:
        rm.open_positions_cache[pos["ticket"]] = pos
    rm.manage_open_positions(SYMBOL, atr)


# ---- breakeven: 1R is the stop that was placed -------------------------------------------------------------

def test_long_moves_to_breakeven_at_one_r_and_names_the_symbol():
    client = FakeClient(bid=1.1010)                         # +1R for a 1x ATR stop
    rm = make_rm(client)
    manage(rm, long_pos())
    (req,) = client.requests
    assert req == {"action": mt5.TRADE_ACTION_SLTP, "symbol": SYMBOL, "position": 42, "sl": pytest.approx(ENTRY),
                   "tp": 1.1012}


def test_short_moves_to_breakeven_at_one_r():
    client = FakeClient(bid=1.0989, ask=1.0990)             # exactly +1R for a 1x ATR stop
    rm = make_rm(client)
    manage(rm, short_pos())
    (req,) = client.requests
    assert req["sl"] == pytest.approx(ENTRY) and req["symbol"] == SYMBOL


def test_short_past_one_r_goes_straight_to_the_trailing_stop():
    client = FakeClient(bid=1.0987, ask=1.0988)             # +1.2R: breakeven, then ask + 1 x ATR is tighter
    rm = make_rm(client)
    manage(rm, short_pos())
    (req,) = client.requests
    assert req["sl"] == pytest.approx(1.0998)


def test_one_r_is_the_stop_placed_not_the_config_multiple():
    """A 2x ATR stop has not reached +1R at +1.2 ATR; the config multiple (1.0) would say it has."""
    client = FakeClient(bid=1.1012)
    rm = make_rm(client)
    manage(rm, long_pos(sl=1.0980, sl_atr_mult=2.0))
    assert client.requests == []


def test_an_entry_without_the_placed_stop_gets_no_breakeven():
    client = FakeClient(bid=1.1012)
    rm = make_rm(client)
    manage(rm, long_pos(sl_atr_mult=None))
    assert client.requests == []


def test_a_wide_stop_is_not_pulled_in_before_the_trade_is_in_profit():
    client = FakeClient(bid=ENTRY)                          # at entry; trailing 1x ATR would give 1.0990
    rm = make_rm(client)
    manage(rm, long_pos(sl=1.0980, sl_atr_mult=2.0))
    assert client.requests == []


# ---- trailing starts at +1R / once the stop is at entry -------------------------------------------------------

def test_trailing_follows_price_once_the_stop_is_at_entry():
    client = FakeClient(bid=1.1030)
    rm = make_rm(client)
    manage(rm, long_pos(sl=ENTRY))
    (req,) = client.requests
    assert req["sl"] == pytest.approx(1.1020)               # bid - 1 x ATR


def test_trailing_never_loosens_the_stop():
    client = FakeClient(bid=1.1025)                         # bid - ATR = 1.1015, below the current stop
    rm = make_rm(client)
    manage(rm, long_pos(sl=1.1020))
    assert client.requests == []


def test_a_stop_that_rounds_back_to_its_current_price_is_not_sent():
    client = FakeClient(bid=1.1030004)                      # bid - ATR = 1.1020004, rounds to the current 1.10200
    rm = make_rm(client)
    manage(rm, long_pos(sl=1.1020))
    assert client.requests == []


def test_short_trailing_follows_price_down_once_the_stop_is_at_entry():
    client = FakeClient(bid=1.0969, ask=1.0970)
    rm = make_rm(client)
    manage(rm, short_pos(sl=ENTRY))
    (req,) = client.requests
    assert req["sl"] == pytest.approx(1.0980)               # ask + 1 x ATR


def test_trailing_alone_starts_at_one_r_when_breakeven_is_off():
    client = FakeClient(bid=1.1030)                         # +3R
    rm = make_rm(client, breakeven_at_1R=False)
    manage(rm, long_pos())
    (req,) = client.requests
    assert req["sl"] == pytest.approx(1.1020)


def test_trailing_alone_waits_for_one_r_when_breakeven_is_off():
    client = FakeClient(bid=1.1005)                         # +0.5R
    rm = make_rm(client, breakeven_at_1R=False)
    manage(rm, long_pos(sl=1.0980, sl_atr_mult=2.0))
    assert client.requests == []


# ---- safety ---------------------------------------------------------------------------------------------------

def test_simulated_entries_never_reach_the_broker():
    client = FakeClient(bid=1.1030)
    rm = make_rm(client)
    manage(rm, long_pos(dry_run=True))
    assert client.requests == []


def test_a_successful_change_is_cached_and_not_sent_again():
    client = FakeClient(bid=1.1010)
    rm = make_rm(client)
    manage(rm, long_pos())
    rm.manage_open_positions(SYMBOL, ATR)
    assert len(client.requests) == 1
    assert rm.open_positions_cache[42]["sl"] == pytest.approx(ENTRY)


def test_a_rejected_change_leaves_the_cached_stop_alone():
    client = FakeClient(bid=1.1010, retcode=10016)          # TRADE_RETCODE_INVALID_STOPS
    rm = make_rm(client)
    manage(rm, long_pos())
    assert rm.open_positions_cache[42]["sl"] == 1.0990


def test_the_cache_is_updated_through_its_own_key():
    client = FakeClient(bid=1.1010)
    rm = make_rm(client)
    rm.open_positions_cache["42"] = long_pos()              # key type differs from the ticket field
    rm.manage_open_positions(SYMBOL, ATR)
    assert rm.open_positions_cache["42"]["sl"] == pytest.approx(ENTRY)


def test_other_symbols_are_left_alone():
    client = FakeClient(bid=1.1010)
    rm = make_rm(client)
    manage(rm, long_pos(symbol="GBPUSD#"))
    assert client.requests == []


# ---- the live loop calls it, but never in dry-run -------------------------------------------------------------

class RecordingRiskManager:
    def __init__(self):
        self.calls = []

    def manage_open_positions(self, symbol, current_atr):
        self.calls.append((symbol, current_atr))


def make_sp(dry_run):
    sp = object.__new__(SymbolProcessor)
    sp.symbol, sp.dry_run = SYMBOL, dry_run
    sp.risk_manager = RecordingRiskManager()
    return sp


def atr_frame():
    idx = pd.date_range("2026-01-05", periods=3, freq="5min")
    return pd.DataFrame({"atr_14": [0.0030, 0.0020, 0.0010]}, index=idx)


def test_live_processor_manages_positions_with_the_last_closed_bar_atr():
    sp = make_sp(dry_run=False)
    sp._manage_positions(atr_frame())
    assert sp.risk_manager.calls == [(SYMBOL, pytest.approx(0.0010))]    # the last closed bar (B12), same bar as the decision


def test_dry_run_processor_never_manages_positions():
    sp = make_sp(dry_run=True)
    sp._manage_positions(atr_frame())
    assert sp.risk_manager.calls == []


def test_a_failure_in_position_management_does_not_propagate():
    sp = make_sp(dry_run=False)

    def boom(symbol, atr):
        raise RuntimeError("terminal not connected")

    sp.risk_manager.manage_open_positions = boom
    sp._manage_positions(atr_frame())                       # must not raise
