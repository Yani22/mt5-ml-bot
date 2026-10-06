"""Two symbols decide on the same bar close. Reading the open risk, sizing and sending the order must be one step per
risk manager, or both pass `max_portfolio_risk` / `max_positions` on the same stale cache."""
import threading

from tests.test_trade_gate import EQUITY, RecordingExecution, decide, make_cfg, make_rm, make_sp

JOIN = 5.0


class BlockingExecution(RecordingExecution):
    """The first order stays 'in flight' until released, then lands in the cache like a real fill."""

    def __init__(self, rm, symbol):
        super().__init__()
        self.rm, self.symbol = rm, symbol
        self.in_send, self.release = threading.Event(), threading.Event()

    def trade(self, **kw):
        self.calls.append(kw)
        self.in_send.set()
        assert self.release.wait(JOIN)
        self.rm.open_positions_cache["1"] = {"symbol": self.symbol, "risk": EQUITY * 0.01, "ticket": 1}


def two_processors(**risk):
    rm = make_rm(make_cfg(max_portfolio_risk=0.015, **risk))
    a, b = make_sp(rm), make_sp(rm)
    a.symbol, b.symbol = "EURUSD#", "GBPUSD#"
    a.execution = BlockingExecution(rm, a.symbol)
    return rm, a, b


def test_the_second_symbol_sizes_against_the_first_symbols_order():
    rm, a, b = two_processors()
    ta = threading.Thread(target=decide, args=(a,))
    ta.start()
    assert a.execution.in_send.wait(JOIN)
    tb = threading.Thread(target=decide, args=(b,))
    tb.start()
    tb.join(0.3)                      # the second decision must wait for the first order to land
    a.execution.release.set()
    ta.join(JOIN)
    tb.join(JOIN)
    assert not ta.is_alive() and not tb.is_alive()
    assert len(b.execution.calls) == 1
    assert b.execution.calls[0]["total_open_risk"] == EQUITY * 0.01
    assert b.execution.calls[0]["lots"] < a.execution.calls[0]["lots"]     # 0.5% left under the 1.5% cap, not 1%


def test_max_positions_is_checked_after_the_first_order_lands():
    rm, a, b = two_processors()
    rm.risk_cfg.max_positions = 1
    ta = threading.Thread(target=decide, args=(a,))
    ta.start()
    assert a.execution.in_send.wait(JOIN)
    tb = threading.Thread(target=decide, args=(b,))
    tb.start()
    tb.join(0.3)
    a.execution.release.set()
    ta.join(JOIN)
    tb.join(JOIN)
    assert len(a.execution.calls) == 1
    assert b.execution.calls == []
    assert rm.entry_lock.acquire(timeout=1.0)         # the early return left the lock free
    rm.entry_lock.release()


def test_a_decision_that_cannot_get_the_entry_lock_in_time_sends_nothing():
    rm = make_rm()
    rm.entry_lock_timeout = 0.05
    sp = make_sp(rm)
    assert rm.entry_lock.acquire(timeout=JOIN)
    try:
        decide(sp)
    finally:
        rm.entry_lock.release()
    assert sp.execution.calls == []


def test_the_entry_lock_is_released_after_a_decision():
    rm = make_rm()
    sp = make_sp(rm)
    decide(sp)
    assert rm.entry_lock.acquire(timeout=1.0)
    rm.entry_lock.release()


def test_the_entry_lock_is_released_when_the_order_raises():
    rm = make_rm()
    sp = make_sp(rm)

    def boom(**kw):
        raise RuntimeError("send failed")

    sp.execution.trade = boom
    try:
        decide(sp)
    except RuntimeError:
        pass
    assert rm.entry_lock.acquire(timeout=1.0)
    rm.entry_lock.release()
