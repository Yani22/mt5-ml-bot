"""B15: the open-position cache is only read or copied while holding `cache_lock`, and nothing that can block on the
network (the Telegram notifier) runs while that lock is held. The notifier has a finite timeout."""
import copy
from types import SimpleNamespace as NS

import pytest
import requests

from src.config import Cfg
from src.notifier import TelegramNotifier
from src.risk import RiskManager
from test_execution_dry_run import FakeClient as ExecClient, make as make_execution
from test_trade_gate import decide, make_cfg, make_rm, make_sp


@pytest.fixture(autouse=True)
def _tmp_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # Execution creates results/ relative to the working directory


class LockWatchingDict(dict):
    """A cache that records whether the lock was held on every read."""

    def __init__(self, lock, *a, **k):
        super().__init__(*a, **k)
        self.lock, self.seen = lock, []

    def _note(self):
        self.seen.append(self.lock.locked())

    def values(self):
        self._note()
        return super().values()

    def items(self):
        self._note()
        return super().items()

    def __iter__(self):
        self._note()
        return super().__iter__()

    def __deepcopy__(self, memo):
        self._note()
        return copy.deepcopy(dict(dict.items(self)), memo)


# ---- notifier timeout ---------------------------------------------------------------------------------------------

def make_notifier():
    cfg = Cfg()
    cfg.monitoring.telegram_bot_token, cfg.monitoring.telegram_chat_id = "123:abc", "42"
    return TelegramNotifier(cfg)


def test_the_telegram_post_has_a_finite_timeout(monkeypatch):
    seen = {}

    def fake_post(url, **kw):
        seen.update(kw)
        return NS(raise_for_status=lambda: None)

    monkeypatch.setattr("src.notifier.requests.post", fake_post)
    make_notifier().send_message("hello")
    timeout = seen["timeout"]
    assert timeout is not None and all(0 < t <= 30 for t in (timeout if isinstance(timeout, tuple) else (timeout,)))


def test_a_telegram_timeout_is_swallowed(monkeypatch):
    def fake_post(url, **kw):
        raise requests.exceptions.Timeout("slow")

    monkeypatch.setattr("src.notifier.requests.post", fake_post)
    make_notifier().send_message("hello")          # must not raise


# ---- no notifier call under the lock -------------------------------------------------------------------------------

class WatchingNotifier:
    def __init__(self, lock):
        self.lock, self.locked_at_send = lock, []

    def send_message(self, message, level="INFO"):
        self.locked_at_send.append(self.lock.locked())


class BrokenClient(ExecClient):
    def positions_get(self, *a, **k):
        raise RuntimeError("terminal gone")


def test_a_failed_live_reconcile_notifies_after_the_cache_lock_is_released():
    ex, rm = make_execution(dry_run=False, client=BrokenClient())
    ex.notifier = WatchingNotifier(rm.cache_lock)
    assert ex.reconcile_open_positions_with_mt5() == []
    assert ex.notifier.locked_at_send == [False]


# ---- cache reads under the lock ---------------------------------------------------------------------------------------

def test_total_open_risk_is_summed_under_the_lock():
    rm = make_rm()
    rm.open_positions_cache = LockWatchingDict(rm.cache_lock, {"1": {"risk": 40.0}, "2": {"risk": 2.5}})
    assert rm.total_open_risk() == pytest.approx(42.5)
    assert rm.open_positions_cache.seen and all(rm.open_positions_cache.seen)


def test_the_live_decision_reads_the_cache_under_the_lock():
    rm = make_rm()
    rm.open_positions_cache = LockWatchingDict(rm.cache_lock, {"9": {"symbol": "GBPUSD#", "risk": 1.0, "ticket": 9}})
    sp = make_sp(rm)
    decide(sp)
    assert rm.open_positions_cache.seen and all(rm.open_positions_cache.seen)


def test_the_snapshot_is_a_deep_copy_taken_under_the_lock():
    rm = make_rm()
    rm.open_positions_cache = LockWatchingDict(rm.cache_lock, {"1": {"risk": 40.0, "context_vector": [1.0]}})
    snap = rm.cache_snapshot()
    assert all(rm.open_positions_cache.seen) and snap == {"1": {"risk": 40.0, "context_vector": [1.0]}}
    snap["1"]["context_vector"].append(2.0)
    assert rm.open_positions_cache["1"]["context_vector"] == [1.0]
