"""B2: the live decision must pass the risk gate (`should_trade`: drawdown block, watchdog cooldown) and the
`max_positions` limit before an order is sent. The gate must fail closed when the account cannot be read, and the
watchdog must count only the bot's own closing deals."""
import datetime
import threading
from types import SimpleNamespace as NS

import MetaTrader5 as mt5
import pandas as pd
import pytest

from src.config import Cfg
from src.risk import RiskManager
from src.symbol_processor import SymbolProcessor

EQUITY = 10_000.0
MAGIC = 424242
NOW = datetime.datetime(2026, 1, 5, 12, 0, tzinfo=datetime.timezone.utc)


class FakeClient:
    def __init__(self, equity=EQUITY, deals=()):
        self.equity = equity
        self.deals = list(deals)

    def symbol_info_tick(self, symbol):
        return NS(ask=1.1001, bid=1.1000)

    def symbol_info(self, symbol):
        return NS(point=1e-5, trade_contract_size=100_000, digits=5, trade_stops_level=0,
                  volume_min=0.01, volume_step=0.01, volume_max=100)

    def account_info(self):
        return None if self.equity is None else NS(equity=self.equity)

    def history_deals_get(self, since, until):
        return self.deals


class FakeEnsemble:
    ensemble_cv_auc_ = 0.6

    def predict_proba(self, X):
        return pd.Series([0.9], index=X.index)


class FakeRiskController:
    def get_params(self, symbol, context):
        return {"atr_multiplier_sl": 2.0, "atr_multiplier_tp": 2.0, "min_prob_long": 0.55, "min_prob_short": 0.55,
                "atr_idx": 0, "min_prob_long_idx": 0, "min_prob_short_idx": 0,
                "exploration_risk_mult": 1.0, "ac_multiplier": 1.0, "context_vector": None}


class RecordingExecution:
    def __init__(self):
        self.calls = []

    def trade(self, **kw):
        self.calls.append(kw)


def make_cfg(**risk):
    cfg = Cfg()
    cfg.risk.risk_per_trade = 0.01
    cfg.risk.dynamic_risk = {"enabled": False}
    cfg.risk.max_portfolio_risk = 0.5
    cfg.risk.atr_multiplier_sl = 1.0
    cfg.risk.session_filter = None
    cfg.risk.block_on_drawdown = 0.10
    cfg.data_source = "csv"
    for key, value in risk.items():
        setattr(cfg.risk, key, value)
    return cfg


def make_rm(cfg=None, client=None):
    return RiskManager(cfg or make_cfg(), client or FakeClient(), threading.Lock())


def make_sp(rm, equity=EQUITY, peak=EQUITY):
    sp = object.__new__(SymbolProcessor)
    sp.symbol, sp.dry_run = "EURUSD#", True
    sp.mt5_client, sp.risk_controller = rm.mt5_client, FakeRiskController()
    sp.risk_manager = rm
    sp.monitor = NS(current_equity=equity, peak_equity=peak)
    sp.execution = RecordingExecution()
    sp.ens_long = sp.ens_short = FakeEnsemble()
    return sp


def decide(sp):
    idx = pd.date_range("2026-01-05", periods=3, freq="5min")
    X = pd.DataFrame({"atr_14": 0.0010}, index=idx)
    data = pd.DataFrame({"close": 1.1000}, index=idx)
    sp._make_trade_decision(data, X)


def open_position(rm, ticket, symbol="GBPUSD#"):
    rm.open_positions_cache[str(ticket)] = {"symbol": symbol, "risk": 1.0, "ticket": ticket}


# ---- the gate is wired into the live decision ---------------------------------------------------------------

def test_a_trade_is_sent_when_nothing_blocks_it():
    sp = make_sp(make_rm())
    decide(sp)
    assert len(sp.execution.calls) == 1


def test_drawdown_over_the_limit_blocks_the_order():
    sp = make_sp(make_rm(), equity=8_500.0, peak=10_000.0)      # 15% down, limit 10%
    decide(sp)
    assert sp.execution.calls == []


def test_drawdown_under_the_limit_does_not_block():
    sp = make_sp(make_rm(), equity=9_500.0, peak=10_000.0)      # 5% down, limit 10%
    decide(sp)
    assert len(sp.execution.calls) == 1


@pytest.mark.parametrize("peak", [None, 0.0])
def test_a_missing_peak_does_not_crash_the_gate(peak):
    sp = make_sp(make_rm(), peak=peak)
    decide(sp)
    assert len(sp.execution.calls) == 1


def test_watchdog_cooldown_blocks_the_order():
    rm = make_rm()
    rm.watchdog_cfg.enabled = True
    rm.watchdog_cfg.max_consecutive_losses = 0              # only the cooldown matters here
    rm.cooldown_until = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)
    sp = make_sp(rm)
    decide(sp)
    assert sp.execution.calls == []


def test_a_drawdown_cooldown_started_by_the_live_gate_can_be_checked_again():
    """The drawdown branch stores an aware UTC cooldown; the caller must hand over an aware time too."""
    cfg = make_cfg()
    cfg.data_source = "mt5"
    client = FakeClient(equity=10_000.0)
    rm = make_rm(cfg, client)
    rm.watchdog_cfg.enabled = True
    rm.watchdog_cfg.max_consecutive_losses = 0
    sp = make_sp(rm)
    decide(sp)                                              # sets the peak, trades
    client.equity = 8_000.0
    decide(sp)                                              # drawdown -> cooldown
    decide(sp)                                              # cooldown check must not raise
    assert len(sp.execution.calls) == 1


def test_max_positions_blocks_the_order_when_the_limit_is_reached():
    rm = make_rm(make_cfg(max_positions=1))
    open_position(rm, 1)
    sp = make_sp(rm)
    decide(sp)
    assert sp.execution.calls == []


def test_max_positions_allows_the_order_below_the_limit():
    rm = make_rm(make_cfg(max_positions=2))
    open_position(rm, 1)
    sp = make_sp(rm)
    decide(sp)
    assert len(sp.execution.calls) == 1


# ---- the gate fails closed ----------------------------------------------------------------------------------

def mt5_rm(client):
    cfg = make_cfg()
    cfg.data_source = "mt5"
    return make_rm(cfg, client)


def test_no_account_info_blocks_trading():
    rm = mt5_rm(FakeClient(equity=None))
    assert rm.should_trade(NOW, 0.0) is False


def test_an_account_info_error_blocks_trading():
    class Broken(FakeClient):
        def account_info(self):
            raise RuntimeError("terminal not connected")

    assert mt5_rm(Broken()).should_trade(NOW, 0.0) is False


def test_a_readable_account_within_the_limit_allows_trading():
    assert mt5_rm(FakeClient(equity=EQUITY)).should_trade(NOW, 0.0) is True


# ---- the watchdog counts only the bot's own closing deals -------------------------------------------------

CLOSING, OPENING = mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_OUT + 1


def deal(t, profit, magic=MAGIC, entry=CLOSING):
    return NS(time=t, profit=profit, magic=magic, entry=entry)


def test_deposits_withdrawals_and_foreign_trades_are_not_counted_as_losses():
    deals = [
        deal(1, -5.0),                      # own closing loss
        deal(2, -50.0, magic=0, entry=OPENING),   # withdrawal / balance operation (not a closing deal)
        deal(3, -9.0, magic=424243),        # another system's loss (H1Bridge EA)
        deal(4, -5.0),                      # own closing loss
    ]
    rm = mt5_rm(FakeClient(deals=deals))
    assert rm._count_consecutive_losses(NOW) == 2


def test_an_own_win_ends_the_losing_streak():
    deals = [deal(1, -5.0), deal(2, -5.0), deal(3, 4.0)]
    rm = mt5_rm(FakeClient(deals=deals))
    assert rm._count_consecutive_losses(NOW) == 0
