"""B7 (withdrawal case): a deposit or withdrawal moves equity without a trade, so a saved peak equity makes a withdrawal look
like a drawdown (and a deposit would hide a real one). The peak is rescaled in proportion to the net money paid in or out, read
from the broker's balance deals, so the drawdown FRACTION (what `block_on_drawdown` tests) is unchanged; each deal counts
once, and a failed read never lowers the peak."""
import datetime
import json
import threading
from types import SimpleNamespace as NS

import MetaTrader5 as mt5

from src.config import Cfg
from src.live_performance_monitor import LivePerformanceMonitor
from src.risk import RiskManager

NOW = datetime.datetime(2026, 1, 5, 12, 0, tzinfo=datetime.timezone.utc)
LATER = NOW + datetime.timedelta(minutes=5)
ACCOUNT = "111@Demo-Server"


def deal(ticket, profit, kind=None):
    return NS(ticket=ticket, type=mt5.DEAL_TYPE_BALANCE if kind is None else kind, profit=profit, magic=0, entry=0)


class FakeClient:
    def __init__(self, equity, deals=(), fail=False):
        self.equity, self.deals, self.fail = equity, list(deals), fail
        self.windows = []

    def account_info(self):
        return None if self.equity is None else NS(equity=self.equity)

    def history_deals_get(self, since, until):
        self.windows.append((since, until))
        if self.fail == "raise":
            raise RuntimeError("terminal gone")
        if self.fail:
            return None            # what MT5Client.history_deals_get gives on an error
        return self.deals


def make_monitor(tmp_path, equity=1_000.0):
    cfg = Cfg()
    cfg.initial_equity = equity
    cfg.monitoring.monitor_state_file = str(tmp_path / "monitor_state.json")
    monitor = LivePerformanceMonitor(cfg)
    monitor.account_id = ACCOUNT
    return monitor


def drawdown(monitor):
    return 1.0 - monitor.current_equity / monitor.peak_equity


def started(tmp_path, client, equity=1_000.0):
    """A monitor that has already taken its first look at the deal history (the start-up baseline)."""
    monitor = make_monitor(tmp_path, equity)
    monitor.apply_balance_flows(client, NOW)
    return monitor


# ---- the monitor's peak -------------------------------------------------------------------------------------

def test_a_withdrawal_does_not_show_as_a_drawdown(tmp_path):
    client = FakeClient(1_000.0)
    monitor = started(tmp_path, client)
    client.equity = 700.0
    client.deals = [deal(11, -300.0)]
    monitor.apply_balance_flows(client, LATER)
    assert monitor.peak_equity == 700.0 and monitor.current_equity == 700.0 and drawdown(monitor) == 0.0


def test_a_floating_loss_with_no_balance_deal_still_shows_a_drawdown(tmp_path):
    client = FakeClient(1_000.0)
    monitor = started(tmp_path, client)
    client.equity = 700.0
    monitor.apply_balance_flows(client, LATER)
    monitor.sync_equity(700.0)
    assert monitor.peak_equity == 1_000.0 and abs(drawdown(monitor) - 0.3) < 1e-9


def test_a_deposit_does_not_hide_a_drawdown_or_create_one(tmp_path):
    client = FakeClient(1_000.0)
    monitor = started(tmp_path, client)
    client.equity = 1_500.0
    client.deals = [deal(12, 500.0)]
    monitor.apply_balance_flows(client, LATER)
    assert monitor.peak_equity == 1_500.0 and drawdown(monitor) == 0.0


def test_a_withdrawal_during_a_drawdown_keeps_the_drawdown_fraction(tmp_path):
    client = FakeClient(1_000.0)
    monitor = started(tmp_path, client)
    monitor.sync_equity(820.0)                                   # 18% down, then half of that equity is withdrawn
    client.equity = 410.0
    client.deals = [deal(13, -410.0)]
    monitor.apply_balance_flows(client, LATER)
    assert monitor.peak_equity == 500.0 and abs(drawdown(monitor) - 0.18) < 1e-9     # not 30.5%: that would block


def test_a_deposit_during_a_drawdown_keeps_the_drawdown_fraction(tmp_path):
    client = FakeClient(1_000.0)
    monitor = started(tmp_path, client)
    monitor.sync_equity(800.0)                                   # 20% down, then the equity is doubled by a deposit
    client.equity = 1_600.0
    client.deals = [deal(14, 800.0)]
    monitor.apply_balance_flows(client, LATER)
    assert monitor.peak_equity == 2_000.0 and abs(drawdown(monitor) - 0.20) < 1e-9   # not 11%: that would lift a block


def test_an_unreadable_equity_moves_the_peak_by_the_amount(tmp_path):
    client = FakeClient(1_000.0)
    monitor = started(tmp_path, client)
    client.equity, client.deals = None, [deal(15, -300.0)]
    monitor.apply_balance_flows(client, LATER)
    assert monitor.peak_equity == 700.0


def test_credit_bonus_and_correction_count_as_money_in_or_out(tmp_path):
    client = FakeClient(1_000.0)
    monitor = started(tmp_path, client)
    client.equity = 1_060.0
    client.deals = [deal(21, 50.0, mt5.DEAL_TYPE_CREDIT), deal(22, 25.0, mt5.DEAL_TYPE_BONUS),
                    deal(23, -15.0, mt5.DEAL_TYPE_CORRECTION)]
    monitor.apply_balance_flows(client, LATER)
    assert monitor.peak_equity == 1_060.0


def test_a_trade_deal_is_not_a_balance_flow(tmp_path):
    client = FakeClient(1_000.0)
    monitor = started(tmp_path, client)
    client.equity = 900.0
    client.deals = [NS(ticket=31, type=0, profit=-100.0, magic=1, entry=1)]
    monitor.apply_balance_flows(client, LATER)
    monitor.sync_equity(900.0)
    assert monitor.peak_equity == 1_000.0


def test_the_same_deal_seen_twice_counts_once(tmp_path):
    client = FakeClient(1_000.0)
    monitor = started(tmp_path, client)
    client.equity = 700.0
    client.deals = [deal(11, -300.0)]
    monitor.apply_balance_flows(client, LATER)
    monitor.apply_balance_flows(client, LATER + datetime.timedelta(seconds=5))
    assert monitor.peak_equity == 700.0


def test_a_deal_already_in_the_history_at_start_up_is_not_applied(tmp_path):
    client = FakeClient(700.0, deals=[deal(11, -300.0)])         # withdrawn before the bot looked: the equity is post-withdrawal
    monitor = make_monitor(tmp_path, 700.0)
    assert monitor.apply_balance_flows(client, NOW) == 0.0
    assert monitor.peak_equity == 700.0
    monitor.apply_balance_flows(client, LATER)
    assert monitor.peak_equity == 700.0


def test_a_failed_history_read_leaves_the_peak_alone(tmp_path):
    client = FakeClient(1_000.0)
    monitor = started(tmp_path, client)
    client.equity = 700.0
    for failure in ("raise", "none"):
        client.fail = failure
        assert monitor.apply_balance_flows(client, LATER) == 0.0
        assert monitor.peak_equity == 1_000.0


def test_a_withdrawal_missed_by_a_failed_read_is_applied_on_the_next_good_one(tmp_path):
    client = FakeClient(1_000.0)
    monitor = started(tmp_path, client)
    client.fail, client.equity, client.deals = "none", 700.0, [deal(11, -300.0)]
    monitor.apply_balance_flows(client, LATER)
    client.fail = False
    monitor.apply_balance_flows(client, LATER + datetime.timedelta(minutes=5))
    assert monitor.peak_equity == 700.0


def test_the_history_window_reaches_back_past_the_last_look(tmp_path):
    client = FakeClient(1_000.0)
    monitor = started(tmp_path, client)
    monitor.apply_balance_flows(client, LATER)
    since, until = client.windows[-1]
    assert since < NOW and until > LATER                         # server clock offset and late-posted deals


def test_a_peak_is_never_pushed_below_zero(tmp_path):
    client = FakeClient(100.0)
    monitor = started(tmp_path, client, 100.0)
    client.deals = [deal(41, -500.0)]
    monitor.apply_balance_flows(client, LATER)
    assert monitor.peak_equity >= monitor.current_equity >= 0


# ---- the state file -----------------------------------------------------------------------------------------

def test_the_state_file_keeps_the_deals_already_counted(tmp_path):
    client = FakeClient(1_000.0)
    monitor = started(tmp_path, client)
    client.equity, client.deals = 700.0, [deal(11, -300.0)]
    monitor.apply_balance_flows(client, LATER)
    monitor.save_state()
    restarted = make_monitor(tmp_path, 700.0)
    restarted.load_state()
    restarted.apply_balance_flows(client, LATER + datetime.timedelta(minutes=1))
    assert restarted.peak_equity == 700.0                        # not 400: the withdrawal is not taken off again


def test_a_withdrawal_made_while_the_bot_was_down_is_applied_at_the_next_start(tmp_path):
    client = FakeClient(1_000.0)
    monitor = started(tmp_path, client)
    monitor.save_state()
    client.equity, client.deals = 700.0, [deal(11, -300.0)]
    restarted = make_monitor(tmp_path, 700.0)
    restarted.load_state()
    restarted.apply_balance_flows(client, LATER + datetime.timedelta(hours=6))
    assert restarted.peak_equity == 700.0 and drawdown(restarted) == 0.0


def test_a_state_file_with_no_marker_applies_nothing_from_old_history(tmp_path):
    old = make_monitor(tmp_path, 1_000.0)
    old.save_state()
    path = tmp_path / "monitor_state.json"
    state = json.loads(path.read_text())
    state.pop("balance_flows", None)                             # a file written before this fix
    path.write_text(json.dumps(state))
    client = FakeClient(700.0, deals=[deal(11, -300.0)])
    restarted = make_monitor(tmp_path, 700.0)
    restarted.load_state()
    restarted.apply_balance_flows(client, NOW)
    restarted.apply_balance_flows(client, LATER)
    assert restarted.peak_equity == 1_000.0                      # the old peak stays; no replay, no new shift


# ---- the risk manager's own peak (the gate in mt5 data mode) ------------------------------------------------

def make_rm(client):
    cfg = Cfg()
    cfg.data_source = "mt5"
    cfg.risk.block_on_drawdown = 0.20
    cfg.risk.session_filter = None
    cfg.watchdog.enabled = False
    return RiskManager(cfg, client, threading.Lock())


def test_the_gate_does_not_block_after_a_withdrawal():
    client = FakeClient(10_000.0)
    rm = make_rm(client)
    assert rm.should_trade(NOW, 0.0)
    client.equity, client.deals = 5_000.0, [deal(11, -5_000.0)]  # half of the account paid out
    assert rm.should_trade(LATER, 0.0)


def test_the_gate_still_blocks_a_real_drawdown():
    client = FakeClient(10_000.0)
    rm = make_rm(client)
    assert rm.should_trade(NOW, 0.0)
    client.equity = 7_000.0                                      # 30% lost, no balance deal
    assert not rm.should_trade(LATER, 0.0)


def test_the_gate_counts_a_withdrawal_once():
    client = FakeClient(10_000.0)
    rm = make_rm(client)
    rm.should_trade(NOW, 0.0)
    client.equity, client.deals = 8_000.0, [deal(11, -2_000.0)]
    rm.should_trade(LATER, 0.0)
    rm.should_trade(LATER + datetime.timedelta(seconds=5), 0.0)
    assert rm.equity_peak == 8_000.0


def test_the_gate_stays_blocked_after_a_deposit_in_a_breached_drawdown():
    client = FakeClient(10_000.0)
    rm = make_rm(client)
    rm.should_trade(NOW, 0.0)
    client.equity = 7_900.0                                      # 21% down, over the 20% limit
    assert not rm.should_trade(LATER, 0.0)
    client.equity, client.deals = 15_800.0, [deal(11, 7_900.0)]  # doubled by a deposit: still 21% down
    assert not rm.should_trade(LATER + datetime.timedelta(minutes=1), 0.0)


def test_the_gate_stays_open_after_a_withdrawal_in_a_small_drawdown():
    client = FakeClient(10_000.0)
    rm = make_rm(client)
    rm.should_trade(NOW, 0.0)
    client.equity = 8_200.0                                      # 18% down, under the limit
    assert rm.should_trade(LATER, 0.0)
    client.equity, client.deals = 4_100.0, [deal(11, -4_100.0)]  # half paid out: still 18% down
    assert rm.should_trade(LATER + datetime.timedelta(minutes=1), 0.0)


def test_a_withdrawal_seen_while_the_account_cannot_be_read_is_applied_once_it_can():
    client = FakeClient(10_000.0)
    rm = make_rm(client)
    rm.should_trade(NOW, 0.0)
    client.equity, client.deals = None, [deal(11, -5_000.0)]
    assert not rm.should_trade(LATER, 0.0)                       # fails closed
    client.equity = 5_000.0
    assert rm.should_trade(LATER + datetime.timedelta(minutes=1), 0.0)
    assert rm.equity_peak == 5_000.0


def test_the_gate_leaves_the_peak_alone_when_the_history_read_fails():
    client = FakeClient(10_000.0)
    rm = make_rm(client)
    rm.should_trade(NOW, 0.0)
    client.fail, client.equity = "raise", 5_000.0
    assert not rm.should_trade(LATER, 0.0)                       # 50% down, nothing known about a withdrawal: stay blocked
    assert rm.equity_peak == 10_000.0


def test_main_looks_at_balance_deals_at_start_up_and_in_the_loop():
    """`main.py` only compiles under the test stub, so this checks the wiring in its source."""
    source = open("main.py").read()
    assert source.count("apply_balance_flows(") == 2
    start_up = source.index("apply_balance_flows(", source.index("live_monitor.load_state()"))
    assert source.index("live_monitor.load_state()") < start_up < source.index("live_monitor.sync_equity(initial_equity)")
    loop = source.index("apply_balance_flows(")
    assert source.index("def _process_closed_trades") < loop < source.index("exe.reconcile_open_positions_with_mt5()")
