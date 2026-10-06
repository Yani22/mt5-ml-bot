"""AUDIT fix 88: `backtesting.early_stop` ends a symbol's replay when its trades are demonstrably losing, so a hopeless variant does not use 40 minutes.
Rule (pre-registered in AUDIT.md): at least `early_stop_min_trades` trades entered before `early_stop_until`, at least `early_stop_min_blocks` retrain blocks
reached, and the upper end of the (1 - `early_stop_alpha`) two-sided t-interval of the mean R below 0. Off by default; never inside an Optuna trial; never
after `early_stop_until` (the holdout stays unread)."""
import pandas as pd
import pytest

from src.config import Cfg, ConfigError
from tests.test_backtest_walkforward import N, TRAIN, build, make_bt, run


def armed(bt, results, **kw):
    cfg = bt.cfg.backtesting
    cfg.early_stop, cfg.early_stop_min_trades, cfg.early_stop_min_blocks = True, 3, 1
    cfg.early_stop_alpha, cfg.early_stop_until = 0.001, None
    for k, v in kw.items():
        setattr(cfg, k, v)
    bt._closed_R = {"USDJPY#": list(results)}


def losers(idx, n=5, r=-1.0):
    return [(idx[0] - pd.Timedelta(hours=k + 1), r) for k in range(n)]


def test_it_is_off_by_default():
    assert Cfg().backtesting.early_stop is False


def test_a_losing_record_stops_the_replay_and_the_results_are_still_written(monkeypatch, tmp_path):
    bt = make_bt(monkeypatch, tmp_path)
    frame, models, idx = build(p={i: 0.9 for i in range(40, N)})
    armed(bt, losers(idx))
    run(bt, frame, models)
    assert bt.signals == 0 and "n=5" in bt.stopped_early["USDJPY#"]
    trades, equity = bt._generate_results()
    assert len(trades) == 0 and (tmp_path / "results").exists()


def test_without_the_switch_the_replay_goes_on(monkeypatch, tmp_path):
    bt = make_bt(monkeypatch, tmp_path)
    frame, models, idx = build(p={40: 0.9})
    armed(bt, losers(idx), early_stop=False)
    assert len(run(bt, frame, models)) == 1 and not bt.stopped_early


def test_too_few_trades_do_not_stop_it(monkeypatch, tmp_path):
    bt = make_bt(monkeypatch, tmp_path)
    frame, models, idx = build(p={40: 0.9})
    armed(bt, losers(idx, n=2), early_stop_min_trades=3)
    assert len(run(bt, frame, models)) == 1 and not bt.stopped_early


def test_too_few_retrain_blocks_do_not_stop_it(monkeypatch, tmp_path):
    bt = make_bt(monkeypatch, tmp_path)
    frame, models, idx = build(p={40: 0.9})
    armed(bt, losers(idx), early_stop_min_blocks=99)
    assert len(run(bt, frame, models)) == 1 and not bt.stopped_early


def test_a_mean_above_zero_or_an_interval_reaching_zero_does_not_stop_it(monkeypatch, tmp_path):
    for results in ([(None, 1.0)] * 5, [(None, -1.0), (None, 3.0), (None, -2.0), (None, 2.5), (None, -0.5)]):
        bt = make_bt(monkeypatch, tmp_path)
        frame, models, idx = build(p={40: 0.9})
        armed(bt, [(idx[0] - pd.Timedelta(hours=k + 1), r) for k, (_, r) in enumerate(results)])
        assert len(run(bt, frame, models)) == 1 and not bt.stopped_early


def test_it_is_not_checked_from_the_cutoff_on_and_trades_after_it_are_not_counted(monkeypatch, tmp_path):
    bt = make_bt(monkeypatch, tmp_path)
    frame, models, idx = build(p={40: 0.9})
    armed(bt, losers(idx), early_stop_until=str(idx[0].date()))        # the cutoff is before every bar of the run
    assert len(run(bt, frame, models)) == 1 and not bt.stopped_early
    bt = make_bt(monkeypatch, tmp_path)
    frame, models, idx = build(p={i: 0.9 for i in range(40, N)})
    late = [(idx[0] + pd.Timedelta(days=31 + k), -1.0) for k in range(5)]
    armed(bt, late, early_stop_until=str((idx[0] + pd.Timedelta(days=30)).date()))   # the losers entered after the cutoff
    run(bt, frame, models)
    assert not bt.stopped_early


def test_it_is_never_applied_inside_an_optuna_trial(monkeypatch, tmp_path):
    bt = make_bt(monkeypatch, tmp_path)
    frame, models, idx = build(p={40: 0.9})
    armed(bt, losers(idx))
    assert bt._early_stop_reason("USDJPY#", idx[TRAIN], 1, trial=object()) is None
    assert bt._early_stop_reason("USDJPY#", idx[TRAIN], 1, trial=None) is not None


def test_each_closed_trade_records_its_r_at_the_placed_stop(monkeypatch, tmp_path):
    from tests.test_backtest_costs import make_bt as costs_bt, open_long, at
    bt = costs_bt()
    pos = open_long(bt, 1.1000, 1.0990, 1.1020)
    bt._update_positions("EURUSD#", at(1.0990))
    assert len(bt._closed_R["EURUSD#"]) == 1 and bt._closed_R["EURUSD#"][0][1] == pytest.approx(bt.positions[0].pnl / bt._open_stop_money(pos))


def test_another_symbols_losing_record_does_not_stop_this_one(monkeypatch, tmp_path):
    bt = make_bt(monkeypatch, tmp_path)
    frame, models, idx = build(p={40: 0.9})
    armed(bt, [])
    bt._closed_R = {"EURUSD#": losers(idx)}                    # symbol A lost; symbol B (USDJPY#) has no trades yet
    assert len(run(bt, frame, models)) == 1 and not bt.stopped_early
    assert bt._early_stop_reason("USDJPY#", idx[TRAIN], 1) is None and bt._early_stop_reason("EURUSD#", idx[TRAIN], 1) is not None


# ---- config ----------------------------------------------------------------------------------------------------------

def load(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return Cfg.from_yaml(str(path))


def test_the_early_stop_keys_load(tmp_path):
    cfg = load(tmp_path, 'backtesting:\n  early_stop: true\n  early_stop_min_trades: 120\n  early_stop_min_blocks: 6\n  early_stop_alpha: 0.001\n  early_stop_until: "2026-04-01"\n')
    b = cfg.backtesting
    assert (b.early_stop, b.early_stop_min_trades, b.early_stop_min_blocks, b.early_stop_alpha, b.early_stop_until) == (True, 120, 6, 0.001, "2026-04-01")


def test_a_wrong_type_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match=r"backtesting\.early_stop"):
        load(tmp_path, 'backtesting:\n  early_stop: "yes"\n')
