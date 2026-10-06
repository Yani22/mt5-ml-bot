import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src import retraining


def _ens(auc):
    return SimpleNamespace(ensemble_cv_auc_=auc)


def _cfg(min_improvement=0.005, retrain_time="23:55"):
    cfg = MagicMock()
    cfg.get_symbol_value.side_effect = lambda sym, key, default=None: {
        "min_auc_improvement": min_improvement, "retrain_time_utc": retrain_time}.get(key, default)
    return cfg


def _accept(old_long, old_short, new_long, new_short):
    longs, shorts, aucs = {"EURUSD": _ens(old_long)}, {"EURUSD": _ens(old_short)}, {}
    monitor = MagicMock()
    new = {"long": _ens(new_long), "short": _ens(new_short)}
    with patch.object(retraining, "_load_staged", side_effect=lambda cfg, sym, side, params: new[side]), \
            patch.object(retraining, "promote_staged_ensemble"), patch.object(retraining, "discard_staged_ensemble"):
        retraining._handle_model_acceptance("EURUSD", _cfg(), longs, shorts, aucs, monitor, None, {})
    return longs["EURUSD"], shorts["EURUSD"], aucs, monitor


def test_model_accepted_only_if_auc_improves_by_margin():
    long_, short_, aucs, monitor = _accept(0.60, 0.60, 0.62, 0.601)
    assert long_.ensemble_cv_auc_ == 0.62      # improved by 0.02 >= 0.005 -> accepted
    assert short_.ensemble_cv_auc_ == 0.60     # improved by 0.001 < 0.005 -> rejected
    assert aucs["EURUSD"] == 0.62
    monitor.update_ensemble_auc.assert_called_once_with(0.62)


def test_worse_model_is_rejected():
    long_, short_, aucs, monitor = _accept(0.60, 0.60, 0.55, 0.50)
    assert long_.ensemble_cv_auc_ == 0.60 and short_.ensemble_cv_auc_ == 0.60
    assert aucs == {} and not monitor.update_ensemble_auc.called


def _trigger(now, last_date=None, in_progress=False):
    mt5c = MagicMock()
    status, procs, last = {"EURUSD": in_progress}, {}, {"EURUSD": last_date}
    rc = MagicMock()
    with patch.object(retraining, "Process") as proc:
        retraining._check_and_trigger_retraining(
            _cfg(), "EURUSD", {"EURUSD": MagicMock()}, True, MagicMock(), {}, procs, status, last, rc, mt5c, now_fn=lambda: now)
    return proc, status, last, rc


UTC = datetime.timezone.utc


def test_retrain_triggers_once_after_scheduled_time():
    proc, status, last, rc = _trigger(datetime.datetime(2025, 1, 6, 23, 56, tzinfo=UTC))
    proc.return_value.start.assert_called_once()
    assert status["EURUSD"] is True and last["EURUSD"] == datetime.date(2025, 1, 6)
    rc.update_last_daily_retrain_date.assert_called_once()


def test_retrain_not_triggered_before_time_or_twice_per_day():
    before, *_ = _trigger(datetime.datetime(2025, 1, 6, 23, 0, tzinfo=UTC))
    assert not before.called
    again, *_ = _trigger(datetime.datetime(2025, 1, 6, 23, 59, tzinfo=UTC), last_date=datetime.date(2025, 1, 6))
    assert not again.called
    busy, *_ = _trigger(datetime.datetime(2025, 1, 6, 23, 59, tzinfo=UTC), in_progress=True)
    assert not busy.called
