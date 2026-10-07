"""Without `fetch.retrain_time_utc` the live bot never retrains (`_check_and_trigger_retraining` returns). The warning used to say it
falls back to `retrain_every_bars`, which only sets the backtester's block length."""
import logging

from src.config import Cfg


def test_the_warning_says_what_really_happens(tmp_path, caplog):
    path = tmp_path / "c.yaml"
    path.write_text("symbols: ['EURUSD#']\n")
    with caplog.at_level(logging.WARNING):
        Cfg.from_yaml(str(path))
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "retrain_time_utc" in text and "never retrain" in text
    assert "falling back" not in text and "retrain_every_bars" in text and "backtester" in text


def test_no_warning_when_a_time_is_set(tmp_path, caplog):
    path = tmp_path / "c.yaml"
    path.write_text("symbols: ['EURUSD#']\nfetch:\n  retrain_time_utc: '23:55'\n")
    with caplog.at_level(logging.WARNING):
        Cfg.from_yaml(str(path))
    assert "retrain_time_utc" not in " ".join(r.getMessage() for r in caplog.records)
