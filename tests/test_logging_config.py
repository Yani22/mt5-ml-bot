"""`logging.to_file`, `rotate` and `retention` in config.yaml were ignored by main.py (it only passed `level`), and importing main
opened logs/bot.log before the config was read."""
import os
import re

from src import utils

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_the_config_block_reaches_setup_logging(monkeypatch):
    seen = {}
    monkeypatch.setattr(utils, "setup_logging", lambda **k: seen.update(k))
    utils.setup_logging_from_config({"level": "DEBUG", "to_file": False, "rotate": "1 MB", "retention": "2 days"})
    assert seen == {"level": "DEBUG", "to_file": False, "rotate": "1 MB", "retention": "2 days"}


def test_missing_keys_fall_back_to_the_defaults(monkeypatch):
    seen = {}
    monkeypatch.setattr(utils, "setup_logging", lambda **k: seen.update(k))
    utils.setup_logging_from_config({})
    assert seen == {"level": "INFO", "to_file": True, "rotate": "10 MB", "retention": "7 days"}


def test_to_file_false_opens_no_log_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    utils.setup_logging_from_config({"level": "INFO", "to_file": False})
    from loguru import logger
    logger.info("x")
    assert not (tmp_path / "logs").exists()


def test_main_reads_the_block_and_does_not_open_a_file_at_import():
    source = open(os.path.join(ROOT, "main.py")).read()
    assert "setup_logging_from_config(cfg.logging)" in source
    bare = [line for line in source.splitlines() if re.match(r"\s*setup_logging\(\s*\)", line)]
    assert not bare, "a bare setup_logging() (file logging on) is still called"
