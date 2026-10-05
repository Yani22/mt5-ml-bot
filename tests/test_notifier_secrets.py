"""B17: the Telegram token never appears in the log, and token and chat id come from the environment (.env) first,
with config.yaml as the fallback."""
from types import SimpleNamespace as NS

import pytest
import requests
from loguru import logger

from src.config import Cfg
from src.notifier import TelegramNotifier

TOKEN = "123456:SECRET-token_value"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)


@pytest.fixture
def log_lines():
    lines = []
    sink = logger.add(lambda m: lines.append(str(m)), format="{level} {message}", level="DEBUG")
    yield lines
    logger.remove(sink)


def make(token="", chat="", env_token=None, env_chat=None, monkeypatch=None):
    cfg = Cfg()
    cfg.monitoring.telegram_bot_token, cfg.monitoring.telegram_chat_id = token, chat
    if env_token is not None:
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", env_token)
    if env_chat is not None:
        monkeypatch.setenv("TELEGRAM_CHAT_ID", env_chat)
    return TelegramNotifier(cfg), cfg


# ---- the token stays out of the log ------------------------------------------------------------------------------

def test_a_connection_error_does_not_log_the_token(monkeypatch, log_lines):
    def fake_post(url, **kw):
        raise requests.exceptions.ConnectionError(f"Max retries exceeded with url: {url}")

    monkeypatch.setattr("src.notifier.requests.post", fake_post)
    make(TOKEN, "42")[0].send_message("hello")
    assert any("Failed to send Telegram message" in line for line in log_lines)
    assert not any(TOKEN in line for line in log_lines)


def test_an_http_error_does_not_log_the_token(monkeypatch, log_lines):
    def fake_post(url, **kw):
        def boom():
            raise requests.exceptions.HTTPError(f"400 Client Error: Bad Request for url: {url}")
        return NS(raise_for_status=boom)

    monkeypatch.setattr("src.notifier.requests.post", fake_post)
    make(TOKEN, "42")[0].send_message("hello")
    assert any("Failed to send Telegram message" in line for line in log_lines)
    assert not any(TOKEN in line for line in log_lines)


def test_an_unexpected_error_does_not_log_the_token(monkeypatch, log_lines):
    def fake_post(url, **kw):
        raise ValueError(f"bad url {url}")

    monkeypatch.setattr("src.notifier.requests.post", fake_post)
    make(TOKEN, "42")[0].send_message("hello")
    assert any("unexpected error" in line for line in log_lines)
    assert not any(TOKEN in line for line in log_lines)


# ---- environment first, config as fallback -----------------------------------------------------------------------

def test_the_environment_wins_over_the_config(monkeypatch):
    n, _ = make("cfg-token", "1", env_token=TOKEN, env_chat="42", monkeypatch=monkeypatch)
    assert (n.bot_token, n.chat_id, n.enabled) == (TOKEN, "42", True)


def test_a_blank_environment_value_falls_back_to_the_config(monkeypatch):
    n, _ = make("cfg-token", "7", env_token="  ", env_chat="", monkeypatch=monkeypatch)
    assert (n.bot_token, n.chat_id, n.enabled) == ("cfg-token", "7", True)


def test_nothing_set_disables_the_notifier():
    assert make()[0].enabled is False


def test_the_environment_token_is_not_written_back_to_the_config(monkeypatch):
    _, cfg = make("", "", env_token=TOKEN, env_chat="42", monkeypatch=monkeypatch)
    assert cfg.monitoring.telegram_bot_token == "" and cfg.monitoring.telegram_chat_id == ""
