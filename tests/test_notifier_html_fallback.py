"""Telegram rejects an HTML message that holds a stray `<` or `&` with HTTP 400. The alert must still arrive, as plain text, and
only a 400 earns that second try (not a rate limit or a server error)."""
from types import SimpleNamespace as NS

import requests

import src.notifier as notifier_module
from src.config import Cfg
from src.notifier import TelegramNotifier


def make(monkeypatch, statuses):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    sent = []

    def post(url, data=None, timeout=None):
        sent.append(dict(data))
        status = statuses[min(len(sent) - 1, len(statuses) - 1)]

        def raise_for_status():
            if status >= 400:
                raise requests.exceptions.HTTPError(f"{status} for url: {url}", response=NS(status_code=status))
        return NS(status_code=status, raise_for_status=raise_for_status)

    monkeypatch.setattr(notifier_module.requests, "post", post)
    return TelegramNotifier(Cfg()), sent


def test_a_400_is_retried_once_as_plain_text_without_the_tags(monkeypatch):
    n, sent = make(monkeypatch, [400, 200])
    n.send_message("<b>CRITICAL:</b> Order failed: <MT5 result at 0x1> & more", level="CRITICAL")
    assert len(sent) == 2
    assert sent[0]["parse_mode"] == "HTML"
    assert "parse_mode" not in sent[1]
    assert sent[1]["text"] == "[CRITICAL] CRITICAL: Order failed: <MT5 result at 0x1> & more"


def test_a_message_that_goes_through_is_sent_once(monkeypatch):
    n, sent = make(monkeypatch, [200])
    n.send_message("<b>OK</b>")
    assert len(sent) == 1


def test_a_rate_limit_or_a_server_error_is_not_retried(monkeypatch):
    for status in (429, 500):
        n, sent = make(monkeypatch, [status])
        n.send_message("<b>x</b>")
        assert len(sent) == 1


def test_a_failed_retry_is_logged_not_raised(monkeypatch):
    n, sent = make(monkeypatch, [400, 400])
    n.send_message("<b>x</b>")
    assert len(sent) == 2
