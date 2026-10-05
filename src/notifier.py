# src/notifier.py
import os

import requests
from loguru import logger  # type: ignore
from src.config import Cfg

# (connect, read) seconds. Callers wait for the send, so an outage must not block them for longer than this.
TELEGRAM_TIMEOUT = (3.05, 10)


class TelegramNotifier:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        # The environment (.env, git-ignored) wins; config.yaml is only a fallback. The token stays on this object and is
        # not written back to the config, so a config dump cannot print it.
        self.bot_token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip() or cfg.monitoring.telegram_bot_token
        self.chat_id = (os.getenv("TELEGRAM_CHAT_ID") or "").strip() or cfg.monitoring.telegram_chat_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram Notifier not fully configured (missing bot_token or chat_id). Notifications will be disabled.")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("Telegram Notifier initialized.")

    def _redact(self, text) -> str:
        """requests puts the full URL, which contains the bot token, into its exception text."""
        text = str(text)
        return text.replace(self.bot_token, "***") if self.bot_token else text

    def send_message(self, message: str, level: str = "INFO"):
        if not self.enabled:
            return

        full_message = f"[{level}] {message}"
        payload = {
            "chat_id": self.chat_id,
            "text": full_message,
            "parse_mode": "HTML"  # Allows basic formatting like bold, italics
        }

        try:
            response = requests.post(self.base_url, data=payload, timeout=TELEGRAM_TIMEOUT)
            response.raise_for_status()  # Raise an exception for HTTP errors
            logger.debug(f"Telegram message sent: {message}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Telegram message: {self._redact(e)}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while sending Telegram message: {self._redact(e)}")
