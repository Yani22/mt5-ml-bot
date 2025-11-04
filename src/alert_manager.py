# src/alert_manager.py
import datetime
import logging
from typing import Optional, Dict

from src.config import Cfg
from src.notifier import Notifier

logger = logging.getLogger(__name__)

class AlertManager:
    def __init__(self, cfg: Cfg, notifier: Optional[Notifier] = None):
        self.cfg = cfg
        self.notifier = notifier
        self.last_alert_time: Dict[str, datetime.datetime] = {}
        
        # Map string levels to logging integers
        self.level_map = {
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        self.min_notification_level_int = self.level_map.get(
            self.cfg.alerts.min_notification_level.upper(), logging.WARNING
        )

    def send_alert(self, message: str, level: str = "INFO", category: str = "GENERAL"):
        if not self.cfg.alerts.enabled:
            return

        current_time = datetime.datetime.utcnow()
        level_int = self.level_map.get(level.upper(), logging.INFO)

        # Log the message regardless of notification settings
        if level_int >= logging.CRITICAL:
            logger.critical(f"ALERT [{category}] {message}")
        elif level_int >= logging.ERROR:
            logger.error(f"ALERT [{category}] {message}")
        elif level_int >= logging.WARNING:
            logger.warning(f"ALERT [{category}] {message}")
        else:
            logger.info(f"ALERT [{category}] {message}")

        # Check if notification should be sent
        if self.notifier and level_int >= self.min_notification_level_int:
            # Apply throttling
            if category in self.last_alert_time:
                time_since_last_alert = (current_time - self.last_alert_time[category]).total_seconds() / 60
                if time_since_last_alert < self.cfg.alerts.alert_throttle_minutes:
                    logger.debug(f"Alert for category '{category}' throttled. Last sent {time_since_last_alert:.1f} minutes ago.")
                    return

            # Send notification
            try:
                self.notifier.send_message(f"<b>{level.upper()} ALERT:</b> [{category}] {message}", level=level)
                self.last_alert_time[category] = current_time
            except Exception as e:
                logger.error(f"Failed to send notification for alert [{category}] {message}: {e}")
