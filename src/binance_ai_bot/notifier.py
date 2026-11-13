"""Telegram bildirimleri (isteğe bağlı).

`TELEGRAM_BOT_TOKEN` ve `TELEGRAM_CHAT_ID` ortam değişkenleri tanımlı değilse hiçbir şey yapmaz.
Bildirim hataları ASLA alım-satım akışını bozmaz; kısa zaman aşımı ile sessizce yutulur.
"""

from __future__ import annotations

import logging
import os

import requests

from .security import redact_secrets


class TelegramNotifier:
    def __init__(self, token: str | None = None, chat_id: str | None = None) -> None:
        self.token = (token if token is not None else os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
        self.chat_id = (chat_id if chat_id is not None else os.getenv("TELEGRAM_CHAT_ID", "")).strip()

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, text: str) -> bool:
        if not self.enabled:
            return False
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": redact_secrets(text)[:900]},
                timeout=3.0,
            )
            return resp.status_code == 200
        except Exception as exc:  # noqa: BLE001 - bildirim akışı asla işlem akışını durdurmaz
            logging.warning("Telegram bildirimi gönderilemedi: %s", type(exc).__name__)
            return False
