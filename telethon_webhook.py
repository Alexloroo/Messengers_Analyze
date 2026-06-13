import asyncio
import json
import os
import sys
from datetime import datetime, timezone

import httpx
from telethon import TelegramClient, events


API_ID = os.getenv("TELETHON_API_ID")
API_HASH = os.getenv("TELETHON_API_HASH")
SESSION_NAME = os.getenv("TELETHON_SESSION", "telethon")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

# Optional filters (comma-separated)
INCLUDE_CHATS = [c.strip() for c in os.getenv("INCLUDE_CHATS", "").split(",") if c.strip()]
EXCLUDE_CHATS = [c.strip() for c in os.getenv("EXCLUDE_CHATS", "").split(",") if c.strip()]

# Optional: restrict to these sender usernames/ids (comma-separated)
INCLUDE_SENDERS = [s.strip() for s in os.getenv("INCLUDE_SENDERS", "").split(",") if s.strip()]

TIMEOUT = float(os.getenv("WEBHOOK_TIMEOUT", "10"))
RETRY_COUNT = int(os.getenv("WEBHOOK_RETRY", "3"))
RETRY_DELAY = float(os.getenv("WEBHOOK_RETRY_DELAY", "1"))


def _require_env(name: str, value: str) -> None:
    if not value:
        print(f"Missing required env var: {name}", file=sys.stderr)
        sys.exit(1)


def _match_list(value: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    value_lower = value.lower()
    return any(p.lower() == value_lower for p in patterns)


async def _post_with_retry(payload: dict) -> None:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        last_error = None
        for attempt in range(RETRY_COUNT):
            try:
                resp = await client.post(WEBHOOK_URL, json=payload)
                if resp.status_code < 300:
                    return
                last_error = f"HTTP {resp.status_code}: {resp.text}"
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
            await asyncio.sleep(RETRY_DELAY)
        print(f"Webhook delivery failed: {last_error}", file=sys.stderr)


async def main() -> None:
    _require_env("TELETHON_API_ID", API_ID)
    _require_env("TELETHON_API_HASH", API_HASH)
    _require_env("WEBHOOK_URL", WEBHOOK_URL)

    client = TelegramClient(SESSION_NAME, int(API_ID), API_HASH)

    @client.on(events.NewMessage)
    async def handler(event: events.NewMessage.Event) -> None:
        chat = await event.get_chat()
        sender = await event.get_sender()

        chat_title = getattr(chat, "title", None) or getattr(chat, "username", None) or str(getattr(chat, "id", ""))
        sender_id = str(getattr(sender, "id", ""))
        sender_username = getattr(sender, "username", None) or ""

        if INCLUDE_CHATS and not _match_list(chat_title, INCLUDE_CHATS):
            return
        if EXCLUDE_CHATS and _match_list(chat_title, EXCLUDE_CHATS):
            return
        if INCLUDE_SENDERS:
            if sender_username and not _match_list(sender_username, INCLUDE_SENDERS):
                if sender_id not in INCLUDE_SENDERS:
                    return
            elif sender_id not in INCLUDE_SENDERS:
                return

        msg = event.message
        payload = {
            "chat": {
                "id": getattr(chat, "id", None),
                "title": chat_title,
                "type": chat.__class__.__name__,
            },
            "sender": {
                "id": getattr(sender, "id", None),
                "username": sender_username,
                "name": " ".join(filter(None, [getattr(sender, "first_name", None), getattr(sender, "last_name", None)])),
            },
            "message": {
                "id": msg.id,
                "text": msg.message or "",
                "date": msg.date.replace(tzinfo=timezone.utc).isoformat() if msg.date else None,
                "is_reply": msg.is_reply,
            },
            "meta": {
                "ts": datetime.now(timezone.utc).isoformat(),
            },
        }

        await _post_with_retry(payload)

    print("Starting Telethon client...")
    async with client:
        await client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(main())
