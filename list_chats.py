"""List available Telegram dialogs to configure ALLOWED_INFO_CHATS."""

import asyncio
import logging
import sys

from telethon import TelegramClient

from config import SESSION_NAME, TELETHON_API_HASH, TELETHON_API_ID

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _require_env(name: str, value: str | None) -> str:
    if not value:
        logger.error("Missing required environment variable: %s", name)
        sys.exit(1)
    return value


async def main() -> None:
    api_id = int(_require_env("TELETHON_API_ID", TELETHON_API_ID))
    api_hash = _require_env("TELETHON_API_HASH", TELETHON_API_HASH)

    client = TelegramClient(SESSION_NAME, api_id, api_hash)
    async with client:
        dialogs = await client.get_dialogs()
        print(f"\n{'ID':<18} {'Type':<12} {'Username':<20} {'Title'}")
        print("-" * 80)
        for dialog in dialogs:
            entity = dialog.entity
            username = getattr(entity, "username", None) or ""
            title = getattr(entity, "title", None) or " ".join(
                filter(None, [getattr(entity, "first_name", None), getattr(entity, "last_name", None)])
            )
            print(f"{entity.id:<18} {entity.__class__.__name__:<12} {username:<20} {title}")


if __name__ == "__main__":
    asyncio.run(main())
