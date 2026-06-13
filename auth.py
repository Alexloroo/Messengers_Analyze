"""Authorize Telethon client and persist the .session file."""

import asyncio
import logging
import sys

from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError

from config import TELETHON_API_HASH, TELETHON_API_ID, SESSION_NAME

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

    logger.info("Starting authorization, session file: %s.session", SESSION_NAME)
    await client.connect()

    if await client.is_user_authorized():
        logger.info("Already authorized.")
    else:
        phone = input("Enter your phone number (with country code): ").strip()
        await client.send_code_request(phone)
        code = input("Enter the code you received: ").strip()
        try:
            await client.sign_in(phone, code)
        except SessionPasswordNeededError:
            password = input("Two-factor authentication enabled. Enter your password: ").strip()
            await client.sign_in(password=password)
        logger.info("Authorization successful.")

    me = await client.get_me()
    logger.info("Logged in as: %s (id=%s)", me.username or me.first_name, me.id)
    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
