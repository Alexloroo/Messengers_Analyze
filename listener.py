"""Telethon listener that saves messages from allowed informational chats to the DB."""

import asyncio
import logging
import sys
from datetime import datetime, timezone

from telethon import TelegramClient, events
from telethon.tl.types import Channel, Chat, User

from config import ALLOWED_INFO_CHATS, SESSION_NAME, TELETHON_API_HASH, TELETHON_API_ID
from database import AsyncSessionLocal, Chat, Message, init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _resolve_chat_identifier(entity) -> str:
    """Return a comparable string identifier for a chat entity."""
    if getattr(entity, "username", None):
        return entity.username.lower()
    if getattr(entity, "title", None):
        return entity.title.lower()
    return str(entity.id)


def _is_allowed_chat(entity) -> bool:
    """Check whether the chat is allowed for collection.

    If ALLOWED_INFO_CHATS is empty, all chats are considered allowed.
    """
    if not ALLOWED_INFO_CHATS:
        return True

    allowed_lower = [a.lower() for a in ALLOWED_INFO_CHATS]
    identifiers = [
        str(entity.id),
        f"-{entity.id}",
        f"-100{entity.id}",
    ]
    if getattr(entity, "username", None):
        identifiers.append(entity.username.lower())
    if getattr(entity, "title", None):
        identifiers.append(entity.title.lower())

    return any(identifier in allowed_lower for identifier in identifiers)


def _chat_type_name(entity) -> str:
    if isinstance(entity, Channel):
        return "channel" if entity.broadcast else "supergroup"
    if isinstance(entity, Chat):
        return "group"
    if isinstance(entity, User):
        return "user"
    return entity.__class__.__name__


async def _get_or_create_chat(session, entity) -> Chat:
    telegram_id = entity.id
    chat = await session.get(Chat, telegram_id)
    if chat is None:
        chat = Chat(
            telegram_id=telegram_id,
            title=getattr(entity, "title", None),
            username=getattr(entity, "username", None),
            type=_chat_type_name(entity),
            category="informational",
            is_allowed=True,
        )
        session.add(chat)
        await session.flush()
        logger.info("Registered new allowed chat: %s (id=%s)", chat.title or chat.username, telegram_id)
    return chat


async def _message_exists(session, chat_id: int, telegram_id: int) -> bool:
    from sqlalchemy import select
    result = await session.execute(
        select(Message).where(Message.chat_id == chat_id, Message.telegram_id == telegram_id)
    )
    return result.scalar_one_or_none() is not None


async def _store_message(session, event) -> None:
    entity = await event.get_chat()

    if not _is_allowed_chat(entity):
        return

    chat = await _get_or_create_chat(session, entity)

    msg = event.message
    if await _message_exists(session, chat.telegram_id, msg.id):
        logger.debug("Message %s from chat %s already stored", msg.id, chat.telegram_id)
        return

    sender = await event.get_sender()
    sender_id = getattr(sender, "id", None)
    sender_username = getattr(sender, "username", None)

    message = Message(
        telegram_id=msg.id,
        chat_id=chat.telegram_id,
        sender_id=sender_id,
        sender_username=sender_username,
        text=msg.message or "",
        date=msg.date.replace(tzinfo=timezone.utc) if msg.date and msg.date.tzinfo is None else msg.date,
        is_reply=bool(msg.is_reply),
        reply_to_msg_id=msg.reply_to_msg_id if msg.is_reply else None,
    )
    message.set_raw({
        "chat": {
            "id": chat.telegram_id,
            "title": chat.title,
            "username": chat.username,
            "type": chat.type,
        },
        "sender": {
            "id": sender_id,
            "username": sender_username,
            "name": " ".join(filter(None, [
                getattr(sender, "first_name", None),
                getattr(sender, "last_name", None),
            ])),
        },
        "message": {
            "id": msg.id,
            "text": msg.message or "",
            "date": msg.date.isoformat() if msg.date else None,
            "is_reply": msg.is_reply,
            "reply_to_msg_id": message.reply_to_msg_id,
        },
        "meta": {"stored_at": datetime.now(timezone.utc).isoformat()},
    })

    session.add(message)
    await session.commit()
    logger.info(
        "Stored message %s from chat '%s' (id=%s)",
        msg.id,
        chat.title or chat.username,
        chat.telegram_id,
    )


async def main() -> None:
    if not TELETHON_API_ID or not TELETHON_API_HASH:
        logger.error("TELETHON_API_ID and TELETHON_API_HASH must be set")
        sys.exit(1)

    await init_db()

    if not ALLOWED_INFO_CHATS:
        logger.warning(
            "ALLOWED_INFO_CHATS is empty. The listener will collect messages from ALL chats. "
            "Set the environment variable to restrict collection to specific chat ids/usernames/titles."
        )

    client = TelegramClient(SESSION_NAME, int(TELETHON_API_ID), TELETHON_API_HASH)

    @client.on(events.NewMessage)
    async def handler(event):
        async with AsyncSessionLocal() as session:
            try:
                await _store_message(session, event)
            except Exception:
                logger.exception("Failed to store message")
                await session.rollback()

    logger.info("Starting informational chat listener...")
    logger.info("Allowed chats: %s", ALLOWED_INFO_CHATS or "none")
    async with client:
        await client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(main())
