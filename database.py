"""SQLAlchemy async ORM models and session factory."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from config import DATABASE_URL


class Base(DeclarativeBase):
    pass


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Chat(Base):
    """A Telegram chat or channel tracked by the collector."""

    __tablename__ = "chats"

    telegram_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    username: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    category: Mapped[str] = mapped_column(
        String(64), default="informational", index=True
    )
    is_allowed: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, onupdate=_utc_now
    )

    messages: Mapped[list["Message"]] = relationship(
        back_populates="chat", cascade="all, delete-orphan", lazy="selectin"
    )


class Message(Base):
    """A single Telegram message stored for analysis."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    telegram_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    chat_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("chats.telegram_id", ondelete="CASCADE"), nullable=False, index=True
    )
    sender_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)
    sender_username: Mapped[str | None] = mapped_column(String(255), nullable=True)
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    is_reply: Mapped[bool] = mapped_column(Boolean, default=False)
    reply_to_msg_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    raw_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, index=True
    )

    chat: Mapped["Chat"] = relationship(back_populates="messages")
    tags: Mapped[list["Tag"]] = relationship(
        back_populates="message", cascade="all, delete-orphan", lazy="selectin"
    )

    def set_raw(self, data: dict[str, Any]) -> None:
        self.raw_json = json.dumps(data, ensure_ascii=False, default=str)


class Tag(Base):
    """Tag attached to a message (topic, urgency, etc.)."""

    __tablename__ = "tags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    message_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("messages.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    value: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)

    message: Mapped["Message"] = relationship(back_populates="tags")


engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    return AsyncSessionLocal()
