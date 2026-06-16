"""Ежедневный batch-анализ информационных сообщений.

Скрипт берёт необработанные сообщения из БД, отправляет их в Groq через
LangChain (структурированный JSON-вывод) и сохраняет результат в таблицу
``analysis_results``.

Пример запуска по cron (каждый день в 03:00):

    0 3 * * * cd /home/alexseyka/Messengers_Analyze && \
        /home/alexseyka/Messengers_Analyze/.venv/bin/python daily_analysis.py \
        >> /var/log/messengers_analysis.log 2>&1
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from sqlalchemy import select

from config import (
    ANALYSIS_BATCH_SIZE,
    ANALYSIS_MAX_WORKERS,
    GROQ_API_KEY,
    GROQ_MODEL,
)
from database import AnalysisResult, AsyncSessionLocal, Message, init_db

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


class MessageAnalysis(BaseModel):
    """Структурированный ответ LLM для одного сообщения."""

    usefulness_class: Literal["high", "medium", "low", "spam"] = Field(
        description="Класс полезности сообщения"
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="3-7 ключевых тем/слов сообщения",
    )
    summary: str = Field(
        description="2-3 предложения сути сообщения",
    )


SYSTEM_PROMPT = """Ты анализатор новостных сообщений из Telegram.

Для каждого сообщения верни строго JSON следующего вида:
{
  "usefulness_class": "high|medium|low|spam",
  "keywords": ["тема1", "тема2", "тема3"],
  "summary": "2-3 предложения суть сообщения"
}

Классы полезности:
- high: важная новость, требует внимания
- medium: полезный контекст
- low: малополезная информация
- spam: реклама, мусор, оффтопик

Отвечай только JSON, без markdown и пояснений."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _build_analyzer() -> ChatGroq:
    """Создать LangChain-чат-модель с structured output поверх Groq."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY не задан. Укажи ключ в .env или переменной окружения."
        )

    model = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0.1,
        max_tokens=500,
    )
    return model.with_structured_output(MessageAnalysis, method="json_mode")


async def fetch_unprocessed(session, limit: int = ANALYSIS_BATCH_SIZE) -> list[Message]:
    """Загрузить следующую партию непроанализированных сообщений."""
    result = await session.execute(
        select(Message)
        .where(Message.analyzed_at.is_(None))
        .order_by(Message.created_at)
        .limit(limit)
    )
    return list(result.scalars().all())


async def analyze_one(
    message: Message,
    analyzer,
    semaphore: asyncio.Semaphore,
) -> tuple[MessageAnalysis, int] | None:
    """Проанализировать одно сообщение через Groq."""
    async with semaphore:
        text = (message.text or "").strip()
        if not text:
            logger.debug("Сообщение id=%s пустое, пропускаем", message.id)
            return None

        try:
            analysis = await analyzer.ainvoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=text[:4000]),
                ]
            )
            return analysis, message.id
        except Exception:
            logger.exception("Ошибка анализа сообщения id=%s", message.id)
            return None


async def save_results(
    session,
    results: list[tuple[MessageAnalysis, int] | None],
) -> int:
    """Сохранить результаты анализа и отметить сообщения обработанными."""
    saved = 0
    model_used = GROQ_MODEL

    for item in results:
        if item is None:
            continue
        analysis, message_id = item
        data = analysis.model_dump()

        existing = await session.execute(
            select(AnalysisResult).where(AnalysisResult.message_id == message_id)
        )
        existing = existing.scalar_one_or_none()

        if existing:
            existing.usefulness_class = data["usefulness_class"]
            existing.keywords = data["keywords"]
            existing.summary = data["summary"]
            existing.analyzed_at = _utc_now()
            existing.model_used = model_used
        else:
            session.add(
                AnalysisResult(
                    message_id=message_id,
                    usefulness_class=data["usefulness_class"],
                    keywords=data["keywords"],
                    summary=data["summary"],
                    model_used=model_used,
                )
            )

        message = await session.get(Message, message_id)
        if message:
            message.analyzed_at = _utc_now()

        saved += 1

    await session.commit()
    return saved


async def main() -> None:
    await init_db()

    analyzer = _build_analyzer()
    semaphore = asyncio.Semaphore(ANALYSIS_MAX_WORKERS)

    while True:
        async with AsyncSessionLocal() as session:
            messages = await fetch_unprocessed(session)
            if not messages:
                logger.info("Нет необработанных сообщений")
                break

            logger.info("Обработка %d сообщений...", len(messages))
            tasks = [
                analyze_one(message, analyzer, semaphore) for message in messages
            ]
            results = await asyncio.gather(*tasks)
            saved = await save_results(session, results)
            logger.info("Сохранено %d результатов", saved)


if __name__ == "__main__":
    asyncio.run(main())
