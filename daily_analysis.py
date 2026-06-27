"""Ежедневный batch-анализ информационных сообщений через LangGraph.

Скрипт берёт необработанные сообщения из БД, пропускает их через
LangGraph-граф (filter → classify → summarize → critic) и сохраняет
результат в таблицу ``analysis_results``.

Пример запуска по cron (каждый день в 03:00):

    0 3 * * * cd /home/alexseyka/Messengers_Analyze && \
        /home/alexseyka/Messengers_Analyze/.venv/bin/python daily_analysis.py \
        >> /var/log/messengers_analysis.log 2>&1
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import datetime, timezone
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from config import (
    ANALYSIS_BATCH_SIZE,
    ANALYSIS_MAX_WORKERS,
    GROQ_MODEL,
)
from database import AnalysisResult, AsyncSessionLocal, Message, init_db
from llm_provider import get_chat_model

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State flowing through the analysis graph
# ---------------------------------------------------------------------------


class MessageAnalysisState(TypedDict):
    """Состояние, которое передаётся по узлам графа анализа."""

    # Input
    raw_text: str
    chat_title: str

    # Processing outputs
    category: str  # news | discussion | task | announcement | article | other | spam
    summary: str
    keywords: list[str]
    quality_score: float  # 0.0 — 1.0
    quality_feedback: str
    retry_count: int

    # Final
    is_useful: bool


# ---------------------------------------------------------------------------
# Graph constants
# ---------------------------------------------------------------------------

MAX_RETRIES = 2
ERROR_RETRIES = 3

# ---------------------------------------------------------------------------
# Error Validation
# ---------------------------------------------------------------------------
def is_error(exc: Exception) -> int | None:
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        return status
    text = str(exc).lower()
    if "429" in text:
        return 429
    return None









# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------






async def classify_node(state: MessageAnalysisState) -> dict:
    """Классифицировать сообщение по категории."""
    llm = get_chat_model(model=GROQ_MODEL, temperature=0.0)

    response = await llm.ainvoke([
        SystemMessage(content=(
            "Classify the following Telegram message into EXACTLY one category. "
            "Respond with ONLY the category name, nothing else.\n"
            "Categories: news, discussion, LLMs,jobs, meetup, other"
        )),
        HumanMessage(content=f"Chat: {state['chat_title']}\nMessage: {state['raw_text']}"),
    ])

    category = response.content.strip().lower()
    valid_categories = {"news", "discussion", "LLMs", "jobs", "meetup", "other"}
    if category not in valid_categories:
        category = "other"

    logger.info("Classified message as: %s", category)
    return {"category": category}


def _extract_summary_and_keywords(text: str) -> tuple[str, list[str]]:
    """Разобрать ответ LLM на summary и keywords."""
    text = text.strip()
    summary = text
    keywords: list[str] = []

    parts = re.split(r"(?i)\n\s*KEYWORDS:\s*", text, maxsplit=1)
    if len(parts) == 2:
        summary_part, keywords_part = parts
        keywords = [
            keyword.strip()
            for keyword in re.split(r"[,;]", keywords_part)
            if keyword.strip()
        ]
    else:
        summary_part = text

    summary = re.sub(r"(?i)^\s*SUMMARY:\s*", "", summary_part).strip()
    if not summary:
        summary = text

    return summary, keywords[:10]


async def summarize_node(state: MessageAnalysisState) -> dict:
    """Сгенерировать краткое резюме и ключевые слова сообщения."""
    llm = get_chat_model(model=GROQ_MODEL, temperature=0.3)

    response = await llm.ainvoke([
        SystemMessage(content=(
            "Summarize the following Telegram message in 1-2 sentences in Russian. "
            "Focus on the key information: who, what, when, action items. "
            "Be concise and factual.\n\n"
            "Also extract 3-7 key topics/keywords (in Russian or English).\n\n"
            "Respond EXACTLY in this format:\n"
            "SUMMARY: <1-2 sentences>\n"
            "KEYWORDS: <keyword1>, <keyword2>, ..."
        )),
        HumanMessage(content=(
            f"Chat: {state['chat_title']}\n"
            f"Category: {state['category']}\n"
            f"Message:\n{state['raw_text']}"
        )),
    ])

    summary, keywords = _extract_summary_and_keywords(response.content)
    logger.info("Generated summary: %s", summary[:80])
    logger.info("Extracted keywords: %s", keywords)
    return {
        "summary": summary,
        "keywords": keywords,
        "retry_count": state.get("retry_count", 0),
    }


async def critic_node(state: MessageAnalysisState) -> dict:
    """Оценить качество сгенерированного резюме.

    Проверяет:
      - Точность (нет галлюцинаций)
      - Полнота (ключевые факты сохранены)
      - Краткость
    """
    llm = get_chat_model(model=GROQ_MODEL, temperature=0.0)

    response = await llm.ainvoke([
        SystemMessage(content=(
            "You are a quality checker for message summaries. "
            "Compare the original message with the summary and rate the quality.\n\n"
            "Respond in EXACTLY this format (2 lines):\n"
            "SCORE: <float 0.0 to 1.0>\n"
            "FEEDBACK: <brief explanation>\n\n"
            "Score >= 0.7 means acceptable quality.\n"
            "Score < 0.7 means the summary needs improvement."
        )),
        HumanMessage(content=(
            f"Original message:\n{state['raw_text']}\n\n"
            f"Summary:\n{state['summary']}"
        )),
    ])

    text = response.content.strip()
    score = 0.8  # default
    feedback = "OK"

    for line in text.split("\n"):
        line = line.strip()
        if line.upper().startswith("SCORE:"):
            try:
                score = float(line.split(":", 1)[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.upper().startswith("FEEDBACK:"):
            feedback = line.split(":", 1)[1].strip()

    logger.info("Critic score: %.2f — %s", score, feedback)
    return {
        "quality_score": score,
        "quality_feedback": feedback,
        "is_useful": True,
    }


async def mark_spam_node(state: MessageAnalysisState) -> dict:
    """Пометить сообщение как спам и завершить обработку."""
    return {
        "is_useful": False,
        "summary": "",
        "category": "spam",
        "keywords": [],
    }


async def bump_retry_node(state: MessageAnalysisState) -> dict:
    """Увеличить счётчик попыток перед повторным саммари."""
    return {"retry_count": state.get("retry_count", 0) + 1}


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------





def after_critic(state: MessageAnalysisState) -> str:
    """Направить после критика: принять если качество хорошее или попытки исчерпаны."""
    score = state.get("quality_score", 1.0)
    retries = state.get("retry_count", 0)

    if score >= 0.7 or retries >= MAX_RETRIES:
        return "accept"
    return "retry_summarize"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_analysis_graph():
    """Построить и скомпилировать граф анализа сообщений.

    Flow:
        START → filter → [spam? → mark_spam → END]
                         [useful? → classify → summarize → critic]
                                                          ↑         |
                                                          └─ retry ─┘
                                                                    → END
    """
    graph = StateGraph(MessageAnalysisState)

    graph.add_node("classify", classify_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("critic", critic_node)
    graph.add_node("mark_spam", mark_spam_node)
    graph.add_node("bump_retry", bump_retry_node)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "summarize")
    graph.add_edge("summarize", "critic")

    graph.add_conditional_edges(
        "critic",
        after_critic,
        {
            "accept": END,
            "retry_summarize": "bump_retry",
        },
    )

    graph.add_edge("bump_retry", "summarize")

    return graph.compile()


async def analyze_message(
    raw_text: str,
    chat_title: str = "Unknown Chat",
) -> MessageAnalysisState:
    """Пропустить одно сообщение через полный аналитический граф.

    Args:
        raw_text: Текст сообщения для анализа.
        chat_title: Название Telegram-чата, из которого пришло сообщение.

    Returns:
        Финальное состояние графа с category, summary, keywords,
        quality_score и т.д.
    """
    graph = build_analysis_graph()
    initial_state: MessageAnalysisState = {
        "raw_text": raw_text,
        "chat_title": chat_title,
        "category": "",
        "summary": "",
        "keywords": [],
        "quality_score": 0.0,
        "quality_feedback": "",
        "retry_count": 0,
        "is_useful": False,
    }

    result = await graph.ainvoke(initial_state)
    return result


# ---------------------------------------------------------------------------
# Database & batch pipeline
# ---------------------------------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _map_usefulness(state: MessageAnalysisState) -> str:
    """Сопоставить выход графа с классом полезности из daily_analysis."""

    category = state.get("category", "other")
    if category in ("news", "announcement", "task"):
        return "high"
    if category == "discussion":
        return "medium"
    return "low"


async def fetch_unprocessed(session, limit: int = ANALYSIS_BATCH_SIZE) -> list[Message]:
    """Загрузить следующую партию непроанализированных сообщений."""
    result = await session.execute(
        select(Message)
        .options(selectinload(Message.chat))
        .where(Message.analyzed_at.is_(None))
        .order_by(Message.created_at)
        .limit(limit)
    )
    return list(result.scalars().all())


async def analyze_one(
    message: Message,
    graph,
    semaphore: asyncio.Semaphore,
) -> tuple[MessageAnalysisState, int] | None:
    """Проанализировать одно сообщение через LangGraph."""
    async with semaphore:
        text = (message.text or "").strip()
        if not text:
            logger.debug("Сообщение id=%s пустое, пропускаем", message.id)
            return None

        chat_title = message.chat.title if message.chat else "Unknown Chat"

        try:
            initial_state: MessageAnalysisState = {
                "raw_text": text[:4000],
                "chat_title": chat_title,
                "category": "",
                "summary": "",
                "keywords": [],
                "quality_score": 0.0,
                "quality_feedback": "",
                "retry_count": 0,
                "is_useful": False,
            }
            for attempt in range(ERROR_RETRIES):
                try:
                    result = await graph.ainvoke(initial_state)
                    return result, message.id
                except Exception as exc:
                    if is_error(exc) == 429 and attempt < ERROR_RETRIES - 1:
                        await asyncio.sleep(4 **(1 + attempt))
                        continue
                    logger.exception("Ошибка анализа сообщения id=%s: %s", message.id, exc)
                    return None  
        except Exception:
            logger.exception("Ошибка анализа сообщения id=%s", message.id)
            return None


async def save_results(
    session,
    results: list[tuple[MessageAnalysisState, int] | None],
) -> int:
    """Сохранить результаты анализа и отметить сообщения обработанными."""
    saved = 0
    model_used = GROQ_MODEL

    for item in results:
        if item is None:
            continue
        state, message_id = item

        usefulness_class = _map_usefulness(state)
        keywords = state.get("keywords", []) or []
        summary = state.get("summary", "") or ""

        existing = await session.execute(
            select(AnalysisResult).where(AnalysisResult.message_id == message_id)
        )
        existing = existing.scalar_one_or_none()

        if existing:
            existing.usefulness_class = usefulness_class
            existing.keywords = keywords
            existing.summary = summary
            existing.analyzed_at = _utc_now()
            existing.model_used = model_used
        else:
            session.add(
                AnalysisResult(
                    message_id=message_id,
                    usefulness_class=usefulness_class,
                    keywords=keywords,
                    summary=summary,
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

    graph = build_analysis_graph()
    semaphore = asyncio.Semaphore(ANALYSIS_MAX_WORKERS)

    while True:
        async with AsyncSessionLocal() as session:
            messages = await fetch_unprocessed(session)
            if not messages:
                logger.info("Нет необработанных сообщений")
                break

            logger.info("Обработка %d сообщений...", len(messages))
            tasks = [
                analyze_one(message, graph, semaphore) for message in messages
            ]
            results = await asyncio.gather(*tasks)
            saved = await save_results(session, results)
            logger.info("Сохранено %d результатов", saved)


if __name__ == "__main__":
    asyncio.run(main())
