"""Tests for the LangGraph message analysis pipeline.

These tests run actual LLM calls through the graph pipeline and validate:
  1. Spam messages are correctly filtered
  2. Useful messages get properly classified
  3. Summaries are generated with acceptable quality
  4. The full pipeline produces expected output structure

All test runs are automatically traced in LangSmith for monitoring.

Run with: pytest tests/test_graph_pipeline.py -v
"""

import pytest

from daily_analysis import analyze_message, build_analysis_graph


# ---------------------------------------------------------------------------
# Test: Graph compiles correctly (no LLM needed)
# ---------------------------------------------------------------------------

def test_graph_compiles():
    """Verify that the analysis graph compiles without errors."""
    graph = build_analysis_graph()
    assert graph is not None
    print("✅ Graph compiled successfully.")


# ---------------------------------------------------------------------------
# Test: Spam detection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.langsmith
@pytest.mark.llm
async def test_spam_is_filtered(langsmith_configured, groq_configured):
    """Verify that spam messages are filtered out by the pipeline."""
    result = await analyze_message(
        raw_text="🎰 ЗАРАБОТАЙ 100000$ ЗА ДЕНЬ!!! Жми сюда >>> bit.ly/scam",
        chat_title="Random Chat",
    )

    assert result["is_spam"] is True
    assert result["is_useful"] is False
    assert result["category"] == "spam"
    print(f"✅ Spam correctly filtered: is_spam={result['is_spam']}")


# ---------------------------------------------------------------------------
# Test: News classification + summary
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.langsmith
@pytest.mark.llm
async def test_news_message_analysis(langsmith_configured, groq_configured):
    """Verify that a news message is properly classified and summarized."""
    result = await analyze_message(
        raw_text=(
            "🔥 Вышла новая версия Python 3.13! Основные изменения: "
            "улучшенный GIL, новый JIT-компилятор, обновлённый модуль typing. "
            "Подробности на python.org"
        ),
        chat_title="Python News RU",
    )

    assert result["is_spam"] is False
    assert result["is_useful"] is True
    assert result["category"] in ("news", "announcement", "article")
    assert len(result["summary"]) > 10, "Summary should be non-trivial"
    assert result["quality_score"] > 0.0
    print(f"✅ News analysis: category={result['category']}, score={result['quality_score']:.2f}")
    print(f"   Summary: {result['summary']}")


# ---------------------------------------------------------------------------
# Test: Task message with deadline extraction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.langsmith
@pytest.mark.llm
async def test_task_message_analysis(langsmith_configured, groq_configured):
    """Verify that a task message is classified and key info is summarized."""
    result = await analyze_message(
        raw_text=(
            "Ребят, нужно до пятницы подготовить презентацию по проекту. "
            "@alexey, ты берёшь слайды по архитектуре, @maria — по тестированию. "
            "Дедлайн — 20 июня, 18:00."
        ),
        chat_title="Рабочий чат — Проект Альфа",
    )

    assert result["is_spam"] is False
    assert result["is_useful"] is True
    assert result["category"] in ("task", "discussion", "announcement")
    assert len(result["summary"]) > 10
    print(f"✅ Task analysis: category={result['category']}, score={result['quality_score']:.2f}")
    print(f"   Summary: {result['summary']}")


# ---------------------------------------------------------------------------
# Test: Output structure validation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.langsmith
@pytest.mark.llm
async def test_output_structure(langsmith_configured, groq_configured):
    """Verify that the pipeline output contains all expected fields."""
    result = await analyze_message(
        raw_text="Обсуждение новой архитектуры микросервисов в среду в 15:00.",
        chat_title="Backend Team",
    )

    required_fields = [
        "raw_text", "chat_title", "is_spam", "category",
        "summary", "keywords", "quality_score", "quality_feedback",
        "retry_count", "is_useful",
    ]
    for field in required_fields:
        assert field in result, f"Missing field: {field}"

    assert isinstance(result["is_spam"], bool)
    assert isinstance(result["is_useful"], bool)
    assert isinstance(result["quality_score"], float)
    assert isinstance(result["retry_count"], int)
    assert isinstance(result["keywords"], list)
    print(f"✅ All {len(required_fields)} required fields present in output.")


# ---------------------------------------------------------------------------
# Test: Discussion classification
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.langsmith
@pytest.mark.llm
async def test_discussion_message(langsmith_configured, groq_configured):
    """Verify that a discussion message is correctly analyzed."""
    result = await analyze_message(
        raw_text=(
            "Кто-нибудь пробовал новый FastAPI 0.115? Говорят, там наконец "
            "нормально сделали dependency injection. Стоит ли переходить с Flask?"
        ),
        chat_title="Backend Developers RU",
    )

    assert result["is_spam"] is False
    assert result["is_useful"] is True
    assert result["category"] in ("discussion", "news", "other")
    assert len(result["summary"]) > 10
    print(f"✅ Discussion analysis: category={result['category']}")
