"""Pytest configuration and shared fixtures for LangSmith integration tests."""

import os
import sys
from pathlib import Path

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env before any test imports
from config import _load_env_file  # noqa: E402
_load_env_file(PROJECT_ROOT / ".env")


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "langsmith: marks tests that sync results to LangSmith (requires API key)",
    )
    config.addinivalue_line(
        "markers",
        "llm: marks tests that call a real LLM (requires GROQ_API_KEY)",
    )


@pytest.fixture(scope="session")
def langsmith_configured():
    """Check that LangSmith is properly configured."""
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        pytest.skip("LANGSMITH_API_KEY not set — skipping LangSmith tests")
    return True


@pytest.fixture(scope="session")
def groq_configured():
    """Check that Groq is properly configured."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set — skipping LLM tests")
    return True


# Sample messages for tests
SAMPLE_MESSAGES = {
    "news_ru": {
        "text": (
            "🔥 Вышла новая версия Python 3.13! Основные изменения: "
            "улучшенный GIL, новый JIT-компилятор, обновлённый модуль typing. "
            "Подробности на python.org"
        ),
        "chat_title": "Python News RU",
        "expected_category": "news",
        "expected_useful": True,
    },
    "spam": {
        "text": "🎰 ЗАРАБОТАЙ 100000$ ЗА ДЕНЬ!!! Жми сюда >>> bit.ly/scam",
        "chat_title": "Random Chat",
        "expected_category": "spam",
        "expected_useful": False,
    },
    "task_ru": {
        "text": (
            "Ребят, нужно до пятницы подготовить презентацию по проекту. "
            "@alexey, ты берёшь слайды по архитектуре, @maria — по тестированию. "
            "Дедлайн — 20 июня, 18:00."
        ),
        "chat_title": "Рабочий чат — Проект Альфа",
        "expected_category": "task",
        "expected_useful": True,
    },
    "discussion_ru": {
        "text": (
            "Кто-нибудь пробовал новый FastAPI 0.115? Говорят, там наконец "
            "нормально сделали dependency injection. Стоит ли переходить с Flask?"
        ),
        "chat_title": "Backend Developers RU",
        "expected_category": "discussion",
        "expected_useful": True,
    },
    "announcement_ru": {
        "text": (
            "📢 Внимание! Завтра с 10:00 до 12:00 будет проводиться плановое "
            "обслуживание серверов. Все сервисы будут недоступны. "
            "Просьба сохранить свою работу заранее."
        ),
        "chat_title": "DevOps Alerts",
        "expected_category": "announcement",
        "expected_useful": True,
    },
}


@pytest.fixture(params=list(SAMPLE_MESSAGES.keys()))
def sample_message(request):
    """Parametrized fixture yielding each sample message."""
    key = request.param
    return {**SAMPLE_MESSAGES[key], "key": key}
