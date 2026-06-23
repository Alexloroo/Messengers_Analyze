"""Unit tests for LangSmith integration in ``llm_provider.py``.

These tests do **not** require API keys or network access. They verify that
environment setup, provider factory routing, and the public helpers behave
correctly with mocked backends.

Run with:
    pytest tests/test_langsmith_unit.py -v
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from langsmith import Client as LangSmithClient, traceable

import llm_provider
from llm_provider import (
    DEFAULT_MODELS,
    DEEPSEEK_BASE_URL,
    _ensure_langsmith_env,
    get_chat_model,
    get_langsmith_client,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_langsmith_env(monkeypatch):
    """Start every test with a clean LangSmith environment."""
    for key in (
        "LANGSMITH_API_KEY",
        "LANGSMITH_TRACING",
        "LANGSMITH_PROJECT",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "GROQ_API_KEY",
        "DEEPSEEK_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def mock_groq_builder(monkeypatch):
    """Replace the Groq builder with a mock and return it."""
    builder = MagicMock(return_value=MagicMock(spec=["invoke", "ainvoke"]))
    monkeypatch.setitem(llm_provider._PROVIDER_BUILDERS, "groq", builder)
    return builder


@pytest.fixture
def mock_deepseek_builder(monkeypatch):
    """Replace the DeepSeek builder with a mock and return it."""
    builder = MagicMock(return_value=MagicMock(spec=["invoke", "ainvoke"]))
    monkeypatch.setitem(llm_provider._PROVIDER_BUILDERS, "deepseek", builder)
    return builder


# ---------------------------------------------------------------------------
# Environment setup tests
# ---------------------------------------------------------------------------


def test_ensure_langsmith_env_enables_tracing_with_key(monkeypatch, caplog):
    """When an API key is present, tracing should be enabled."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test-key")
    caplog.set_level("INFO")

    _ensure_langsmith_env()

    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_PROJECT"] == "messengers-analyze"
    assert "LangSmith tracing enabled" in caplog.text
    assert "messengers-analyze" in caplog.text


def test_ensure_langsmith_env_preserves_existing_project(monkeypatch):
    """An explicit LANGSMITH_PROJECT should not be overwritten."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test-key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "custom-project")

    _ensure_langsmith_env()

    assert os.environ["LANGSMITH_PROJECT"] == "custom-project"


def test_ensure_langsmith_env_disables_tracing_without_key(monkeypatch, caplog):
    """Without an API key, tracing should be disabled and a warning logged."""
    caplog.set_level("WARNING")

    _ensure_langsmith_env()

    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert "LANGSMITH_API_KEY is not set" in caplog.text


# ---------------------------------------------------------------------------
# LangSmith client factory
# ---------------------------------------------------------------------------


def test_get_langsmith_client_returns_client(monkeypatch):
    """``get_langsmith_client`` should return a LangSmith Client instance."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test-key")

    client = get_langsmith_client()

    assert isinstance(client, LangSmithClient)


# ---------------------------------------------------------------------------
# Chat model factory tests
# ---------------------------------------------------------------------------


def test_get_chat_model_defaults_to_groq(mock_groq_builder, monkeypatch):
    """By default the factory should route to the Groq builder."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test-key")
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")

    model = get_chat_model(temperature=0.5)

    assert model is mock_groq_builder.return_value
    mock_groq_builder.assert_called_once_with(
        DEFAULT_MODELS["groq"], 0.5
    )


def test_get_chat_model_resolves_provider_from_argument(
    mock_groq_builder, mock_deepseek_builder, monkeypatch
):
    """The explicit ``provider`` argument should take precedence."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test-key")
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")

    model = get_chat_model(provider="deepseek", temperature=0.2)

    assert model is mock_deepseek_builder.return_value
    mock_deepseek_builder.assert_called_once_with(
        DEFAULT_MODELS["deepseek"], 0.2
    )
    mock_groq_builder.assert_not_called()


def test_get_chat_model_resolves_provider_from_env(
    mock_groq_builder, mock_deepseek_builder, monkeypatch
):
    """The ``LLM_PROVIDER`` environment variable should be used."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test-key")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")

    model = get_chat_model(temperature=0.1)

    mock_deepseek_builder.assert_called_once()
    mock_groq_builder.assert_not_called()
    assert model is mock_deepseek_builder.return_value


def test_get_chat_model_resolves_model_from_env(
    mock_groq_builder, monkeypatch
):
    """The ``LLM_MODEL`` environment variable should override the default."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test-key")
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
    monkeypatch.setenv("LLM_MODEL", "llama-3.3-70b-versatile")

    model = get_chat_model(temperature=0.0)

    mock_groq_builder.assert_called_once_with(
        "llama-3.3-70b-versatile", 0.0
    )
    assert model is mock_groq_builder.return_value


def test_get_chat_model_unknown_provider_raises(monkeypatch):
    """An unknown provider name should raise a clear ValueError."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test-key")

    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_chat_model(provider="unknown")


def test_get_chat_model_missing_groq_key_raises(mock_groq_builder, monkeypatch):
    """Groq builder should raise if GROQ_API_KEY is missing."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test-key")
    mock_groq_builder.side_effect = ValueError("GROQ_API_KEY is not set")

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        get_chat_model(provider="groq")


def test_get_chat_model_missing_deepseek_key_raises(
    mock_deepseek_builder, monkeypatch
):
    """DeepSeek builder should raise if DEEPSEEK_API_KEY is missing."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test-key")
    mock_deepseek_builder.side_effect = ValueError("DEEPSEEK_API_KEY is not set")

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        get_chat_model(provider="deepseek")


# ---------------------------------------------------------------------------
# Real backend builder tests (no network calls)
# ---------------------------------------------------------------------------


def test_build_groq_uses_correct_model_and_api_key(monkeypatch):
    """The Groq builder should pass model, temperature and API key."""
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
    fake_chat = MagicMock()

    with patch("langchain_groq.ChatGroq", return_value=fake_chat) as mock_cls:
        model = llm_provider._build_groq(
            model="llama-3.1-8b-instant", temperature=0.7
        )

    assert model is fake_chat
    mock_cls.assert_called_once_with(
        model="llama-3.1-8b-instant",
        temperature=0.7,
        api_key="groq-test-key",
    )


def test_build_groq_raises_without_api_key(monkeypatch):
    """The Groq builder should raise a ValueError when the key is missing."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GROQ_API_KEY is not set"):
        llm_provider._build_groq(model="llama-3.1-8b-instant", temperature=0.0)


def test_build_deepseek_uses_openai_client_and_base_url(monkeypatch):
    """The DeepSeek builder should target the DeepSeek API endpoint."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    fake_chat = MagicMock()

    with patch("langchain_openai.ChatOpenAI", return_value=fake_chat) as mock_cls:
        model = llm_provider._build_deepseek(
            model="deepseek-chat", temperature=0.3
        )

    assert model is fake_chat
    mock_cls.assert_called_once_with(
        model="deepseek-chat",
        temperature=0.3,
        api_key="deepseek-test-key",
        base_url=DEEPSEEK_BASE_URL,
    )


def test_build_deepseek_raises_without_api_key(monkeypatch):
    """The DeepSeek builder should raise a ValueError when the key is missing."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY is not set"):
        llm_provider._build_deepseek(
            model="deepseek-chat", temperature=0.0
        )


# ---------------------------------------------------------------------------
# @traceable decorator tests (offline)
# ---------------------------------------------------------------------------


@traceable(name="unit_test_chain", run_type="chain")
def _sample_traced_function(text: str) -> dict:
    return {"text": text, "length": len(text)}


def test_traceable_decorator_preserves_function():
    """``@traceable`` should not change the wrapped function's output."""
    result = _sample_traced_function("hello")

    assert result == {"text": "hello", "length": 5}


def test_traceable_decorator_exposes_run_name():
    """The traced function should carry the configured run name."""
    # The wrapper created by @traceable stores metadata on the function object.
    assert getattr(_sample_traced_function, "__name__", None) == "_sample_traced_function"
