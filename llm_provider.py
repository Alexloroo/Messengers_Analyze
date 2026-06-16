"""LLM provider module with LangSmith tracing integration.

Centralizes LLM client creation and ensures all calls are traced
in LangSmith for monitoring, debugging, and evaluation.
"""

import os
import logging

from langchain_groq import ChatGroq
from langsmith import Client as LangSmithClient

from config import PROJECT_ROOT

logger = logging.getLogger(__name__)


def _ensure_langsmith_env() -> None:
    """Verify that LangSmith environment variables are set and enable tracing."""
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        logger.warning(
            "LANGSMITH_API_KEY is not set. Tracing will be disabled. "
            "Get your key at https://smith.langchain.com/settings"
        )
        os.environ.setdefault("LANGSMITH_TRACING", "false")
        return

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "messengers-analyze")
    logger.info(
        "LangSmith tracing enabled for project: %s",
        os.environ["LANGSMITH_PROJECT"],
    )


def get_chat_model(
    model: str = "llama-3.1-8b-instant",
    temperature: float = 0.0,
    **kwargs,
) -> ChatGroq:
    """Create a ChatGroq instance with LangSmith tracing auto-enabled.

    All calls made through the returned model will be automatically
    traced and visible in the LangSmith dashboard.

    Args:
        model: Groq model name (default: llama-3.1-8b-instant for speed/cost).
        temperature: Sampling temperature (default: 0.0 for determinism).
        **kwargs: Additional arguments forwarded to ChatGroq.

    Returns:
        Configured ChatGroq instance.
    """
    _ensure_langsmith_env()

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )

    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=groq_api_key,
        **kwargs,
    )


def get_langsmith_client() -> LangSmithClient:
    """Return a LangSmith client for programmatic access (datasets, evaluations, etc.)."""
    _ensure_langsmith_env()
    return LangSmithClient()
