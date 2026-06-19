"""LLM provider module with multi-provider support and LangSmith tracing.

Supports:
  - Groq  (fast inference, llama models)
  - DeepSeek (deepseek-chat, deepseek-reasoner)

Provider is selected via the ``provider`` argument or the ``LLM_PROVIDER``
environment variable.  All calls are automatically traced in LangSmith.
"""

import os
import logging
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langsmith import Client as LangSmithClient

from config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# Supported provider names
ProviderName = Literal["groq", "deepseek"]

# Default models per provider
DEFAULT_MODELS: dict[str, str] = {
    "groq": "llama-3.1-8b-instant",
    "deepseek": "deepseek-chat",
}

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


# ---------------------------------------------------------------------------
# LangSmith helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Provider factories
# ---------------------------------------------------------------------------

def _build_groq(model: str, temperature: float, **kwargs) -> BaseChatModel:
    """Create a ChatGroq instance."""
    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Get your free key at https://console.groq.com/keys"
        )

    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=api_key,
        **kwargs,
    )


def _build_deepseek(model: str, temperature: float, **kwargs) -> BaseChatModel:
    """Create a ChatOpenAI instance pointing at the DeepSeek API."""
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY is not set. "
            "Get your key at https://platform.deepseek.com/api_keys"
        )

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
        **kwargs,
    )


_PROVIDER_BUILDERS = {
    "groq": _build_groq,
    "deepseek": _build_deepseek,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_chat_model(
    model: str | None = None,
    temperature: float = 0.0,
    provider: ProviderName | None = None,
    **kwargs,
) -> BaseChatModel:
    """Create a chat model with LangSmith tracing auto-enabled.

    The provider is resolved in this order:
      1. Explicit ``provider`` argument
      2. ``LLM_PROVIDER`` environment variable
      3. Falls back to ``"groq"``

    If ``model`` is not specified, uses the default model for the provider.

    Args:
        model: Model name (e.g. "llama-3.1-8b-instant", "deepseek-chat").
        temperature: Sampling temperature (default: 0.0 for determinism).
        provider: Which LLM backend to use ("groq" or "deepseek").
        **kwargs: Additional arguments forwarded to the underlying client.

    Returns:
        Configured chat model instance.

    Raises:
        ValueError: If the provider is unknown or the API key is missing.
    """
    _ensure_langsmith_env()

    # Resolve provider
    provider = provider or os.getenv("LLM_PROVIDER", "groq").lower()  # type: ignore[assignment]
    if provider not in _PROVIDER_BUILDERS:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            f"Supported: {', '.join(_PROVIDER_BUILDERS)}"
        )

    # Resolve model
    model = model or os.getenv("LLM_MODEL") or DEFAULT_MODELS[provider]

    logger.info("Creating %s model: %s (temperature=%.1f)", provider, model, temperature)
    return _PROVIDER_BUILDERS[provider](model, temperature, **kwargs)


def get_langsmith_client() -> LangSmithClient:
    """Return a LangSmith client for programmatic access (datasets, evaluations, etc.)."""
    _ensure_langsmith_env()
    return LangSmithClient()
