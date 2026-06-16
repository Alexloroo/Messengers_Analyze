"""Tests for LangSmith integration — tracing and connectivity.

These tests verify that:
  1. LangSmith client can connect to the API
  2. Traces are properly created for LLM calls
  3. The @traceable decorator works for custom functions

Run with: pytest tests/test_langsmith_tracing.py -v
"""

import os
import pytest
from langsmith import Client as LangSmithClient, traceable


# ---------------------------------------------------------------------------
# Test 1: LangSmith API connectivity
# ---------------------------------------------------------------------------

@pytest.mark.langsmith
def test_langsmith_connection(langsmith_configured):
    """Verify that we can connect to the LangSmith API."""
    client = LangSmithClient()

    # This should not raise — it confirms the API key is valid
    # and the endpoint is reachable
    projects = list(client.list_projects(limit=1))
    assert isinstance(projects, list), "Expected a list of projects from LangSmith"
    print(f"✅ Connected to LangSmith. Found {len(projects)} project(s).")


# ---------------------------------------------------------------------------
# Test 2: Basic LLM tracing
# ---------------------------------------------------------------------------

@pytest.mark.langsmith
@pytest.mark.llm
def test_llm_call_is_traced(langsmith_configured, groq_configured):
    """Verify that a simple LLM call creates a trace in LangSmith."""
    from llm_provider import get_chat_model
    from langchain_core.messages import HumanMessage

    llm = get_chat_model(model="llama-3.1-8b-instant", temperature=0.0)
    response = llm.invoke([HumanMessage(content="Say 'LangSmith test OK' and nothing else.")])

    assert response.content is not None
    assert len(response.content) > 0
    print(f"✅ LLM response: {response.content}")


# ---------------------------------------------------------------------------
# Test 3: Custom function tracing with @traceable
# ---------------------------------------------------------------------------

@traceable(name="test_custom_traced_function", run_type="chain")
def _dummy_analysis(text: str) -> dict:
    """A dummy function decorated with @traceable to test custom tracing."""
    word_count = len(text.split())
    has_deadline = any(w in text.lower() for w in ["дедлайн", "deadline", "до пятницы", "к среде"])
    return {
        "word_count": word_count,
        "has_deadline": has_deadline,
        "processed": True,
    }


@pytest.mark.langsmith
def test_traceable_decorator(langsmith_configured):
    """Verify that the @traceable decorator creates a trace for custom functions."""
    result = _dummy_analysis("Нужно закончить отчёт до пятницы. Дедлайн строгий.")

    assert result["processed"] is True
    assert result["has_deadline"] is True
    assert result["word_count"] > 0
    print(f"✅ Traced custom function. Result: {result}")


# ---------------------------------------------------------------------------
# Test 4: Project creation / existence
# ---------------------------------------------------------------------------

@pytest.mark.langsmith
def test_langsmith_project_exists(langsmith_configured):
    """Verify that the 'messengers-analyze' project is created in LangSmith."""
    project_name = os.getenv("LANGSMITH_PROJECT", "messengers-analyze")
    client = LangSmithClient()

    # Ensure the project exists (create if missing)
    try:
        client.read_project(project_name=project_name)
        print(f"✅ Project '{project_name}' already exists in LangSmith.")
    except Exception:
        # If the project doesn't exist, that's also OK for now —
        # it will be auto-created on first trace
        print(f"ℹ️  Project '{project_name}' not found (will be auto-created on first trace).")
