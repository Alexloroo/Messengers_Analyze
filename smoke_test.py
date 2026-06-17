"""Quick smoke test for LangSmith + LangGraph integration.

Run this script to verify that everything is configured correctly
before running the full test suite.

Usage:
    python smoke_test.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import _load_env_file, PROJECT_ROOT

_load_env_file(PROJECT_ROOT / ".env")


def _check(name: str, ok: bool, detail: str = "") -> bool:
    icon = "✅" if ok else "❌"
    msg = f"  {icon} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return ok


def check_env_vars() -> bool:
    """Check that all required environment variables are set."""
    print("\n🔍 Checking environment variables...\n")
    all_ok = True

    all_ok &= _check(
        "GROQ_API_KEY",
        bool(os.getenv("GROQ_API_KEY")),
        "set" if os.getenv("GROQ_API_KEY") else "MISSING — add to .env",
    )
    all_ok &= _check(
        "LANGSMITH_API_KEY",
        bool(os.getenv("LANGSMITH_API_KEY")),
        "set" if os.getenv("LANGSMITH_API_KEY") else "MISSING — get from https://smith.langchain.com/settings",
    )
    all_ok &= _check(
        "LANGSMITH_TRACING",
        os.getenv("LANGSMITH_TRACING", "").lower() == "true",
        os.getenv("LANGSMITH_TRACING", "not set"),
    )
    all_ok &= _check(
        "LANGSMITH_PROJECT",
        bool(os.getenv("LANGSMITH_PROJECT")),
        os.getenv("LANGSMITH_PROJECT", "not set (will default to 'default')"),
    )

    return all_ok


def check_langsmith_connection() -> bool:
    """Test LangSmith API connectivity."""
    print("\n🔗 Testing LangSmith connection...\n")
    try:
        from langsmith import Client
        client = Client()
        projects = list(client.list_projects(limit=3))
        return _check(
            "LangSmith API",
            True,
            f"connected, {len(projects)} project(s) found",
        )
    except Exception as e:
        return _check("LangSmith API", False, str(e))


def check_llm_call() -> bool:
    """Test a simple LLM call with tracing."""
    print("\n🤖 Testing LLM call with tracing...\n")
    try:
        from llm_provider import get_chat_model
        from langchain_core.messages import HumanMessage

        llm = get_chat_model(model="llama-3.1-8b-instant", temperature=0.0)
        response = llm.invoke([HumanMessage(content="Respond with exactly: OK")])

        return _check(
            "LLM Call (llama-3.1-8b-instant)",
            "ok" in response.content.lower(),
            f"response: '{response.content.strip()}'",
        )
    except Exception as e:
        return _check("LLM Call", False, str(e))


def check_graph_pipeline() -> bool:
    """Test the full analysis pipeline."""
    print("\n📊 Testing LangGraph pipeline...\n")
    try:
        import asyncio
        from daily_analysis import analyze_message

        result = asyncio.run(analyze_message(
            raw_text="Вышел новый релиз Python 3.13 с улучшенным GIL.",
            chat_title="Python News",
        ))

        ok = result.get("is_useful", False) and len(result.get("summary", "")) > 5
        return _check(
            "LangGraph Pipeline",
            ok,
            f"category={result.get('category')}, score={result.get('quality_score', 0):.2f}, "
            f"summary={result.get('summary', '')[:60]}...",
        )
    except Exception as e:
        return _check("LangGraph Pipeline", False, str(e))


def main():
    print("=" * 60)
    print("  Messengers_Analyze — Smoke Test")
    print("  LangSmith + LangGraph Integration")
    print("=" * 60)

    results = []

    results.append(check_env_vars())

    if os.getenv("LANGSMITH_API_KEY"):
        results.append(check_langsmith_connection())
    else:
        print("\n⏭️  Skipping LangSmith connection test (no API key)")
        results.append(False)

    if os.getenv("GROQ_API_KEY"):
        results.append(check_llm_call())
        results.append(check_graph_pipeline())
    else:
        print("\n⏭️  Skipping LLM and pipeline tests (no Groq key)")
        results.append(False)
        results.append(False)

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    if all(results):
        print(f"  🎉 All checks passed ({passed}/{total})!")
        print("  You can now run: pytest tests/ -v")
    else:
        print(f"  ⚠️  {passed}/{total} checks passed.")
        print("  Fix the issues above and re-run this script.")
    print("=" * 60)

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
