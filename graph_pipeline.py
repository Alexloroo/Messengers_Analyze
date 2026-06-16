"""LangGraph pipeline for message analysis.

Implements the batch analysis graph from Этап 2 of the project plan:
  1. Filter node — removes spam/noise
  2. Classify node — categorizes message (news, discussion, task, etc.)
  3. Summarize node — generates a concise summary
  4. Critic node — validates summary quality; loops back if needed

All nodes are traced in LangSmith automatically via the LangChain integration.
"""

from __future__ import annotations

import logging
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from llm_provider import get_chat_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class MessageAnalysisState(TypedDict):
    """State flowing through the analysis graph."""

    # Input
    raw_text: str
    chat_title: str

    # Processing outputs
    is_spam: bool
    category: str          # news | discussion | task | announcement | other
    summary: str
    quality_score: float   # 0.0 — 1.0
    quality_feedback: str
    retry_count: int

    # Final
    is_useful: bool


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

MAX_RETRIES = 2


def filter_node(state: MessageAnalysisState) -> dict:
    """Determine if the message is spam/noise or potentially useful."""
    llm = get_chat_model(temperature=0.0)

    response = llm.invoke([
        SystemMessage(content=(
            "You are a spam filter for Telegram messages. "
            "Respond with EXACTLY one word: 'spam' or 'useful'. "
            "Mark as spam: ads, promotions, empty forwards, stickers-only, "
            "join/leave notifications, bot commands. "
            "Mark as useful: news, discussions, tasks, announcements, articles."
        )),
        HumanMessage(content=f"Chat: {state['chat_title']}\nMessage: {state['raw_text']}"),
    ])

    is_spam = "spam" in response.content.strip().lower()
    logger.info("Filter result for message: %s", "spam" if is_spam else "useful")
    return {"is_spam": is_spam}


def classify_node(state: MessageAnalysisState) -> dict:
    """Classify the message into a category."""
    llm = get_chat_model(temperature=0.0)

    response = llm.invoke([
        SystemMessage(content=(
            "Classify the following Telegram message into EXACTLY one category. "
            "Respond with ONLY the category name, nothing else.\n"
            "Categories: news, discussion, task, announcement, article, other"
        )),
        HumanMessage(content=f"Chat: {state['chat_title']}\nMessage: {state['raw_text']}"),
    ])

    category = response.content.strip().lower()
    valid_categories = {"news", "discussion", "task", "announcement", "article", "other"}
    if category not in valid_categories:
        category = "other"

    logger.info("Classified message as: %s", category)
    return {"category": category}


def summarize_node(state: MessageAnalysisState) -> dict:
    """Generate a concise summary of the message."""
    llm = get_chat_model(temperature=0.3)

    response = llm.invoke([
        SystemMessage(content=(
            "Summarize the following Telegram message in 1-2 sentences in Russian. "
            "Focus on the key information: who, what, when, action items. "
            "Be concise and factual."
        )),
        HumanMessage(content=(
            f"Chat: {state['chat_title']}\n"
            f"Category: {state['category']}\n"
            f"Message:\n{state['raw_text']}"
        )),
    ])

    summary = response.content.strip()
    logger.info("Generated summary: %s", summary[:80])
    return {"summary": summary, "retry_count": state.get("retry_count", 0)}


def critic_node(state: MessageAnalysisState) -> dict:
    """Evaluate the quality of the generated summary.

    Checks for:
      - Accuracy (no hallucinations)
      - Completeness (key facts preserved)
      - Conciseness
    """
    llm = get_chat_model(temperature=0.0)

    response = llm.invoke([
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


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def after_filter(state: MessageAnalysisState) -> str:
    """Route after spam filter: skip spam, continue with useful messages."""
    if state.get("is_spam", False):
        return "end_spam"
    return "classify"


def after_critic(state: MessageAnalysisState) -> str:
    """Route after critic: accept if quality is good or retries exhausted."""
    score = state.get("quality_score", 1.0)
    retries = state.get("retry_count", 0)

    if score >= 0.7 or retries >= MAX_RETRIES:
        return "accept"
    return "retry_summarize"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_analysis_graph() -> StateGraph:
    """Build and compile the message analysis graph.

    Flow:
        START → filter → [spam? → END] / [useful? → classify → summarize → critic]
                                                                    ↑         |
                                                                    └─ retry ─┘
                                                                          → END
    """
    graph = StateGraph(MessageAnalysisState)

    # Add nodes
    graph.add_node("filter", filter_node)
    graph.add_node("classify", classify_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("critic", critic_node)

    # Mark spam as not useful and exit
    graph.add_node("mark_spam", lambda state: {"is_useful": False, "summary": "", "category": "spam"})

    # Increment retry counter before re-summarizing
    graph.add_node("bump_retry", lambda state: {"retry_count": state.get("retry_count", 0) + 1})

    # Edges
    graph.add_edge(START, "filter")

    graph.add_conditional_edges(
        "filter",
        after_filter,
        {
            "end_spam": "mark_spam",
            "classify": "classify",
        },
    )

    graph.add_edge("mark_spam", END)
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


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def analyze_message(raw_text: str, chat_title: str = "Unknown Chat") -> MessageAnalysisState:
    """Run a single message through the full analysis pipeline.

    Args:
        raw_text: The message text to analyze.
        chat_title: Name of the Telegram chat the message came from.

    Returns:
        Final state dict with category, summary, quality_score, etc.
    """
    graph = build_analysis_graph()
    initial_state: MessageAnalysisState = {
        "raw_text": raw_text,
        "chat_title": chat_title,
        "is_spam": False,
        "category": "",
        "summary": "",
        "quality_score": 0.0,
        "quality_feedback": "",
        "retry_count": 0,
        "is_useful": False,
    }

    result = graph.invoke(initial_state)
    return result
