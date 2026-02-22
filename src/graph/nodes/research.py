"""Research node: wraps deep research + Tavily enrichment from pipeline.py."""

import logging
from typing import Any

from src.graph.helpers import log_action
from src.graph.state import ArticleState

logger = logging.getLogger(__name__)


def research_node(state: ArticleState) -> dict[str, Any]:
    """Run deep research + Tavily enrichment for a topic."""
    from src.pipeline import run_deep_research, run_tavily_enrichment, format_tavily_for_research, emit_status

    emit_status("Starting research...")
    topic = state.get("intent_params", {}).get("topic", state.get("user_message", ""))

    logger.info("[RESEARCH] Starting research for topic: %s", topic[:100])

    research_text = run_deep_research(topic)
    logger.info("[RESEARCH] Deep research done, len=%d", len(research_text))

    tavily_results = run_tavily_enrichment(topic, max_results=20)
    tavily_text = format_tavily_for_research(tavily_results)
    combined = f"{research_text}\n\n## Additional Sources (Tavily Enrichment)\n\n{tavily_text}"
    logger.info("[RESEARCH] Tavily done, sources=%d, combined_len=%d", len(tavily_results), len(combined))

    # Extract compact sources for state
    sources = [
        {"url": r.get("url", ""), "title": r.get("title", ""), "snippet": r.get("content", "")[:200]}
        for r in tavily_results[:10]
    ]

    return {
        "stage": "synthesizing",
        "research_text": combined,
        "sources": sources,
        "response_to_user": f"Research complete. Found {len(tavily_results)} sources. Synthesizing outline...",
        "actions_log": log_action("research", f"topic={topic[:80]} sources={len(tavily_results)}"),
    }
