"""Research node: optional Tavily enrichment (deep research is handled inside PromptLayer)."""

import logging
from typing import Any

from src.graph.helpers import log_action
from src.graph.state import ArticleState

logger = logging.getLogger(__name__)


def research_node(state: ArticleState) -> dict[str, Any]:
    """Optionally run Tavily enrichment. Deep research is now done inside PromptLayer."""
    from src.pipeline import run_tavily_enrichment, format_tavily_for_research, emit_status

    topic = state.get("intent_params", {}).get("topic", state.get("user_message", ""))
    include_tavily = state.get("include_tavily", False)

    if include_tavily:
        emit_status("Getting sources from Tavily...")
        logger.info("[RESEARCH] Tavily enrichment for topic: %s", topic[:100])
        tavily_results = run_tavily_enrichment(topic, max_results=20)
        tavily_text = format_tavily_for_research(tavily_results)
        research_text = f"## Sources (Tavily Enrichment)\n\n{tavily_text}"
        sources = [
            {"url": r.get("url", ""), "title": r.get("title", ""), "snippet": r.get("content", "")[:200]}
            for r in tavily_results[:10]
        ]
        logger.info("[RESEARCH] Tavily done, sources=%d", len(tavily_results))
        return {
            "stage": "synthesizing",
            "research_text": research_text,
            "sources": sources,
            "response_to_user": f"Got {len(tavily_results)} sources. Writing article...",
            "actions_log": log_action("research", f"topic={topic[:80]} tavily_sources={len(tavily_results)}"),
        }
    else:
        logger.info("[RESEARCH] No Tavily requested, passing topic directly to PromptLayer")
        return {
            "stage": "synthesizing",
            "research_text": "",
            "sources": [],
            "response_to_user": "Writing article...",
            "actions_log": log_action("research", f"topic={topic[:80]} tavily=False"),
        }
