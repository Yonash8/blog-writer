"""LangGraph StateGraph definition: nodes, edges, checkpointer.

This is the main graph that replaces the old agent.py tool-calling loop.
"""

import logging
import os
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.graph.intents import Intent
from src.graph.state import ArticleState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import all node functions
# ---------------------------------------------------------------------------

from src.graph.router import router_node
from src.graph.nodes.article_resolver import article_resolver_node, SKIP_RESOLVER_INTENTS
from src.graph.nodes.confirmation import confirmation_gate
from src.graph.nodes.general import general_question_node
from src.graph.nodes.status import help_node, list_articles_node, show_article_info_node, show_status_node
from src.graph.nodes.research import research_node
from src.graph.nodes.synthesis import synthesis_node
from src.graph.nodes.writer import writer_node
from src.graph.nodes.draft_approval import draft_approval_gate
from src.graph.nodes.editor import editor_node
from src.graph.nodes.hero import hero_approval_gate, hero_description_gate, hero_generator_node
from src.graph.nodes.infographic import (
    infographic_analysis_node,
    infographic_approval_gate,
    infographic_generator_node,
)
from src.graph.nodes.publisher import publisher_node, qa_node


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def route_after_router(state: ArticleState) -> str:
    """Route from router to article_resolver (for article intents) or directly to action nodes."""
    intent = state.get("intent", "unknown")

    # Intents that skip the article resolver entirely
    if intent == Intent.GENERAL_QUESTION:
        return "general_question"
    if intent == Intent.HELP:
        return "help"
    if intent == Intent.LIST_ARTICLES:
        return "list_articles"
    if intent == Intent.START_ARTICLE:
        return "confirmation_gate"
    if intent == Intent.UNKNOWN:
        return "general_question"

    # All other intents go through article_resolver first
    return "article_resolver"


def route_after_resolver(state: ArticleState) -> str:
    """Route from article_resolver to the correct action node.

    If the resolver set intent to 'unknown' (couldn't find article, user rejected),
    the flow ends. Otherwise, route to the original action node.
    """
    intent = state.get("intent", "unknown")

    # Resolver may have reset intent to unknown (article not found / user rejected)
    if intent == Intent.UNKNOWN or intent == "unknown":
        return END

    # Show info / status (may or may not have article_id)
    if intent in (Intent.SHOW_STATUS, Intent.SHOW_OUTLINE):
        return "show_status"
    if intent == Intent.SHOW_ARTICLE_INFO:
        return "show_article_info"

    # Edit intents
    if intent in (Intent.EDIT_SECTION, Intent.EDIT_LAST_SECTION,
                  Intent.REWRITE_WITH_CONSTRAINTS, Intent.REGENERATE_DRAFT):
        return "editor"

    # Draft approval
    if intent == Intent.APPROVE_DRAFT:
        return "draft_approval_gate"

    # Hero image / visuals
    if intent in (Intent.GENERATE_HERO, Intent.GENERATE_VISUALS):
        return "hero_description_gate"
    if intent == Intent.REGENERATE_HERO:
        return "hero_generator"
    if intent == Intent.APPROVE_HERO:
        return "hero_description_gate"

    # Infographic
    if intent in (Intent.GENERATE_INFOGRAPHIC, Intent.REGENERATE_INFOGRAPHIC):
        return "infographic_analysis"
    if intent == Intent.APPROVE_INFOGRAPHIC:
        return "infographic_analysis"

    # Publishing
    if intent in (Intent.PUBLISH_ARTICLE, Intent.INJECT_TO_DOCS):
        return "publisher"

    # Cancel / switch
    if intent == Intent.CANCEL_ARTICLE:
        return "show_status"
    if intent == Intent.SWITCH_ARTICLE:
        return "show_status"

    return END


def route_after_confirmation(state: ArticleState) -> str:
    """After confirmation: proceed to research or end if cancelled."""
    if state.get("stage") == "researching":
        return "research"
    return "end"


def route_after_draft_approval(state: ArticleState) -> str:
    """After draft approval gate, route based on stage."""
    stage = state.get("stage", "")
    if stage == "editing":
        return "editor"
    return "hero_description_gate"


def route_after_hero_generator(state: ArticleState) -> str:
    """After hero generator: go to approval gate if image was created, END on failure."""
    if state.get("hero_image_url") and state.get("stage") == "awaiting_hero_approval":
        return "hero_approval_gate"
    # Generation failed (no article, API error, etc.) — stage is idle or awaiting_hero_description
    return END


def route_after_hero_approval(state: ArticleState) -> str:
    """After hero approval gate: re-iterate or move to infographic."""
    stage = state.get("stage", "")
    if stage == "generating_hero":
        return "hero_generator"
    return "infographic_analysis"


def route_after_infographic_generator(state: ArticleState) -> str:
    """After infographic generator: go to approval gate if image was created, END on failure."""
    if state.get("infographic_image_url") and state.get("stage") == "awaiting_infographic_approval":
        return "infographic_approval_gate"
    return END


def route_after_infographic_approval(state: ArticleState) -> str:
    """After infographic approval: re-iterate or move to publisher."""
    stage = state.get("stage", "")
    if stage == "generating_infographic":
        return "infographic_generator"
    return "publisher"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None) -> StateGraph:
    """Construct and compile the article workflow graph."""

    graph = StateGraph(ArticleState)

    # Add all nodes
    graph.add_node("router", router_node)
    graph.add_node("article_resolver", article_resolver_node)
    graph.add_node("confirmation_gate", confirmation_gate)
    graph.add_node("general_question", general_question_node)
    graph.add_node("help", help_node)
    graph.add_node("list_articles", list_articles_node)
    graph.add_node("show_status", show_status_node)
    graph.add_node("show_article_info", show_article_info_node)
    graph.add_node("research", research_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("writer", writer_node)
    graph.add_node("draft_approval_gate", draft_approval_gate)
    graph.add_node("editor", editor_node)
    graph.add_node("hero_description_gate", hero_description_gate)
    graph.add_node("hero_generator", hero_generator_node)
    graph.add_node("hero_approval_gate", hero_approval_gate)
    graph.add_node("infographic_analysis", infographic_analysis_node)
    graph.add_node("infographic_generator", infographic_generator_node)
    graph.add_node("infographic_approval_gate", infographic_approval_gate)
    graph.add_node("publisher", publisher_node)
    graph.add_node("qa", qa_node)

    # Entry point
    graph.set_entry_point("router")

    # Router -> conditional edges (some go to article_resolver, some direct)
    graph.add_conditional_edges("router", route_after_router)

    # Article resolver -> conditional edges to action nodes
    graph.add_conditional_edges("article_resolver", route_after_resolver)

    # Terminal nodes (go to END)
    graph.add_edge("general_question", END)
    graph.add_edge("help", END)
    graph.add_edge("list_articles", END)
    graph.add_edge("show_status", END)
    graph.add_edge("show_article_info", END)

    # Confirmation gate -> research or END
    graph.add_conditional_edges(
        "confirmation_gate",
        route_after_confirmation,
        {"research": "research", "end": END},
    )

    # Article pipeline: research -> synthesis -> writer -> draft_approval_gate (sub-steps: status only, no approval)
    graph.add_edge("research", "synthesis")
    graph.add_edge("synthesis", "writer")
    graph.add_edge("writer", "draft_approval_gate")

    # Draft approval -> conditional (approve -> hero, edit -> editor)
    graph.add_conditional_edges("draft_approval_gate", route_after_draft_approval)

    # Editor -> back to draft approval gate
    graph.add_edge("editor", "draft_approval_gate")

    # Hero flow: description_gate -> generator -> (approval_gate or END on failure)
    graph.add_edge("hero_description_gate", "hero_generator")
    graph.add_conditional_edges("hero_generator", route_after_hero_generator)

    # Hero approval: conditional (re-iterate or continue)
    graph.add_conditional_edges("hero_approval_gate", route_after_hero_approval)

    # Infographic flow: analysis -> generator -> (approval_gate or END on failure)
    graph.add_edge("infographic_analysis", "infographic_generator")
    graph.add_conditional_edges("infographic_generator", route_after_infographic_generator)

    # Infographic approval: conditional (re-iterate or continue)
    graph.add_conditional_edges("infographic_approval_gate", route_after_infographic_approval)

    # Publisher -> QA -> END
    graph.add_edge("publisher", "qa")
    graph.add_edge("qa", END)

    # Compile with checkpointer
    if checkpointer is None:
        checkpointer = MemorySaver()

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("[GRAPH] Article workflow graph compiled with %d nodes", len(graph.nodes))
    return compiled


# ---------------------------------------------------------------------------
# Singleton graph instance
# ---------------------------------------------------------------------------

_compiled_graph = None


def get_graph():
    """Get or create the singleton compiled graph."""
    global _compiled_graph
    if _compiled_graph is None:
        checkpointer = _create_checkpointer()
        _compiled_graph = build_graph(checkpointer=checkpointer)
    return _compiled_graph


def _create_checkpointer():
    """Create the appropriate checkpointer based on environment."""
    checkpointer_type = os.getenv("LANGGRAPH_CHECKPOINTER", "sqlite")

    if checkpointer_type == "sqlite":
        try:
            import sqlite3
            from langgraph.checkpoint.sqlite import SqliteSaver

            db_path = os.getenv("LANGGRAPH_SQLITE_PATH", "langgraph.db")
            conn = sqlite3.connect(db_path, check_same_thread=False)
            logger.info("[GRAPH] Using SQLite checkpointer: %s", db_path)
            return SqliteSaver(conn)
        except ImportError:
            logger.warning("[GRAPH] langgraph-checkpoint-sqlite not installed, falling back to memory")
            return MemorySaver()
    elif checkpointer_type == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            conn_string = os.getenv("LANGGRAPH_POSTGRES_URL", "")
            if not conn_string:
                logger.warning("[GRAPH] LANGGRAPH_POSTGRES_URL not set, falling back to memory")
                return MemorySaver()
            logger.info("[GRAPH] Using Postgres checkpointer")
            return PostgresSaver.from_conn_string(conn_string)
        except ImportError:
            logger.warning("[GRAPH] langgraph-checkpoint-postgres not installed, falling back to memory")
            return MemorySaver()
    else:
        logger.info("[GRAPH] Using in-memory checkpointer")
        return MemorySaver()
