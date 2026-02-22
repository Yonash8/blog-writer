"""LangGraph state schema for the article workflow."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Optional, TypedDict


# Stages the article workflow can be in
Stage = Literal[
    "idle",
    "researching",
    "synthesizing",
    "drafting",
    "awaiting_draft_approval",
    "editing",
    "awaiting_hero_description",
    "generating_hero",
    "awaiting_hero_approval",
    "analyzing_infographic",
    "generating_infographic",
    "awaiting_infographic_approval",
    "publishing",
    "published",
]


def _replace(existing: Any, new: Any) -> Any:
    """Reducer that always takes the new value (standard overwrite)."""
    return new


def _merge_dict(existing: dict | None, new: dict | None) -> dict:
    """Reducer that merges dicts (new keys overwrite existing)."""
    base = dict(existing) if existing else {}
    if new:
        base.update(new)
    return base


def _append_log(existing: list | None, new: list | None) -> list:
    """Reducer that appends to audit log."""
    base = list(existing) if existing else []
    if new:
        base.extend(new)
    return base


class ArticleState(TypedDict, total=False):
    """State for a single article workflow thread.

    Large blobs (article content, research) live in Supabase.
    State holds IDs, compact artifacts, and workflow control fields.
    """

    # --- Input (set each invocation) ---
    user_message: str
    whatsapp_user_id: str

    # --- Routing ---
    intent: str
    intent_params: dict

    # --- Chat context (recent messages for conversational tone) ---
    recent_messages: list  # [{role, content}] injected from Supabase

    # --- Article identity ---
    article_id: Optional[str]
    resolved_article_title: Optional[str]  # title of the resolved article (set by article_resolver)
    stage: Stage

    # --- Supabase references (keep state small) ---
    research_bundle_id: Optional[str]
    research_text: Optional[str]  # combined research text (passed to writer)
    outline_id: Optional[str]
    draft_id: Optional[str]
    draft_version: int

    # --- Compact research artifacts ---
    sources: list       # [{url, title, snippet}]
    key_claims: list    # [str]

    # --- Approvals ---
    approval_needed: Optional[str]
    approvals: Annotated[dict, _merge_dict]

    # --- Hero image ---
    hero_description: Optional[str]
    hero_image_id: Optional[str]
    hero_image_url: Optional[str]
    hero_feedback: Optional[str]
    hero_attempt: int

    # --- Infographic ---
    infographic_analysis: Optional[dict]
    infographic_image_id: Optional[str]
    infographic_image_url: Optional[str]
    infographic_feedback: Optional[str]
    infographic_attempt: int

    # --- Publishing ---
    doc_id: Optional[str]
    doc_url: Optional[str]

    # --- Output to user ---
    response_to_user: str

    # --- Audit ---
    actions_log: Annotated[list, _append_log]
