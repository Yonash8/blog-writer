"""Closed intent enum for the article workflow router."""

from enum import Enum


class Intent(str, Enum):
    # Article workflow (full pipeline)
    START_ARTICLE = "start_article"
    EDIT_SECTION = "edit_section"
    EDIT_LAST_SECTION = "edit_last_section"
    REWRITE_WITH_CONSTRAINTS = "rewrite_with_constraints"
    REGENERATE_DRAFT = "regenerate_draft"
    APPROVE_DRAFT = "approve_draft"
    PUBLISH_ARTICLE = "publish_article"
    INJECT_TO_DOCS = "inject_to_docs"

    # Hero image (standalone or part of pipeline)
    GENERATE_HERO = "generate_hero"
    REGENERATE_HERO = "regenerate_hero"
    APPROVE_HERO = "approve_hero"

    # Visuals (hero + infographic pipeline) - "let's do visuals for X", "add images to X"
    GENERATE_VISUALS = "generate_visuals"

    # Infographic (standalone or part of pipeline)
    GENERATE_INFOGRAPHIC = "generate_infographic"
    REGENERATE_INFOGRAPHIC = "regenerate_infographic"
    APPROVE_INFOGRAPHIC = "approve_infographic"

    # Generic images (bulk placement images)
    GENERATE_IMAGES = "generate_images"
    APPROVE_IMAGES = "approve_images"

    # Read-only / meta
    SHOW_STATUS = "show_status"
    SHOW_OUTLINE = "show_outline"
    SHOW_ARTICLE_INFO = "show_article_info"
    LIST_ARTICLES = "list_articles"

    # General / control
    GENERAL_QUESTION = "general_question"
    SWITCH_ARTICLE = "switch_article"
    CANCEL_ARTICLE = "cancel_article"
    HELP = "help"
    UNKNOWN = "unknown"


# Intents that don't touch article state (handled by pre-router)
NON_ARTICLE_INTENTS = frozenset({
    Intent.GENERAL_QUESTION,
    Intent.HELP,
    Intent.LIST_ARTICLES,
})

# Intents requiring an active article
REQUIRES_ARTICLE_INTENTS = frozenset({
    Intent.EDIT_SECTION,
    Intent.EDIT_LAST_SECTION,
    Intent.REWRITE_WITH_CONSTRAINTS,
    Intent.REGENERATE_DRAFT,
    Intent.APPROVE_DRAFT,
    Intent.GENERATE_HERO,
    Intent.GENERATE_VISUALS,
    Intent.REGENERATE_HERO,
    Intent.APPROVE_HERO,
    Intent.GENERATE_INFOGRAPHIC,
    Intent.REGENERATE_INFOGRAPHIC,
    Intent.APPROVE_INFOGRAPHIC,
    Intent.GENERATE_IMAGES,
    Intent.APPROVE_IMAGES,
    Intent.PUBLISH_ARTICLE,
    Intent.INJECT_TO_DOCS,
    Intent.SHOW_STATUS,
    Intent.SHOW_OUTLINE,
})

ALL_INTENT_VALUES = frozenset(i.value for i in Intent)
