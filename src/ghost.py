"""Ghost CMS Admin API client for creating draft posts."""
import json
import logging
import os
import re
import time
from typing import Any, Optional

import httpx
import jwt
import markdown2

logger = logging.getLogger(__name__)

GHOST_ADMIN_URL = os.getenv("GHOST_ADMIN_URL", "").rstrip("/")
GHOST_ADMIN_API_KEY = os.getenv("GHOST_ADMIN_API_KEY", "")
GHOST_AUTHOR_EMAIL = os.getenv("GHOST_AUTHOR_EMAIL", "")

# ---------------------------------------------------------------------------
# Metadata normalisation
# ---------------------------------------------------------------------------

# Ordered lists of aliases for each canonical Ghost field.
# The first key that is present and non-empty wins.
_METADATA_ALIASES: dict[str, list[str]] = {
    "title":              ["title", "seo_title", "page_title", "post_title"],
    "slug":               ["slug", "url_slug", "post_slug"],
    "meta_title":         ["meta_title", "seo_title", "title"],
    "meta_description":   ["meta_description", "description", "seo_description",
                           "meta_desc", "seo_desc"],
    "excerpt":            ["excerpt", "summary", "custom_excerpt",
                           "short_description", "intro"],
    "tags":               ["tags", "keywords", "categories", "tag_list"],
    "codeinjection_head": ["codeinjection_head", "code_injection_head"],
    "canonical_url":      ["canonical_url", "canonical"],
}


def _normalize_seo_metadata(raw: Any) -> dict[str, Any]:
    """Normalise PromptLayer metadata output into a clean Ghost-ready dict.

    Handles:
    - JSON strings (with or without markdown code fences)
    - Single-key wrapper dicts like {"raw": "..."} or {"value": {...}}
    - Nested wrappers like {"seo": {...}} or {"metadata": {...}}
    - Key-name aliases across different metadata agent conventions
    - Tags as a comma-separated string, a list of strings, or a list of dicts
    """
    # --- Step 1: ensure we have a plain dict ---
    if isinstance(raw, str):
        s = raw.strip()
        # Strip markdown code fences (```json ... ``` or ``` ... ```)
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s.rstrip())
        try:
            raw = json.loads(s)
        except Exception:
            logger.warning("[GHOST] _normalize_seo_metadata: could not JSON-parse metadata string: %r", s[:200])
            return {}

    if not isinstance(raw, dict):
        logger.warning("[GHOST] _normalize_seo_metadata: expected dict, got %s", type(raw).__name__)
        return {}

    # --- Step 2: unwrap single-key container dicts ---
    # e.g. {"raw": "json-string"} or {"value": {...}}
    if set(raw.keys()) <= {"raw"} and isinstance(raw.get("raw"), str):
        try:
            raw = json.loads(raw["raw"])
        except Exception:
            return {}
    elif set(raw.keys()) == {"value"} and isinstance(raw.get("value"), dict):
        raw = raw["value"]

    if not isinstance(raw, dict):
        return {}

    # --- Step 3: unwrap top-level nesting ---
    # e.g. {"seo": {...}} or {"metadata": {...}}
    for wrapper in ("seo", "metadata", "seo_metadata"):
        if wrapper in raw and isinstance(raw[wrapper], dict) and len(raw) <= 2:
            raw = raw[wrapper]
            break

    # --- Step 4: map aliases → canonical keys ---
    result: dict[str, Any] = {}
    for canonical, aliases in _METADATA_ALIASES.items():
        for alias in aliases:
            val = raw.get(alias)
            if val is not None and val != "" and val != []:
                result[canonical] = val
                break

    # --- Step 5: normalise tags ---
    tags = result.get("tags")
    if isinstance(tags, str):
        result["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
    elif isinstance(tags, list):
        normalised: list[str] = []
        for t in tags:
            if isinstance(t, str):
                # A single list item may itself be "tag1, tag2"
                normalised.extend(p.strip() for p in t.split(",") if p.strip())
            elif isinstance(t, dict):
                name = t.get("name") or t.get("tag") or t.get("label")
                if name:
                    normalised.append(str(name))
        result["tags"] = normalised

    logger.info("[GHOST] normalised metadata keys: %s", list(result.keys()))
    return result


def _ghost_jwt() -> str:
    """Generate a Ghost Admin API JWT token (HS256, 5-minute expiry).

    Ghost expects an API key in ``id:secret`` format where secret is hex-encoded.
    The JWT must carry a ``kid`` header set to the key ID.
    """
    key_id, secret = GHOST_ADMIN_API_KEY.split(":")
    iat = int(time.time())
    payload = {"iat": iat, "exp": iat + 300, "aud": "/admin/"}
    return jwt.encode(
        payload,
        bytes.fromhex(secret),
        algorithm="HS256",
        headers={"kid": key_id},
    )


def _upload_image_to_ghost(image_url: str, token: str) -> str:
    """Download an image from a URL and upload it to Ghost's media storage.

    Returns the Ghost-hosted URL.
    Raises httpx.HTTPStatusError on download or upload failure.
    """
    dl = httpx.get(image_url, follow_redirects=True, timeout=30)
    dl.raise_for_status()

    content_type = dl.headers.get("content-type", "image/jpeg").split(";")[0].strip()
    # Derive a clean filename from the URL path
    filename = image_url.rstrip("/").split("/")[-1].split("?")[0] or "image.jpg"

    upload_url = f"{GHOST_ADMIN_URL}/ghost/api/admin/images/upload/"
    headers = {
        "Authorization": f"Ghost {token}",
        "Accept-Version": "v5.0",
    }
    # ``ref`` is optional but useful for Ghost to preserve the original filename
    files = {
        "file": (filename, dl.content, content_type),
        "ref": (None, filename),
    }
    resp = httpx.post(upload_url, headers=headers, files=files, timeout=60)
    resp.raise_for_status()
    return resp.json()["images"][0]["url"]


def _rehost_images_in_html(html: str, token: str) -> str:
    """Upload every <img src="..."> in the HTML to Ghost and replace the URLs.

    Images that fail to upload are left with their original URL so the post
    still saves (Ghost will display them from the external URL as a fallback).
    """
    def _replace(match: re.Match) -> str:
        original_url = match.group(1)
        try:
            ghost_url = _upload_image_to_ghost(original_url, token)
            logger.debug("Ghost image rehosted: %s -> %s", original_url, ghost_url)
            return f'src="{ghost_url}"'
        except Exception as e:
            logger.warning("Ghost image upload failed for %s: %s", original_url, e)
            return match.group(0)  # keep original on failure

    return re.sub(r'src="([^"]+)"', _replace, html)


def _extract_hero_url_from_content(content: str) -> Optional[str]:
    """If the article starts with a hero image markdown line, extract its URL.

    ``inject_hero_into_markdown()`` prepends ``![alt](url)`` as the very first
    line.  This helper detects that pattern and returns the URL so it can be
    used as the Ghost ``feature_image`` even when the DB field is not set.
    """
    if not content:
        return None
    first_line = content.split("\n")[0].strip()
    m = re.match(r"!\[.*?\]\((.+?)\)", first_line)
    return m.group(1) if m else None


def _strip_hero_from_md(content: str, hero_url: Optional[str]) -> str:
    """Remove the hero image line that inject_hero_into_markdown() prepends.

    ``inject_hero_into_markdown()`` in src/images.py prepends::

        ![Hero image](url)

        {rest of article}

    For Ghost we pass the hero URL as ``feature_image`` instead, so we strip
    the first markdown image line at position 0.  We match loosely: if the
    first line is any ``![...](...)`` element we remove it, regardless of
    whether the URL exactly matches ``hero_url`` (the stored DB field can
    sometimes differ from the embedded URL due to URL normalisation).
    """
    lines = content.split("\n")
    if lines and re.match(r"!\[.*?\]\(.+?\)", lines[0].strip()):
        return "\n".join(lines[1:]).lstrip("\n")
    return content


def _md_to_html(md: str) -> str:
    """Convert markdown to HTML suitable for Ghost's ``html`` content field."""
    return markdown2.markdown(
        md,
        extras=["fenced-code-blocks", "tables", "strike", "header-ids"],
    )


def _strip_h1_from_html(html: str) -> str:
    """Remove the first <h1>…</h1> from HTML.

    Article markdown starts with the title as H1, but Ghost renders the post
    title separately — keeping H1 in the body would duplicate it.
    """
    return re.sub(r'<h1[^>]*>.*?</h1>\s*', '', html, count=1, flags=re.DOTALL)


def create_ghost_draft(
    title: str,
    content_md: str,
    hero_image_url: Optional[str],
    metadata: dict[str, Any],
) -> dict[str, str]:
    """Create a Ghost draft post via the Admin API.

    Images (hero + all inline images in the body) are uploaded to Ghost's own
    media storage before the post is created, so Ghost hosts them on its CDN.

    Args:
        title: Fallback title (used when metadata lacks one).
        content_md: Full article markdown (may include a prepended hero image).
        hero_image_url: URL of the hero image, used as Ghost ``feature_image``.
        metadata: SEO metadata dict (title, slug, meta_title, meta_description,
                  excerpt, tags, codeinjection_head).

    Returns:
        ``{"id": str, "url": str, "editor_url": str}``

    Raises:
        ValueError: If env vars are not configured.
        httpx.HTTPStatusError: If the Ghost API returns an error response.
    """
    if not GHOST_ADMIN_URL or not GHOST_ADMIN_API_KEY:
        raise ValueError(
            "GHOST_ADMIN_URL and GHOST_ADMIN_API_KEY environment variables must be set"
        )

    # Normalise metadata regardless of what the caller stored
    meta = _normalize_seo_metadata(metadata) if metadata else {}
    logger.info("[GHOST] create_ghost_draft: normalised metadata=%s", list(meta.keys()))

    # Generate token once — reused for all image uploads + the post creation call
    token = _ghost_jwt()

    # --- Hero image ---
    # Use the explicit hero_image_url first; fall back to extracting from content
    # (the hero is prepended as the first line by inject_hero_into_markdown).
    effective_hero_url = hero_image_url or _extract_hero_url_from_content(content_md)

    feature_image: Optional[str] = None
    if effective_hero_url:
        try:
            feature_image = _upload_image_to_ghost(effective_hero_url, token)
            logger.info("[GHOST] hero image uploaded to Ghost: %s", feature_image)
        except Exception as e:
            logger.warning("[GHOST] hero image upload failed, using original URL: %s", e)
            feature_image = effective_hero_url

    # --- Body content ---
    # Strip the prepended hero line from markdown (it goes to feature_image)
    body_md = _strip_hero_from_md(content_md, effective_hero_url)
    html_content = _md_to_html(body_md)
    html_content = _strip_h1_from_html(html_content)
    # Upload all inline images to Ghost and rewrite their src URLs
    html_content = _rehost_images_in_html(html_content, token)

    # --- Build post payload ---
    raw_tags = meta.get("tags", [])
    ghost_tags = [{"name": t} for t in raw_tags] if isinstance(raw_tags, list) else []

    # Ghost field length limits (422 if exceeded)
    _excerpt = meta.get("excerpt")
    if _excerpt and len(_excerpt) > 300:
        logger.warning("[GHOST] custom_excerpt too long (%d chars), truncating to 300", len(_excerpt))
        _excerpt = _excerpt[:297] + "..."
    _meta_title = meta.get("meta_title")
    if _meta_title and len(_meta_title) > 300:
        _meta_title = _meta_title[:297] + "..."
    _meta_desc = meta.get("meta_description")
    if _meta_desc and len(_meta_desc) > 500:
        _meta_desc = _meta_desc[:497] + "..."

    post: dict[str, Any] = {
        "title": meta.get("title") or title or "Untitled",
        "slug": meta.get("slug"),
        "html": html_content,
        "source_format": "html",  # Tell Ghost 5 the content is HTML, not lexical/mobiledoc
        "feature_image": feature_image,
        "tags": ghost_tags,
        "meta_title": _meta_title,
        "meta_description": _meta_desc,
        "custom_excerpt": _excerpt,
        "codeinjection_head": meta.get("codeinjection_head"),
        "status": "draft",
        "authors": [{"email": GHOST_AUTHOR_EMAIL}] if GHOST_AUTHOR_EMAIL else None,
    }
    # Drop None / empty-list values so Ghost applies its own defaults
    post = {k: v for k, v in post.items() if v is not None and v != []}

    endpoint = f"{GHOST_ADMIN_URL}/ghost/api/admin/posts/"
    headers = {
        "Authorization": f"Ghost {token}",
        "Content-Type": "application/json",
        "Accept-Version": "v5.0",
    }

    response = httpx.post(endpoint, json={"posts": [post]}, headers=headers, timeout=30)
    if response.status_code == 422:
        error_body = response.text[:1000]
        logger.error("[GHOST] 422 from Ghost API. Payload keys: %s. Response: %s", list(post.keys()), error_body)
        # Slug conflicts are the most common 422 cause — retry without slug
        if "slug" in post and ("Duplicate" in error_body or "slug" in error_body.lower() or "unique" in error_body.lower()):
            logger.info("[GHOST] Retrying without slug (likely duplicate)")
            post.pop("slug", None)
            response = httpx.post(endpoint, json={"posts": [post]}, headers=headers, timeout=30)
            if not response.is_success:
                logger.error("[GHOST] Retry also failed %s: %s", response.status_code, response.text[:500])
                response.raise_for_status()
        else:
            response.raise_for_status()
    elif not response.is_success:
        logger.error("[GHOST] %s from Ghost API: %s", response.status_code, response.text[:500])
        response.raise_for_status()

    created = response.json()["posts"][0]
    ghost_id = created["id"]
    # ``url`` in the response is the public URL (only set when published).
    # Fall back to the Ghost editor URL so the user can always open it.
    editor_url = f"{GHOST_ADMIN_URL}/ghost/#/editor/post/{ghost_id}"
    return {
        "id": ghost_id,
        "url": created.get("url") or editor_url,
        "editor_url": editor_url,
    }
