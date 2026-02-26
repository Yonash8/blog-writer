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


def _rehost_images_in_md(md: str, token: str) -> str:
    """Upload every ![alt](url) image in the markdown to Ghost and replace URLs.

    Images that fail to upload are left with their original URL.
    """
    def _replace(match: re.Match) -> str:
        alt, original_url = match.group(1), match.group(2)
        try:
            ghost_url = _upload_image_to_ghost(original_url, token)
            logger.debug("Ghost image rehosted: %s -> %s", original_url, ghost_url)
            return f"![{alt}]({ghost_url})"
        except Exception as e:
            logger.warning("Ghost image upload failed for %s: %s", original_url, e)
            return match.group(0)

    return re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', _replace, md)


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
    """Convert markdown to HTML suitable for Ghost's content field."""
    return markdown2.markdown(
        md,
        extras=["fenced-code-blocks", "tables", "strike", "header-ids"],
    )


_FMT_BOLD = 1
_FMT_ITALIC = 1 << 1
_FMT_STRIKETHROUGH = 1 << 2
_FMT_UNDERLINE = 1 << 3
_FMT_CODE = 1 << 4


def _lexical_root(children: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "root": {
            "children": children,
            "direction": None,
            "format": "",
            "indent": 0,
            "type": "root",
            "version": 1,
        }
    }


def _lexical_text(text: str, fmt: int = 0) -> dict[str, Any]:
    return {
        "detail": 0,
        "format": fmt,
        "mode": "normal",
        "style": "",
        "text": text,
        "type": "text",
        "version": 1,
    }


def _lexical_linebreak() -> dict[str, Any]:
    return {"type": "linebreak", "version": 1}


def _lexical_paragraph(children: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "children": children or [_lexical_text("")],
        "direction": None,
        "format": "",
        "indent": 0,
        "type": "paragraph",
        "version": 1,
    }


def _lexical_heading(tag: str, children: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "children": children or [_lexical_text("")],
        "direction": None,
        "format": "",
        "indent": 0,
        "type": "heading",
        "tag": tag,
        "version": 1,
    }


def _lexical_quote(children: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "children": children or [_lexical_paragraph([_lexical_text("")])],
        "direction": None,
        "format": "",
        "indent": 0,
        "type": "quote",
        "version": 1,
    }


def _lexical_list(list_type: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "children": items,
        "direction": None,
        "format": "",
        "indent": 0,
        "listType": list_type,
        "start": 1,
        "tag": "ol" if list_type == "number" else "ul",
        "type": "list",
        "version": 1,
    }


def _lexical_list_item(children: list[dict[str, Any]], value: int) -> dict[str, Any]:
    return {
        "children": children or [_lexical_paragraph([_lexical_text("")])],
        "direction": None,
        "format": "",
        "indent": 0,
        "type": "listitem",
        "value": value,
        "version": 1,
    }


def _lexical_link(url: str, children: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "children": children or [_lexical_text(url)],
        "direction": None,
        "format": "",
        "indent": 0,
        "rel": "noreferrer",
        "target": "_blank",
        "title": None,
        "type": "link",
        "url": url,
        "version": 1,
    }


def _lexical_table_cell(text: str, is_header: bool) -> dict[str, Any]:
    return {
        "children": [_lexical_paragraph(_inline_md_to_lexical_nodes(text))],
        "colSpan": 1,
        "rowSpan": 1,
        "headerState": 2 if is_header else 0,
        "type": "tablecell",
        "version": 1,
    }


def _lexical_table_row(values: list[str], is_header: bool) -> dict[str, Any]:
    return {
        "children": [_lexical_table_cell(v, is_header=is_header) for v in values],
        "type": "tablerow",
        "version": 1,
    }


def _lexical_table(headers: list[str], rows: list[list[str]]) -> dict[str, Any]:
    return {
        "children": [
            _lexical_table_row(headers, is_header=True),
            *[_lexical_table_row(r, is_header=False) for r in rows],
        ],
        "type": "table",
        "version": 1,
    }


_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*|__([^_]+)__")
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)|(?<!_)_([^_\n]+)_(?!_)")
_CODE_RE = re.compile(r"`([^`\n]+)`")


def _inline_md_to_lexical_nodes(text: str, fmt: int = 0) -> list[dict[str, Any]]:
    """Parse basic inline markdown into Lexical text/link nodes."""
    nodes: list[dict[str, Any]] = []
    i = 0
    while i < len(text):
        candidates: list[tuple[int, str, re.Match[str]]] = []
        for kind, pat in (("link", _LINK_RE), ("bold", _BOLD_RE), ("italic", _ITALIC_RE), ("code", _CODE_RE)):
            m = pat.search(text, i)
            if m:
                candidates.append((m.start(), kind, m))
        if not candidates:
            tail = text[i:]
            if tail:
                nodes.append(_lexical_text(tail, fmt))
            break
        candidates.sort(key=lambda x: x[0])
        start, kind, m = candidates[0]
        if start > i:
            nodes.append(_lexical_text(text[i:start], fmt))
        if kind == "link":
            label = m.group(1)
            url = m.group(2).strip()
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            nodes.append(_lexical_link(url, _inline_md_to_lexical_nodes(label, fmt)))
        elif kind == "bold":
            inner = m.group(1) or m.group(2) or ""
            nodes.extend(_inline_md_to_lexical_nodes(inner, fmt | _FMT_BOLD))
        elif kind == "italic":
            inner = m.group(1) or m.group(2) or ""
            nodes.extend(_inline_md_to_lexical_nodes(inner, fmt | _FMT_ITALIC))
        else:  # code
            inner = m.group(1) or ""
            nodes.append(_lexical_text(inner, fmt | _FMT_CODE))
        i = m.end()
    return nodes


_IMAGE_LINE_RE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_UL_ITEM_RE = re.compile(r"^\s*[-*+]\s+(.+)$")
_OL_ITEM_RE = re.compile(r"^\s*\d+\.\s+(.+)$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")


def _split_table_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _looks_like_table_start(lines: list[str], i: int) -> bool:
    if i + 1 >= len(lines):
        return False
    return ("|" in lines[i]) and bool(_TABLE_SEPARATOR_RE.match(lines[i + 1].strip()))


def _parse_md_table(lines: list[str], i: int) -> tuple[dict[str, Any], int]:
    header_cells = _split_table_row(lines[i])
    if not header_cells or all(c == "" for c in header_cells):
        raise ValueError("Malformed markdown table: missing header row")
    col_count = len(header_cells)
    j = i + 1
    if not _TABLE_SEPARATOR_RE.match(lines[j].strip()):
        raise ValueError("Malformed markdown table: missing separator row")
    j += 1
    body_rows: list[list[str]] = []
    while j < len(lines):
        line = lines[j].rstrip()
        if not line.strip():
            break
        if "|" not in line:
            break
        row = _split_table_row(line)
        if len(row) != col_count:
            raise ValueError(
                f"Malformed markdown table: row has {len(row)} columns, expected {col_count}"
            )
        body_rows.append(row)
        j += 1
    return {"type": "table", "headers": header_cells, "rows": body_rows}, j


def _parse_markdown_blocks(md: str) -> list[dict[str, Any]]:
    lines = md.splitlines()
    blocks: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        if stripped.startswith("```"):
            fence = stripped[:3]
            i += 1
            code_lines: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith(fence):
                code_lines.append(lines[i])
                i += 1
            if i >= len(lines):
                raise ValueError("Unclosed fenced code block in markdown content")
            i += 1
            blocks.append({"type": "code", "text": "\n".join(code_lines)})
            continue

        if _looks_like_table_start(lines, i):
            table_block, i = _parse_md_table(lines, i)
            blocks.append(table_block)
            continue

        m_image = _IMAGE_LINE_RE.match(stripped)
        if m_image:
            blocks.append({"type": "image", "alt": m_image.group(1), "url": m_image.group(2)})
            i += 1
            continue

        m_heading = _HEADING_RE.match(stripped)
        if m_heading:
            level = min(len(m_heading.group(1)), 6)
            blocks.append({"type": "heading", "level": level, "text": m_heading.group(2).strip()})
            i += 1
            continue

        if stripped.startswith(">"):
            q_lines: list[str] = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                q_lines.append(lines[i].strip()[1:].lstrip())
                i += 1
            blocks.append({"type": "quote", "text": "\n".join(q_lines).strip()})
            continue

        m_ul = _UL_ITEM_RE.match(line)
        m_ol = _OL_ITEM_RE.match(line)
        if m_ul or m_ol:
            list_type = "bullet" if m_ul else "number"
            items: list[str] = []
            while i < len(lines):
                cur = lines[i]
                mm = _UL_ITEM_RE.match(cur) if list_type == "bullet" else _OL_ITEM_RE.match(cur)
                if not mm:
                    break
                items.append(mm.group(1).strip())
                i += 1
            blocks.append({"type": "list", "list_type": list_type, "items": items})
            continue

        para_lines: list[str] = []
        while i < len(lines):
            cur = lines[i]
            cur_s = cur.strip()
            if not cur_s:
                break
            if (
                cur_s.startswith("```")
                or _looks_like_table_start(lines, i)
                or _IMAGE_LINE_RE.match(cur_s)
                or _HEADING_RE.match(cur_s)
                or cur_s.startswith(">")
                or _UL_ITEM_RE.match(cur)
                or _OL_ITEM_RE.match(cur)
            ):
                break
            para_lines.append(cur_s)
            i += 1
        blocks.append({"type": "paragraph", "text": " ".join(para_lines).strip()})
    return blocks


def _md_to_lexical(md: str) -> str:
    """Convert markdown to native Ghost Lexical blocks (no markdown-card box)."""
    blocks = _parse_markdown_blocks(md)
    children: list[dict[str, Any]] = []
    for block in blocks:
        t = block["type"]
        if t == "paragraph":
            children.append(_lexical_paragraph(_inline_md_to_lexical_nodes(block["text"])))
        elif t == "heading":
            tag = f"h{block['level']}"
            children.append(_lexical_heading(tag, _inline_md_to_lexical_nodes(block["text"])))
        elif t == "quote":
            children.append(_lexical_quote([_lexical_paragraph(_inline_md_to_lexical_nodes(block["text"]))]))
        elif t == "code":
            code_text = block["text"]
            code_children: list[dict[str, Any]] = []
            for idx, line in enumerate(code_text.split("\n")):
                if idx > 0:
                    code_children.append(_lexical_linebreak())
                code_children.append(_lexical_text(line, _FMT_CODE))
            children.append(_lexical_paragraph(code_children))
        elif t == "list":
            list_items: list[dict[str, Any]] = []
            for idx, item in enumerate(block["items"], 1):
                list_items.append(_lexical_list_item([_lexical_paragraph(_inline_md_to_lexical_nodes(item))], value=idx))
            children.append(_lexical_list(block["list_type"], list_items))
        elif t == "table":
            children.append(_lexical_table(block["headers"], block["rows"]))
        elif t == "image":
            # Use an HTML card node for image rendering while keeping main content native Lexical.
            alt = block["alt"].replace('"', "&quot;")
            src = block["url"].replace('"', "&quot;")
            children.append(
                {
                    "type": "html",
                    "version": 1,
                    "html": f'<img src="{src}" alt="{alt}" />',
                }
            )

    return json.dumps(_lexical_root(children))


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
    Hero image upload is required; this call fails if no hero exists or upload fails.

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
    if not effective_hero_url:
        raise ValueError(
            "No hero image found. Approve or set a hero image before pushing to Ghost."
        )

    try:
        feature_image = _upload_image_to_ghost(effective_hero_url, token)
        logger.info("[GHOST] hero image uploaded to Ghost: %s", feature_image)
    except Exception as e:
        raise RuntimeError(f"Failed to upload hero image to Ghost: {e}") from e

    # --- Body content ---
    # Strip the prepended hero line from markdown (it goes to feature_image)
    body_md = _strip_hero_from_md(content_md, effective_hero_url)
    # Strip the H1 title line (Ghost renders the title separately)
    body_md = re.sub(r'^#\s+.*\n*', '', body_md, count=1)
    # Upload inline images to Ghost and rewrite their URLs in the markdown
    body_md = _rehost_images_in_md(body_md, token)
    # Convert to native Lexical JSON blocks for Ghost 6+
    lexical_content = _md_to_lexical(body_md)
    logger.info("[GHOST] lexical content length: %d chars", len(lexical_content))

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
        "lexical": lexical_content,  # Ghost 6+ uses Lexical format
        "feature_image": feature_image,
        "tags": ghost_tags,
        "meta_title": _meta_title,
        "meta_description": _meta_desc,
        "canonical_url": meta.get("canonical_url"),
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
