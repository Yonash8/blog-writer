"""Supabase client and database helpers."""

import os
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

_url = os.getenv("SUPABASE_URL")
# Prefer service_role key for backend: bypasses RLS, full DB + storage access
_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

_client: Optional[Client] = None


def get_client() -> Client:
    """Get or create Supabase client."""
    global _client
    if _client is None:
        if not _url or not _key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_KEY) must be set. "
                "For backend writes, use the service_role key from Dashboard -> API."
            )
        _client = create_client(_url, _key)
    return _client


def _to_dict(row: dict) -> dict:
    """Convert Supabase row to plain dict (handle UUIDs)."""
    if not row:
        return {}
    d = dict(row)
    for k, v in d.items():
        if isinstance(v, UUID):
            d[k] = str(v)
    return d


# --- General-purpose SQL execution ---

def execute_sql(query: str) -> list[dict]:
    """Execute a read-only SQL query via the execute_readonly_sql RPC function.

    Only SELECT and WITH (CTE) statements are allowed.
    Returns a list of row dicts.
    """
    client = get_client()
    r = client.rpc("execute_readonly_sql", {"query": query}).execute()
    rows = r.data if r.data else []
    # RPC returns JSON; if it's a list, return as-is; if wrapped, unwrap
    if isinstance(rows, list):
        return rows
    # Some Supabase versions wrap RPC results
    return [rows] if rows else []


# --- Messages ---

def add_message(channel: str, channel_user_id: str, role: str, content: str) -> dict:
    """Add a message for a user."""
    client = get_client()
    r = client.table("messages").insert({
        "channel": channel,
        "channel_user_id": channel_user_id,
        "role": role,
        "content": content,
    }).execute()
    if not r.data or len(r.data) == 0:
        raise RuntimeError("Failed to add message")
    return _to_dict(r.data[0])


def get_messages(channel: str, channel_user_id: str, limit: int = 100) -> list[dict]:
    """Get messages for a user, ordered by created_at."""
    client = get_client()
    r = (
        client.table("messages")
        .select("*")
        .eq("channel", channel)
        .eq("channel_user_id", channel_user_id)
        .order("created_at")
        .limit(limit)
        .execute()
    )
    return [_to_dict(m) for m in (r.data or [])]


# --- Articles ---

def append_article_changelog(article_id: str, action: str) -> None:
    """Append a changelog entry to an article."""
    client = get_client()
    r = client.table("articles").select("changelog").eq("id", article_id).execute()
    if not r.data or len(r.data) == 0:
        return
    current = r.data[0].get("changelog") or []
    if not isinstance(current, list):
        current = []
    entry = {"action": action, "at": datetime.now(timezone.utc).isoformat()}
    updated = current + [entry]
    client.table("articles").update({"changelog": updated}).eq("id", article_id).execute()


def create_article(
    channel: str,
    channel_user_id: str,
    topic_id: str,
    content: str,
    sources: Optional[list] = None,
    version: int = 1,
    title: Optional[str] = None,
    status: str = "draft",
    changelog_entry: Optional[str] = None,
) -> dict:
    """Create a new article. Always saves to DB. Requires topic_id. Optional changelog_entry from agent."""
    client = get_client()
    changelog = []
    if changelog_entry and changelog_entry.strip():
        changelog = [{"action": changelog_entry.strip(), "at": datetime.now(timezone.utc).isoformat()}]
    data = {
        "channel": channel,
        "channel_user_id": channel_user_id,
        "topic_id": topic_id,
        "content": content,
        "version": version,
        "status": status,
        "changelog": changelog,
    }
    if sources is not None:
        data["sources"] = sources
    if title:
        data["title"] = title
    r = client.table("articles").insert(data).execute()
    if not r.data or len(r.data) == 0:
        raise RuntimeError("Failed to create article")
    return _to_dict(r.data[0])


def set_article_status(article_id: str, status: str, changelog_entry: Optional[str] = None) -> dict:
    """Set article status: draft, posted, in_progress. Optional changelog_entry (e.g. from agent)."""
    if status not in ("draft", "posted", "in_progress"):
        raise ValueError(f"Invalid status: {status}")
    client = get_client()
    r = client.table("articles").update({"status": status}).eq("id", article_id).execute()
    if not r.data or len(r.data) == 0:
        raise RuntimeError("Failed to update article status")
    if changelog_entry and changelog_entry.strip():
        append_article_changelog(article_id, changelog_entry.strip())
    return _to_dict(r.data[0])


def set_article_google_doc_url(article_id: str, google_doc_url: str) -> dict:
    """Set the Google Doc URL for an article."""
    client = get_client()
    r = client.table("articles").update({"google_doc_url": google_doc_url}).eq("id", article_id).execute()
    if not r.data or len(r.data) == 0:
        raise RuntimeError("Failed to update article google_doc_url")
    return _to_dict(r.data[0])


def update_article(
    article_id: str,
    content: str,
    sources: Optional[list] = None,
    google_doc_url: Optional[str] = None,
    status: Optional[str] = None,
    changelog_action: Optional[str] = None,
) -> dict:
    """Update article content (creates new version or overwrites). Optionally append to changelog."""
    client = get_client()
    r = client.table("articles").select("version").eq("id", article_id).execute()
    if not r.data or len(r.data) == 0:
        raise ValueError(f"Article {article_id} not found")
    current = r.data[0]
    new_version = (current.get("version") or 1) + 1
    data = {"content": content, "version": new_version}
    if sources is not None:
        data["sources"] = sources
    if google_doc_url is not None:
        data["google_doc_url"] = google_doc_url
    if status is not None:
        if status not in ("draft", "posted", "in_progress"):
            raise ValueError(f"Invalid status: {status}")
        data["status"] = status
    r = client.table("articles").update(data).eq("id", article_id).execute()
    if changelog_action:
        append_article_changelog(article_id, changelog_action)
    if not r.data or len(r.data) == 0:
        raise RuntimeError("Failed to update article")
    return _to_dict(r.data[0])


def _is_full_uuid(s: str) -> bool:
    """Check if a string is a full UUID (8-4-4-4-12 hex with dashes)."""
    import re
    return bool(re.match(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$', s))


def get_article(article_id: str) -> Optional[dict]:
    """Get article by full UUID. Returns None for invalid/partial UUIDs."""
    if not article_id or not _is_full_uuid(article_id.strip()):
        return None
    client = get_client()
    try:
        r = client.table("articles").select("*").eq("id", article_id.strip()).execute()
    except Exception:
        return None
    if not r.data or len(r.data) == 0:
        return None
    return _to_dict(r.data[0])


def get_article_by_id_or_prefix(
    article_id: str,
    channel: Optional[str] = None,
    channel_user_id: Optional[str] = None,
) -> Optional[dict]:
    """Get article by exact ID or by UUID prefix (e.g. 'e799b28b').
    When channel/channel_user_id are set, only returns articles belonging to that user.
    """
    if not article_id or not article_id.strip():
        return None
    tid = article_id.strip()

    # Only try exact match if it looks like a full UUID (Postgres rejects partial UUIDs)
    if _is_full_uuid(tid):
        article = get_article(tid)
        if article:
            if channel and article.get("channel") != channel:
                return None
            if channel_user_id and article.get("channel_user_id") != channel_user_id:
                return None
            return article

    # Prefix match (e.g. e799b28b) - fetch user's articles and filter by ID prefix
    if len(tid) >= 4 and all(c in "0123456789abcdefABCDEF-" for c in tid):
        arts = list_articles(
            channel=channel,
            channel_user_id=channel_user_id,
            limit=100,
        )
        for a in arts:
            if a.get("id", "").lower().startswith(tid.lower()):
                return a
    return None


def list_articles(
    channel: Optional[str] = None,
    channel_user_id: Optional[str] = None,
    title_query: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """List saved articles, optionally filtered by channel, user, and title search."""
    client = get_client()
    q = client.table("articles").select(
        "id, channel, channel_user_id, topic_id, title, version, status, changelog, google_doc_url, created_at"
    ).order("created_at", desc=True)
    if channel:
        q = q.eq("channel", channel)
    if channel_user_id:
        q = q.eq("channel_user_id", channel_user_id)
    if title_query and title_query.strip():
        q = q.ilike("title", f"%{title_query.strip()}%")
    r = q.range(offset, offset + limit - 1).execute()
    return [_to_dict(a) for a in (r.data or [])]


def get_latest_article_for_user(channel: str, channel_user_id: str) -> Optional[dict]:
    """Get the most recently created article for a user."""
    client = get_client()
    r = (
        client.table("articles")
        .select("*")
        .eq("channel", channel)
        .eq("channel_user_id", channel_user_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if not r.data or len(r.data) == 0:
        return None
    return _to_dict(r.data[0])


def list_articles_for_user(
    channel: str,
    channel_user_id: str,
    title_query: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """List articles for a user. Optionally filter by title (partial match)."""
    return list_articles(
        channel=channel,
        channel_user_id=channel_user_id,
        title_query=title_query,
        limit=limit,
        offset=offset,
    )


def create_article_for_user(
    channel: str,
    channel_user_id: str,
    topic_id: str,
    content: str,
    sources: Optional[list] = None,
    version: int = 1,
    title: Optional[str] = None,
    status: str = "draft",
    changelog_entry: Optional[str] = None,
) -> dict:
    """Create an article for a user. Alias for create_article with same params."""
    return create_article(
        channel=channel,
        channel_user_id=channel_user_id,
        topic_id=topic_id,
        content=content,
        sources=sources,
        version=version,
        title=title,
        status=status,
        changelog_entry=changelog_entry,
    )


def get_messages_for_user(channel: str, channel_user_id: str, limit: int = 100) -> list[dict]:
    """Get messages for a user. Alias for get_messages."""
    return get_messages(channel, channel_user_id, limit=limit)


def add_message_for_user(channel: str, channel_user_id: str, role: str, content: str) -> dict:
    """Add a message for a user. Alias for add_message."""
    return add_message(channel, channel_user_id, role, content)


# --- Article Images ---

def add_article_image(
    article_id: str,
    url: str,
    position: int = 0,
    alt_text: Optional[str] = None,
    prompt_used: Optional[str] = None,
    status: str = "approved",
    image_type: str = "generic",
    metadata: Optional[dict] = None,
) -> dict:
    """Add an article image record.

    status: 'pending_approval', 'approved', or 'rejected'
    image_type: 'generic', 'hero', or 'infographic'
    metadata: Extra info (e.g. infographic analysis, position_after snippet)
    """
    client = get_client()
    data = {
        "article_id": article_id,
        "position": position,
        "url": url,
        "alt_text": alt_text,
        "prompt_used": prompt_used,
        "status": status,
        "image_type": image_type,
        "metadata": metadata or {},
    }
    r = client.table("article_images").insert(data).execute()
    if not r.data or len(r.data) == 0:
        raise RuntimeError("Failed to add article image")
    return _to_dict(r.data[0])


def update_article_image_status(image_id: str, status: str) -> dict:
    """Update the status of an article image (pending_approval -> approved/rejected)."""
    if status not in ("pending_approval", "approved", "rejected"):
        raise ValueError(f"Invalid image status: {status}")
    client = get_client()
    r = client.table("article_images").update({"status": status}).eq("id", image_id).execute()
    if not r.data or len(r.data) == 0:
        raise RuntimeError(f"Failed to update article image {image_id}")
    return _to_dict(r.data[0])


def get_article_image(image_id: str) -> Optional[dict]:
    """Get a single article image by ID."""
    client = get_client()
    r = client.table("article_images").select("*").eq("id", image_id).execute()
    if not r.data or len(r.data) == 0:
        return None
    return _to_dict(r.data[0])


def get_pending_article_images(article_id: str, image_type: Optional[str] = None) -> list[dict]:
    """Get pending-approval images for an article, optionally filtered by type."""
    client = get_client()
    q = (
        client.table("article_images")
        .select("*")
        .eq("article_id", article_id)
        .eq("status", "pending_approval")
    )
    if image_type:
        q = q.eq("image_type", image_type)
    r = q.order("created_at", desc=True).execute()
    return [_to_dict(i) for i in (r.data or [])]


def get_article_images(article_id: str) -> list[dict]:
    """Get images for an article."""
    client = get_client()
    r = client.table("article_images").select("*").eq(
        "article_id", article_id
    ).order("position").execute()
    return [_to_dict(i) for i in (r.data or [])]


# --- Topics ---

def get_topics(query: Optional[str] = None, limit: int = 20) -> list[dict]:
    """Retrieve topics, optionally filtered by search query."""
    client = get_client()
    q = client.table("topics").select("id, title, description, keywords, \"group\", created_at")
    if query and query.strip():
        q = q.ilike("title", f"%{query.strip()}%")
    r = q.order("created_at", desc=True).limit(limit).execute()
    return [_to_dict(t) for t in (r.data or [])]


def get_topic_by_title(title: str) -> Optional[dict]:
    """Find topic by exact title match (case-insensitive)."""
    client = get_client()
    r = client.table("topics").select("*").ilike("title", title.strip()).limit(1).execute()
    if not r.data or len(r.data) == 0:
        return None
    return _to_dict(r.data[0])


def get_or_create_topic_for_article(title: str) -> dict:
    """Get topic by title, or create one. Returns topic dict with id."""
    existing = get_topic_by_title(title)
    if existing:
        return existing
    return create_topic(title=title.strip())


def create_topic(title: str, description: Optional[str] = None, keywords: Optional[list] = None, group: Optional[str] = None) -> dict:
    """Create a topic."""
    client = get_client()
    data = {"title": title}
    if description:
        data["description"] = description
    if keywords is not None:
        data["keywords"] = keywords
    if group is not None:
        data["group"] = group
    r = client.table("topics").insert(data).execute()
    if not r.data or len(r.data) == 0:
        raise RuntimeError("Failed to create topic")
    return _to_dict(r.data[0])


# --- Observability traces ---

def list_traces(
    channel: Optional[str] = None,
    channel_user_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """List observability traces, optionally filtered by channel and channel_user_id."""
    client = get_client()
    q = client.table("observability_traces").select(
        "trace_id, channel, channel_user_id, user_message, final_message, payload, created_at"
    )
    if channel:
        q = q.eq("channel", channel)
    if channel_user_id:
        q = q.eq("channel_user_id", channel_user_id)
    r = q.order("created_at", desc=True).range(offset, offset + limit - 1).execute()
    return [_to_dict(t) for t in (r.data or [])]


def get_trace(trace_id: str) -> Optional[dict]:
    """Get a single trace by trace_id including full payload."""
    client = get_client()
    r = client.table("observability_traces").select("*").eq("trace_id", trace_id).limit(1).execute()
    if not r.data or len(r.data) == 0:
        return None
    return _to_dict(r.data[0])


# --- Agent config (admin) ---

def upsert_agent_config(key: str, value: str) -> None:
    """Upsert a single agent_config row."""
    client = get_client()
    client.table("agent_config").upsert(
        {"key": key, "value": value},
        on_conflict="key",
    ).execute()


def upsert_prompt(key: str, content: str) -> None:
    """Upsert a single prompts row."""
    client = get_client()
    client.table("prompts").upsert(
        {"key": key, "content": content},
        on_conflict="key",
    ).execute()
