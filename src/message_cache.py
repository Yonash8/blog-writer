from __future__ import annotations
"""Local SQLite cache for recent conversation messages.

Acts as a fast read layer in front of Supabase. Writes go to both.
On cold start (empty cache for a user), backfills from Supabase.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "messages.db"
_conn: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                channel TEXT NOT NULL,
                channel_user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        _conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_msg_channel_user ON messages(channel, channel_user_id, created_at)"
        )
        # PromptLayer execution ID cache (last 20, auto-pruned)
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS pl_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exec_id TEXT NOT NULL,
                workflow TEXT NOT NULL,
                article_id TEXT,
                channel TEXT,
                channel_user_id TEXT,
                created_at TEXT NOT NULL
            )
        """)
        _conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pl_exec_channel ON pl_executions(channel, channel_user_id, created_at)"
        )
        _conn.commit()
    return _conn


def add(id: str, channel: str, channel_user_id: str, role: str, content: str, created_at: str) -> None:
    """Insert a message into the local cache (ignore if already exists)."""
    conn = _get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO messages (id, channel, channel_user_id, role, content, created_at) VALUES (?,?,?,?,?,?)",
        (id, channel, channel_user_id, role, content, created_at),
    )
    conn.commit()


def get_recent(channel: str, channel_user_id: str, limit: int = 10) -> list[dict]:
    """Return the most recent `limit` messages in chronological order."""
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT id, channel, channel_user_id, role, content, created_at
        FROM messages
        WHERE channel = ? AND channel_user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (channel, channel_user_id, limit),
    ).fetchall()
    # Reverse so oldest-first (chronological for Claude)
    return [dict(r) for r in reversed(rows)]


def get_before(channel: str, channel_user_id: str, before_timestamp: str, limit: int = 20) -> list[dict]:
    """Return up to `limit` messages older than `before_timestamp`, chronological order."""
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT id, channel, channel_user_id, role, content, created_at
        FROM messages
        WHERE channel = ? AND channel_user_id = ? AND created_at < ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (channel, channel_user_id, before_timestamp, limit),
    ).fetchall()
    return [dict(r) for r in reversed(rows)]


def count(channel: str, channel_user_id: str) -> int:
    """Return total cached messages for a user."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE channel = ? AND channel_user_id = ?",
        (channel, channel_user_id),
    ).fetchone()
    return row[0] if row else 0


def backfill_from_supabase(channel: str, channel_user_id: str, limit: int = 50) -> int:
    """Pull recent messages from Supabase into local cache. Returns count inserted."""
    try:
        from src.db import get_messages
        rows = get_messages(channel, channel_user_id, limit=limit)
        inserted = 0
        for m in rows:
            mid = str(m.get("id", ""))
            ts = str(m.get("created_at", ""))
            if mid and ts:
                add(mid, channel, channel_user_id, m["role"], m["content"], ts)
                inserted += 1
        logger.debug("[CACHE] Backfilled %d messages for %s/%s", inserted, channel, channel_user_id)
        return inserted
    except Exception as e:
        logger.warning("[CACHE] Backfill failed: %s", e)
        return 0


def add_pl_execution(
    exec_id: str,
    workflow: str,
    article_id: Optional[str] = None,
    channel: Optional[str] = None,
    channel_user_id: Optional[str] = None,
    keep_last: int = 20,
) -> None:
    """Record a PromptLayer execution ID. Auto-prunes to keep_last rows per channel/user."""
    from datetime import datetime, timezone
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO pl_executions (exec_id, workflow, article_id, channel, channel_user_id, created_at) VALUES (?,?,?,?,?,?)",
        (exec_id, workflow, article_id, channel, channel_user_id, now),
    )
    # Prune: keep only the last `keep_last` rows for this channel/user combo
    conn.execute(
        """
        DELETE FROM pl_executions
        WHERE id NOT IN (
            SELECT id FROM pl_executions
            WHERE channel IS ? AND channel_user_id IS ?
            ORDER BY created_at DESC
            LIMIT ?
        ) AND channel IS ? AND channel_user_id IS ?
        """,
        (channel, channel_user_id, keep_last, channel, channel_user_id),
    )
    conn.commit()


def get_recent_pl_executions(
    channel: Optional[str] = None,
    channel_user_id: Optional[str] = None,
    limit: int = 5,
) -> list[dict]:
    """Return the most recent PromptLayer execution records, newest first."""
    conn = _get_conn()
    if channel and channel_user_id:
        rows = conn.execute(
            "SELECT * FROM pl_executions WHERE channel=? AND channel_user_id=? ORDER BY created_at DESC LIMIT ?",
            (channel, channel_user_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM pl_executions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_recent_with_backfill(channel: str, channel_user_id: str, limit: int = 10) -> list[dict]:
    """Get recent messages, backfilling from Supabase if cache is empty."""
    if count(channel, channel_user_id) == 0:
        backfill_from_supabase(channel, channel_user_id, limit=max(50, limit))
    return get_recent(channel, channel_user_id, limit=limit)
