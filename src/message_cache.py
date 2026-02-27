from __future__ import annotations
"""Local SQLite cache for recent conversation messages.

Acts as a fast read layer in front of Supabase. Writes go to both.
On cold start (empty cache for a user), backfills from Supabase.
"""

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "messages.db"
_conn_lock = threading.RLock()
_schema_initialized = False


def _is_corruption_error(err: Exception) -> bool:
    msg = str(err).lower()
    return (
        "database disk image is malformed" in msg
        or "malformed" in msg
        or "file is not a database" in msg
    )


def _open_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), timeout=5.0, check_same_thread=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _rebuild_cache_db(reason: str) -> None:
    """Quarantine corrupted SQLite files and recreate fresh cache DB."""
    global _schema_initialized
    _schema_initialized = False
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(_DB_PATH) + suffix)
        if not p.exists():
            continue
        backup = p.with_name(f"{p.name}.corrupt.{ts}")
        try:
            p.rename(backup)
        except Exception:
            # If rename fails (e.g. lock race), best effort remove.
            try:
                p.unlink()
            except Exception:
                pass
    logger.warning("[CACHE] Rebuilt local cache DB after corruption: %s", reason)


def _with_recovery(op):
    """Run a SQLite operation and auto-recover once on corruption."""
    with _conn_lock:
        _ensure_schema()
        conn = _open_conn()
        try:
            return op(conn)
        except sqlite3.DatabaseError as e:
            if not _is_corruption_error(e):
                raise
            try:
                conn.close()
            except Exception:
                pass
            _rebuild_cache_db(str(e))
            _ensure_schema()
            conn = _open_conn()
            return op(conn)
        finally:
            try:
                conn.close()
            except Exception:
                pass


def _ensure_schema() -> None:
    global _schema_initialized
    if _schema_initialized:
        return
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = _open_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                channel TEXT NOT NULL,
                channel_user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_msg_channel_user ON messages(channel, channel_user_id, created_at)"
        )
        conn.execute("""
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
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pl_exec_channel ON pl_executions(channel, channel_user_id, created_at)"
        )
        conn.execute("""
            CREATE TABLE IF NOT EXISTS current_missions (
                channel TEXT NOT NULL,
                channel_user_id TEXT NOT NULL,
                mission TEXT NOT NULL,
                article_id TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                updated_at TEXT NOT NULL,
                PRIMARY KEY (channel, channel_user_id)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_missions_status ON current_missions(status, updated_at)"
        )
        conn.commit()
        _schema_initialized = True
    finally:
        conn.close()


def add(id: str, channel: str, channel_user_id: str, role: str, content: str, created_at: str) -> None:
    """Insert a message into the local cache (ignore if already exists)."""
    def _op(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT OR IGNORE INTO messages (id, channel, channel_user_id, role, content, created_at) VALUES (?,?,?,?,?,?)",
            (id, channel, channel_user_id, role, content, created_at),
        )
        conn.commit()
    _with_recovery(_op)


def get_recent(channel: str, channel_user_id: str, limit: int = 10) -> list[dict]:
    """Return the most recent `limit` messages in chronological order."""
    def _op(conn: sqlite3.Connection) -> list[dict]:
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
        return [dict(r) for r in reversed(rows)]
    return _with_recovery(_op)


def get_before(channel: str, channel_user_id: str, before_timestamp: str, limit: int = 20) -> list[dict]:
    """Return up to `limit` messages older than `before_timestamp`, chronological order."""
    def _op(conn: sqlite3.Connection) -> list[dict]:
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
    return _with_recovery(_op)


def count(channel: str, channel_user_id: str) -> int:
    """Return total cached messages for a user."""
    def _op(conn: sqlite3.Connection) -> int:
        row = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE channel = ? AND channel_user_id = ?",
            (channel, channel_user_id),
        ).fetchone()
        return row[0] if row else 0
    return _with_recovery(_op)


def clear_for_user(channel: str, channel_user_id: str) -> int:
    """Delete all cached messages for this channel/user. Returns count deleted."""
    def _op(conn: sqlite3.Connection) -> int:
        cur = conn.execute(
            "DELETE FROM messages WHERE channel = ? AND channel_user_id = ?",
            (channel, channel_user_id),
        )
        deleted = cur.rowcount
        conn.commit()
        return deleted
    deleted = _with_recovery(_op)
    if deleted > 0:
        logger.info("[CACHE] Cleared %d messages for %s/%s", deleted, channel, channel_user_id[:20])
    return deleted


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
    def _op(conn: sqlite3.Connection) -> None:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO pl_executions (exec_id, workflow, article_id, channel, channel_user_id, created_at) VALUES (?,?,?,?,?,?)",
            (exec_id, workflow, article_id, channel, channel_user_id, now),
        )
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
    _with_recovery(_op)


def get_recent_pl_executions(
    channel: Optional[str] = None,
    channel_user_id: Optional[str] = None,
    limit: int = 5,
) -> list[dict]:
    """Return the most recent PromptLayer execution records, newest first."""
    def _op(conn: sqlite3.Connection) -> list[dict]:
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
    return _with_recovery(_op)


def get_recent_with_backfill(channel: str, channel_user_id: str, limit: int = 10) -> list[dict]:
    """Get recent messages, backfilling from Supabase if cache is empty."""
    if count(channel, channel_user_id) == 0:
        backfill_from_supabase(channel, channel_user_id, limit=max(50, limit))
    return get_recent(channel, channel_user_id, limit=limit)


def set_current_mission(
    channel: str,
    channel_user_id: str,
    mission: str,
    article_id: Optional[str] = None,
    status: str = "active",
) -> None:
    """Upsert current mission for a channel/user pair."""
    if status not in {"active", "done"}:
        status = "active"
    def _op(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            INSERT INTO current_missions (channel, channel_user_id, mission, article_id, status, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(channel, channel_user_id)
            DO UPDATE SET
                mission=excluded.mission,
                article_id=excluded.article_id,
                status=excluded.status,
                updated_at=excluded.updated_at
            """,
            (
                channel,
                channel_user_id,
                mission.strip(),
                article_id,
                status,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    _with_recovery(_op)


def get_current_mission(channel: str, channel_user_id: str) -> Optional[dict]:
    """Get current mission for user, if any."""
    def _op(conn: sqlite3.Connection) -> Optional[dict]:
        row = conn.execute(
            """
            SELECT channel, channel_user_id, mission, article_id, status, updated_at
            FROM current_missions
            WHERE channel = ? AND channel_user_id = ?
            """,
            (channel, channel_user_id),
        ).fetchone()
        return dict(row) if row else None
    return _with_recovery(_op)


def clear_current_mission(channel: str, channel_user_id: str) -> int:
    """Delete mission state for channel/user. Returns rows deleted."""
    def _op(conn: sqlite3.Connection) -> int:
        cur = conn.execute(
            "DELETE FROM current_missions WHERE channel = ? AND channel_user_id = ?",
            (channel, channel_user_id),
        )
        deleted = cur.rowcount
        conn.commit()
        return deleted
    return _with_recovery(_op)
