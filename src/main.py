from __future__ import annotations
"""FastAPI application for the article-writing chatbot."""

import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (works regardless of cwd when uvicorn runs)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import asyncio
import logging
from typing import Optional

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.observability import log_event, persist_trace, subscribe_events, unsubscribe_events

templates = Jinja2Templates(directory=Path(__file__).resolve().parent / "templates")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)

# Throttle for CRM progress messages: (channel, user_id) -> last_send_time
_progress_last_send: dict[tuple[str, str], float] = {}
_progress_lock = threading.Lock()

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure DB schema is up to date (adds missing columns if needed)
    from src.db import ensure_articles_schema, ensure_sessions_schema
    try:
        ensure_articles_schema()
    except Exception as e:
        logger.warning("Schema migration on startup failed (non-fatal): %s", e)
    try:
        ensure_sessions_schema()
    except Exception as e:
        logger.warning("Sessions schema migration on startup failed (non-fatal): %s", e)

    # Eager-load all prompts, then apply tool description overrides from PromptLayer
    from src.prompts_loader import load_all_prompts
    from src.agent import apply_tool_description_overrides
    load_all_prompts()
    apply_tool_description_overrides()
    yield

app = FastAPI(
    title="Article-Writing Chatbot",
    description="Chatbot that writes articles via deep research, PromptLayer, and more.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


import base64
import hashlib
import hmac
import secrets


_AUTH_TOKEN_TTL_SEC = 30 * 24 * 3600  # 30 days
_PROD_HOST_HINT = ("fly.dev", "fly.io")


def _is_localhost(host: str) -> bool:
    h = (host or "").split(":")[0].lower()
    return h in {"localhost", "127.0.0.1", "::1", ""}


def _mint_auth_token(password: str) -> str:
    """Return a base64-url token: expiry|nonce|hmac(expiry|nonce, password)."""
    expiry = int(time.time()) + _AUTH_TOKEN_TTL_SEC
    nonce = secrets.token_urlsafe(12)
    msg = f"{expiry}|{nonce}".encode()
    mac = hmac.new(password.encode(), msg, hashlib.sha256).hexdigest()
    raw = f"{expiry}|{nonce}|{mac}".encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _verify_auth_token(token: str, password: str) -> bool:
    """True iff token is a valid HMAC over expiry|nonce signed with password,
    and the expiry hasn't passed.
    """
    if not token or not password:
        return False
    try:
        padding = "=" * (-len(token) % 4)
        raw = base64.urlsafe_b64decode(token + padding).decode()
        expiry_str, nonce, mac = raw.rsplit("|", 2)
        expiry = int(expiry_str)
    except Exception:
        return False
    if expiry < int(time.time()):
        return False
    msg = f"{expiry}|{nonce}".encode()
    expected = hmac.new(password.encode(), msg, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, mac)


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """Gate all /api/* and /admin/* with the console password (raw header value
    OR a signed token issued by POST /api/auth/login).

    Localhost remains open if no password is configured (dev convenience).
    Anything else (prod, PR previews) MUST set CONSOLE_PASSWORD/WEB_API_KEY or
    everything 401s — no silent allow-all.
    """
    path = request.url.path
    open_paths = {"/", "/health", "/docs", "/openapi.json", "/redoc"}
    # Static SPA assets are public — the SPA enforces its own login flow
    # against /api/auth/login. Gating /console/* here would 401 the bundle
    # and leave the user staring at a JSON error instead of the login screen.
    if (
        path in open_paths
        or path.startswith("/webhooks/")
        or path.startswith("/console")
    ):
        return await call_next(request)
    # Auth endpoints are themselves open (the body carries the credential).
    if path in ("/api/auth/login", "/api/auth/me"):
        return await call_next(request)

    password = os.getenv("CONSOLE_PASSWORD") or os.getenv("WEB_API_KEY")
    host = request.headers.get("host", "")
    if not password:
        if _is_localhost(host) and not any(p in host for p in _PROD_HOST_HINT):
            # Local dev without a password configured — open.
            return await call_next(request)
        return JSONResponse(
            {"detail": "Server password is not configured. Set CONSOLE_PASSWORD."},
            status_code=503,
        )

    auth = request.headers.get("Authorization", "")
    x_key = request.headers.get("X-API-Key", "")
    # EventSource cannot send custom headers — allow `?key=...` as a fallback.
    query_key = request.query_params.get("key", "")
    if auth.startswith("Bearer "):
        provided = auth[7:]
    elif x_key:
        provided = x_key
    else:
        provided = query_key

    if not provided:
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    # Accept either the raw password (legacy / curl convenience) OR a valid
    # signed token previously issued by POST /api/auth/login.
    if hmac.compare_digest(provided, password) or _verify_auth_token(provided, password):
        return await call_next(request)
    return JSONResponse({"detail": "Unauthorized"}, status_code=401)


class LoginRequest(BaseModel):
    password: str


@app.post("/api/auth/login")
async def auth_login(req: LoginRequest):
    """Exchange the console password for a signed bearer token (TTL 30d).

    The token is HMAC(expiry|nonce, CONSOLE_PASSWORD) — stateless, so we don't
    need a sessions table. Verified by the api_key_middleware on every request.
    """
    password = os.getenv("CONSOLE_PASSWORD") or os.getenv("WEB_API_KEY")
    if not password:
        raise HTTPException(status_code=503, detail="Server password is not configured.")
    if not hmac.compare_digest(req.password, password):
        raise HTTPException(status_code=401, detail="Invalid password")
    token = _mint_auth_token(password)
    return {"token": token, "ttl_seconds": _AUTH_TOKEN_TTL_SEC}


@app.get("/api/auth/me")
async def auth_me(request: Request):
    """Returns whether the current credential is valid. Used by the SPA on
    boot to decide if a token in localStorage is still good.
    """
    password = os.getenv("CONSOLE_PASSWORD") or os.getenv("WEB_API_KEY")
    if not password:
        return {"authenticated": False, "reason": "no_password_configured"}
    auth = request.headers.get("Authorization", "")
    x_key = request.headers.get("X-API-Key", "")
    query_key = request.query_params.get("key", "")
    provided = auth[7:] if auth.startswith("Bearer ") else (x_key or query_key)
    ok = bool(provided) and (
        hmac.compare_digest(provided, password) or _verify_auth_token(provided, password)
    )
    return {"authenticated": ok}


class ChatRequest(BaseModel):
    message: str
    channel_user_id: Optional[str] = None  # For web: defaults to "default" if not provided


class ChatResponse(BaseModel):
    message: str
    article: Optional[dict] = None


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/")
async def root():
    """Root endpoint — redirect to the console SPA if it's been built,
    otherwise fall back to the API docs."""
    from fastapi.responses import RedirectResponse
    target = "/console/" if _WEB_DIST.exists() else "/docs"
    return RedirectResponse(url=target)


# Mount the React console SPA at /console if the bundle exists.
# Dev: `cd web && npm run dev` (Vite on :5173 proxies to this server on :8000).
# Prod: the Dockerfile builds web/dist via a Node stage and copies it in.
_WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"
if _WEB_DIST.exists():
    app.mount("/console", StaticFiles(directory=str(_WEB_DIST), html=True), name="console")
    logger.info("[CONSOLE] mounted /console -> %s", _WEB_DIST)
else:
    logger.info("[CONSOLE] %s not found - run `cd web && npm run build` to enable /console", _WEB_DIST)


@app.get("/api/chat/status")
async def chat_status(channel_user_id: Optional[str] = None):
    """
    Get current task status for a user (e.g. during article writing).
    Returns {busy: bool, status?: str} without starting an agent run.
    """
    from src.task_state import has_task, get_task

    user_key = channel_user_id or "default"
    if has_task("web", user_key):
        task = get_task("web", user_key)
        return {"busy": True, "status": task.status_text if task else "Working..."}
    return {"busy": False}


@app.get("/api")
async def api_info():
    """API info."""
    return {
        "message": "Article-Writing Chatbot API",
        "docs": "/docs",
        "chat": "POST /api/chat",
        "chat_stream": "POST /api/chat/stream (SSE with status updates)",
        "articles": "GET /api/articles (list saved articles)",
        "export": "GET /api/articles/{id}/export (markdown)",
        "create_google_doc": "POST /api/articles/{id}/google-doc",
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Chat endpoint - channel-agnostic.
    Receives message + optional channel_user_id (web defaults to "default").
    """
    from src.agent import run_agent
    from src.db import get_messages_for_user, add_message_for_user
    from src.response_interpreter import explain_to_user

    channel_user_id = req.channel_user_id or "default"
    trace_id = str(uuid.uuid4())
    log_event(
        "request_start",
        channel="web",
        channel_user_id=channel_user_id,
        message_preview=req.message[:100] + ("..." if len(req.message) > 100 else ""),
    )
    logger.info("[CHAT] Request: message=%r, channel_user_id=%s", req.message[:100], channel_user_id)

    msgs = get_messages_for_user("web", channel_user_id)
    history = [{"role": m["role"], "content": m["content"]} for m in msgs]
    logger.info("[CHAT] Loaded %d messages for web/%s", len(history), channel_user_id)

    request_start = time.perf_counter()
    try:
        result = run_agent(
            user_message=req.message,
            history=history,
            channel="web",
            channel_user_id=channel_user_id,
            trace_id=trace_id,
        )
    except Exception as e:
        logger.exception("[CHAT] Agent error: %s", e)
        friendly = explain_to_user(str(e))
        raise HTTPException(status_code=500, detail=friendly)

    message_out = result.get("message", "")
    article = result.get("article")
    if article and article.get("google_doc_url"):
        message_out = f"Here's your article: {article['google_doc_url']}"
    elif article and article.get("content"):
        message_out = article["content"]
    add_message_for_user("web", channel_user_id, "user", req.message)
    add_message_for_user("web", channel_user_id, "assistant", message_out)

    total_ms = (time.perf_counter() - request_start) * 1000
    log_event(
        "request_done",
        total_latency_ms=round(total_ms, 2),
        has_article=article is not None,
        final_message_len=len(message_out),
    )
    persist_trace(message_out)

    logger.info("[CHAT] Completed: has_article=%s", article is not None)
    return ChatResponse(
        message=message_out,
        article=article,
    )


def _sse(event: dict) -> str:
    """Serialize an event for SSE wire format."""
    return f"data: {json.dumps(event, default=str)}\n\n"


def _generate_session_title_async(session_id: str, user_message: str) -> None:
    """Best-effort Haiku call to produce a short session title from the first
    user message. Runs in a daemon thread; any failure is logged and swallowed.
    Prioritizes the article topic / main intent ('Find me X about Y', etc.).
    """
    def _run() -> None:
        try:
            from anthropic import Anthropic
            client = Anthropic()
            resp = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=40,
                system=(
                    "Return a 3-7 word title (no quotes, no period) that captures the "
                    "user's request in this message. If they're asking for an article, "
                    "use the article topic. If it's a research request, use 'Find X about Y' "
                    "style. Otherwise summarize the main intent in plain English. "
                    "Output the title only, nothing else."
                ),
                messages=[{"role": "user", "content": user_message[:1000]}],
            )
            text_blocks = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
            title = " ".join(text_blocks).strip().strip('"\'`').rstrip(".")
            if not title:
                return
            from src.db import rename_session
            rename_session(session_id, title[:120])
            logger.info("[SESSION] Auto-titled %s -> %r", session_id[:8], title[:60])
        except Exception as e:
            logger.warning("[SESSION] Title generation failed for %s: %s", session_id[:8], e)

    threading.Thread(target=_run, daemon=True).start()


def _run_agent_into_buffer(buf, session_id: str, channel: str, channel_user_id: str,
                            message: str, history: list) -> None:
    """Run the agent in the current thread, writing all events into ``buf``
    and persisting the user message + final assistant message (or stub on
    error/cancel) to Supabase tied to the session.
    """
    from src.agent import run_agent
    from src.db import add_message_for_session, get_session, rename_session
    from src.task_state import set_task_by_key, clear_task_by_key, is_cancel_requested_by_key
    from src.response_interpreter import explain_to_user

    task_key = f"session:{session_id}"
    set_task_by_key(task_key, "Processing your request...")
    req_start = time.perf_counter()

    def on_status(text: str) -> None:
        set_task_by_key(task_key, text)
        buf.append("status", text=text)

    def on_token(delta: str) -> None:
        buf.append("token", text=delta)

    try:
        result = run_agent(
            user_message=message,
            history=history,
            channel=channel,
            channel_user_id=channel_user_id,
            on_status=on_status,
            on_token=on_token,
            trace_id=buf.run_id,
            session_id=session_id,
        )

        article = result.get("article")
        message_out = result.get("message", "")
        if article and article.get("google_doc_url"):
            message_out = f"Here's your article: {article['google_doc_url']}"
        elif article and article.get("content"):
            message_out = article["content"]

        # Persist user + assistant turn into the session transcript.
        add_message_for_session(session_id, channel, channel_user_id, "user", message)
        add_message_for_session(session_id, channel, channel_user_id, "assistant", message_out)

        total_ms = (time.perf_counter() - req_start) * 1000
        log_event(
            "request_done",
            total_latency_ms=round(total_ms, 2),
            has_article=article is not None,
            final_message_len=len(message_out),
        )
        persist_trace(message_out)

        buf.finish({"message": message_out, "article": article})

        # Auto-title sessions that don't have one yet.
        try:
            sess = get_session(session_id)
            if sess and not (sess.get("title") or "").strip():
                _generate_session_title_async(session_id, message)
        except Exception:
            pass

    except Exception as e:
        logger.exception("[STREAM] Agent error for session %s: %s", session_id[:8], e)
        # Persist user message + a stub assistant reply so the transcript
        # reflects what was attempted when the user comes back.
        cancelled = is_cancel_requested_by_key(task_key)
        stub = "[cancelled]" if cancelled else f"[error: {explain_to_user(str(e))}]"
        try:
            add_message_for_session(session_id, channel, channel_user_id, "user", message)
            add_message_for_session(session_id, channel, channel_user_id, "assistant", stub)
        except Exception as persist_err:
            logger.warning("[STREAM] Failed to persist stub message: %s", persist_err)
        buf.fail(stub)
    finally:
        clear_task_by_key(task_key)


def _session_chat_generator(session_id: str, channel: str, channel_user_id: str,
                             message: str, history: list):
    """SSE generator: start a new run for the session and stream events.

    Yields a 409-shaped event if there's already an active run for this session.
    """
    from src.run_buffer import start_run, replay_and_tail

    trace_id = str(uuid.uuid4())
    log_event(
        "request_start",
        channel=channel,
        channel_user_id=channel_user_id,
        session_id=session_id,
        message_preview=message[:100] + ("..." if len(message) > 100 else ""),
    )

    buf = start_run(session_id, trace_id)
    if buf is None:
        yield _sse({
            "type": "error",
            "code": "session_busy",
            "message": "This session already has a run in progress.",
        })
        return

    logger.info("[STREAM] Starting run for session %s, history_len=%d", session_id[:8], len(history))

    thread = threading.Thread(
        target=_run_agent_into_buffer,
        args=(buf, session_id, channel, channel_user_id, message, history),
        daemon=True,
    )
    thread.start()

    for event in replay_and_tail(buf, from_seq=0):
        yield _sse(event)


def _attach_session_stream(session_id: str, from_seq: int):
    """SSE generator that attaches to an existing run's buffer (if any)."""
    from src.run_buffer import get_run, replay_and_tail

    buf = get_run(session_id)
    if buf is None:
        # No active run — emit a snapshot saying so and close.
        yield _sse({
            "seq": from_seq,
            "type": "snapshot",
            "session_id": session_id,
            "running": False,
            "last_seq": from_seq,
            "last_tool": None,
            "last_status": None,
        })
        return

    for event in replay_and_tail(buf, from_seq=from_seq):
        yield _sse(event)


# --- Session CRUD + run endpoints --------------------------------------------

class SessionCreateRequest(BaseModel):
    channel_user_id: Optional[str] = None
    title: Optional[str] = None


class SessionRenameRequest(BaseModel):
    title: str


class SessionChatRequest(BaseModel):
    message: str


_DEFAULT_CHANNEL = "web"
_DEFAULT_CHANNEL_USER_ID = "web-console"


def _running_meta(session_id: str) -> dict:
    """Look up the run buffer for a session and return a small status dict
    suitable for inlining into a sessions list response.
    """
    from src.run_buffer import get_run
    buf = get_run(session_id)
    if not buf:
        return {"running": False, "last_tool": None, "last_status": None, "run_id": None}
    snap = buf.snapshot()
    return {
        "running": snap["running"],
        "last_tool": snap["last_tool"],
        "last_status": snap["last_status"],
        "run_id": snap["run_id"],
    }


@app.get("/api/sessions")
async def list_sessions_endpoint(channel_user_id: Optional[str] = None):
    """List sessions for the given web user. Each entry includes the live
    running state (derived from the in-memory run buffer) so the sidebar can
    show running/idle dots and the current tool/status per session.
    """
    from src.db import list_sessions
    user = channel_user_id or _DEFAULT_CHANNEL_USER_ID
    sessions = list_sessions(_DEFAULT_CHANNEL, user)
    for s in sessions:
        s.update(_running_meta(s["id"]))
    return {"sessions": sessions}


@app.post("/api/sessions")
async def create_session_endpoint(req: Optional[SessionCreateRequest] = Body(None)):
    from src.db import create_session
    user = (req.channel_user_id if req else None) or _DEFAULT_CHANNEL_USER_ID
    title = req.title if req else None
    session = create_session(_DEFAULT_CHANNEL, user, title=title)
    session.update({"running": False, "last_tool": None, "last_status": None, "run_id": None})
    return session


@app.get("/api/sessions/{session_id}")
async def get_session_endpoint(session_id: str):
    """Single-session fetch. Useful for the ChatPane header which needs the
    title (especially after the async Haiku rename finishes)."""
    from src.db import get_session
    sess = get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    sess.update(_running_meta(session_id))
    return sess


@app.patch("/api/sessions/{session_id}")
async def rename_session_endpoint(session_id: str, req: SessionRenameRequest):
    from src.db import rename_session, get_session
    if not get_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return rename_session(session_id, req.title)


@app.delete("/api/sessions/{session_id}")
async def delete_session_endpoint(session_id: str):
    """Cancel the in-flight run (if any), then soft-delete the session."""
    from src.db import soft_delete_session, get_session
    from src.run_buffer import get_run, discard_run
    from src.task_state import request_cancel_by_key

    if not get_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    buf = get_run(session_id)
    if buf and not buf.finished:
        request_cancel_by_key(f"session:{session_id}")
        buf.request_cancel()
        # Don't wait for the agent to honor it — soft-delete makes the row
        # invisible immediately. Buffer is discarded so reconnects 404.
    discard_run(session_id)
    soft_delete_session(session_id)
    return {"ok": True}


@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    from src.db import get_session, get_messages_for_session
    if not get_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    messages = get_messages_for_session(session_id)
    return {"messages": messages}


@app.post("/api/sessions/{session_id}/chat")
async def session_chat_endpoint(session_id: str, req: SessionChatRequest):
    """Kick off a new run on this session. SSE stream from seq 0."""
    from src.db import get_session, get_messages_for_session

    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    msgs = get_messages_for_session(session_id)
    history = [{"role": m["role"], "content": m["content"]} for m in msgs]
    channel = session.get("channel") or _DEFAULT_CHANNEL
    channel_user_id = session.get("channel_user_id") or _DEFAULT_CHANNEL_USER_ID

    return StreamingResponse(
        _session_chat_generator(session_id, channel, channel_user_id, req.message, history),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/sessions/{session_id}/stream")
async def session_stream_attach(session_id: str, from_seq: int = 0):
    """Attach to an in-flight run. Replays from ``from_seq`` then live-tails.
    First event yielded is always a `snapshot` so the client can render
    immediately.
    """
    from src.db import get_session
    if not get_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return StreamingResponse(
        _attach_session_stream(session_id, from_seq),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/sessions/{session_id}/cancel")
async def session_cancel_endpoint(session_id: str):
    from src.db import get_session
    from src.run_buffer import get_run
    from src.task_state import request_cancel_by_key

    if not get_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    cancelled = request_cancel_by_key(f"session:{session_id}")
    buf = get_run(session_id)
    if buf:
        buf.request_cancel()
    return {"cancelled": cancelled}


@app.get("/api/events/stream")
async def events_stream():
    """SSE firehose of all log_event calls across the server.

    Subscribers receive every structured event in real time. Used by the
    web console's live-stream drawer. Auth: same WEB_API_KEY as other
    /api/* routes (header or ?key= query param — EventSource can't send headers).
    """
    loop, q = subscribe_events()

    async def gen():
        try:
            # Initial hello so the client knows the connection is alive
            yield f"data: {json.dumps({'event_type': 'connected', 'ts': time.time()})}\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=20.0)
                    yield f"data: {json.dumps(event, default=str)}\n\n"
                except asyncio.TimeoutError:
                    # Keepalive comment — prevents proxies from closing idle conns
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            raise
        finally:
            unsubscribe_events(loop, q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _log_whatsapp(msg: str) -> None:
    """Print to stderr so it appears even when uvicorn buffers stdout."""
    print(msg, file=sys.stderr, flush=True)


def _process_whatsapp_message(
    chat_id: str, text: str, quoted_message_id: Optional[str] = None
) -> None:
    """Process incoming WhatsApp message via Opus agent (runs in background)."""
    trace_id = str(uuid.uuid4())
    whatsapp_start = time.perf_counter()
    log_event(
        "request_start",
        channel="whatsapp",
        channel_user_id=chat_id,
        message_preview=text[:100] + ("..." if len(text) > 100 else ""),
    )
    logger.info("[WHATSAPP] Processing: chat_id=%s, text=%r", chat_id, text[:100])
    _log_whatsapp(f"[WhatsApp] IN {chat_id}: {repr(text)}")

    from src.agent import run_agent
    from src.channels.whatsapp import send_message_chunked
    from src.db import add_message_for_user
    from src.task_state import set_task, clear_task

    set_task("whatsapp", chat_id, "Processing your request...")

    def _on_status(status_text: str) -> None:
        set_task("whatsapp", chat_id, status_text)

    response = ""
    try:
        result = run_agent(
            user_message=text,
            history=[],  # run_agent loads its own history from SQLite cache
            channel="whatsapp",
            channel_user_id=chat_id,
            format_for_whatsapp=True,
            on_status=_on_status,
            trace_id=trace_id,
        )
        response = result.get("message", "")
    except Exception as e:
        logger.exception("[WHATSAPP] Agent error for %s: %s", chat_id, e)
        from src.response_interpreter import explain_to_user
        response = explain_to_user(str(e))
        add_message_for_user("whatsapp", chat_id, "user", text)
        add_message_for_user("whatsapp", chat_id, "assistant", response)
        _log_whatsapp(f"[WhatsApp] OUT {chat_id} (error->interpreted): {repr(response[:200])}")
        try:
            send_message_chunked(chat_id, response, quoted_message_id=quoted_message_id)
        except Exception as send_err:
            logger.exception("Failed to send error reply: %s", send_err)
        return
    finally:
        clear_task("whatsapp", chat_id)
        with _progress_lock:
            _progress_last_send.pop(("whatsapp", chat_id), None)

    logger.info("[WHATSAPP] Agent done: response_len=%d", len(response))
    out_preview = response[:200] + ("..." if len(response) > 200 else "")
    _log_whatsapp(f"[WhatsApp] OUT {chat_id}: {repr(out_preview)}")

    # Send reply immediately — don't wait for DB writes
    try:
        if response:
            send_message_chunked(chat_id, response, quoted_message_id=quoted_message_id)
    except Exception as e:
        logger.exception("Failed to send WhatsApp reply to %s: %s", chat_id, e)

    # Persist to Supabase + observability in background (SQLite cache is already written via add_message)
    def _persist():
        try:
            add_message_for_user("whatsapp", chat_id, "user", text)
            add_message_for_user("whatsapp", chat_id, "assistant", response)
            total_ms = (time.perf_counter() - whatsapp_start) * 1000
            log_event("request_done", total_latency_ms=round(total_ms, 2), final_message_len=len(response))
            persist_trace(response)
        except Exception as e:
            logger.warning("[WHATSAPP] Background persist failed: %s", e)

    import threading
    threading.Thread(target=_persist, daemon=True).start()


@app.post("/webhooks/green-api/whatsapp")
async def green_api_whatsapp_webhook(request: Request):
    """
    Green API webhook: receives incoming WhatsApp messages.
    Returns 200 immediately to avoid timeout; processes and replies in background.
    """
    from src.channels.whatsapp import parse_incoming_text_webhook, send_message_chunked
    from src.task_state import has_task, get_task, is_status_question, is_cancel_message, request_cancel

    try:
        payload = await request.json()
    except Exception:
        return {"ok": True}

    _log_whatsapp(f"[WhatsApp] Webhook payload:\n{json.dumps(payload, indent=2, default=str)}")

    parsed = parse_incoming_text_webhook(payload)
    if not parsed:
        _log_whatsapp("[WhatsApp] Webhook: not a text message, ignoring")
        return {"ok": True}

    chat_id = parsed["chatId"]
    text = parsed["textMessage"]
    quoted = parsed.get("quotedMessage")
    # Only quote when user replied to a specific message (keeps thread clear)
    quoted_message_id = parsed.get("idMessage") if quoted else None

    if quoted:
        text = f"[Replying to: \"{quoted}\"]\n\n{text}"

    if not text:
        return {"ok": True}

    # Only respond to this chat ID - ignore all others
    ALLOWED_CHAT_ID = os.getenv("WHATSAPP_ALLOWED_CHAT_ID", "")
    if chat_id != ALLOWED_CHAT_ID:
        _log_whatsapp(f"[WhatsApp] Ignored {chat_id} (allowed: {ALLOWED_CHAT_ID}): {repr(text)[:50]}")
        return {"ok": True}

    # If a task is in flight: cancel -> set cancel flag; status question -> reply with status; else -> ask to wait
    if has_task("whatsapp", chat_id):
        if is_cancel_message(text):
            cancelled = request_cancel("whatsapp", chat_id)
            cancel_msg = "Cancelling..." if cancelled else "Nothing to cancel."
            asyncio.create_task(
                asyncio.to_thread(
                    send_message_chunked,
                    chat_id,
                    cancel_msg,
                    quoted_message_id=quoted_message_id,
                )
            )
            _log_whatsapp(f"[WhatsApp] Cancel request from {chat_id}: {cancel_msg}")
        elif is_status_question(text):
            task = get_task("whatsapp", chat_id)
            status_msg = (task.status_text if task else "Still working...") + " I'll continue with your previous request."
            asyncio.create_task(
                asyncio.to_thread(
                    send_message_chunked,
                    chat_id,
                    status_msg,
                    quoted_message_id=quoted_message_id,
                )
            )
            _log_whatsapp(f"[WhatsApp] Status reply to {chat_id}: {status_msg[:60]}")
        else:
            asyncio.create_task(
                asyncio.to_thread(
                    send_message_chunked,
                    chat_id,
                    "Still working on your previous request. Say *stop* to cancel, or wait for it to finish.",
                    quoted_message_id=quoted_message_id,
                )
            )
            _log_whatsapp(f"[WhatsApp] Busy reply to {chat_id}")
        return {"ok": True}

    _log_whatsapp(f"[WhatsApp] Received from {chat_id}, processing...")
    asyncio.create_task(
        asyncio.to_thread(_process_whatsapp_message, chat_id, text, quoted_message_id)
    )

    return {"ok": True}


def _process_gdrive_change(article_id: str) -> None:
    """Fetch the Google Doc and detect manual edits. Runs in a background thread."""
    from src.db import get_article
    from src.google_docs import fetch_doc_content, _document_id_from_url
    from src.tools import _detect_and_log_manual_edits

    try:
        article = get_article(article_id)
        if not article:
            logger.warning("[GDRIVE] article %s not found", article_id[:8])
            return

        google_doc_url = article.get("google_doc_url")
        if not google_doc_url:
            return

        doc_id = _document_id_from_url(google_doc_url)
        if not doc_id:
            return

        doc_result = fetch_doc_content(doc_id)
        if not doc_result.get("success"):
            logger.warning("[GDRIVE] Could not fetch doc for article %s: %s", article_id[:8], doc_result.get("error"))
            return

        gdoc_text = doc_result.get("content", "")
        db_content = article.get("content", "")
        _detect_and_log_manual_edits(article_id, db_content, gdoc_text)

        # Opportunistically renew the watch if expiring within 24 hours
        try:
            from src.db import get_gdrive_watch
            from src.google_docs import ensure_gdrive_watch
            watch = get_gdrive_watch(article_id)
            if watch and doc_id:
                import time as _time
                expiry_ms = watch.get("expiry_ms", 0)
                if expiry_ms - int(_time.time() * 1000) < 86_400_000:
                    renewed = ensure_gdrive_watch(doc_id, article_id)
                    if renewed:
                        logger.info("[GDRIVE] Watch renewed for article %s (channel=%s)", article_id[:8], renewed["channel_id"][:8])
        except Exception as re:
            logger.warning("[GDRIVE] Watch renewal failed: %s", re)

        logger.info("[GDRIVE] Processed change for article %s", article_id[:8])
    except Exception as e:
        logger.exception("[GDRIVE] Error processing change notification: %s", e)


@app.post("/webhooks/gdrive-changes")
async def gdrive_changes_webhook(request: Request):
    """Google Drive push notification webhook.

    Drive sends a POST here whenever a watched document changes.
    Key headers:
      X-Goog-Channel-Id      — our channel UUID
      X-Goog-Channel-Token   — article_id we supplied at registration time
      X-Goog-Resource-State  — 'sync' (initial ack) | 'update' | 'change'
      X-Goog-Message-Number  — sequential int (1 = sync)
    """
    resource_state = request.headers.get("X-Goog-Resource-State", "")
    channel_id = request.headers.get("X-Goog-Channel-Id", "")
    article_id = request.headers.get("X-Goog-Channel-Token", "")

    logger.info(
        "[GDRIVE] Webhook: state=%s, channel=%s, article=%s",
        resource_state,
        channel_id[:8] if channel_id else "?",
        article_id[:8] if article_id else "?",
    )

    # Always return 200 immediately — Drive will retry on non-2xx
    if resource_state == "sync":
        # Initial acknowledgement event; no actual change yet
        return {"ok": True}

    if not article_id or resource_state not in ("update", "change"):
        return {"ok": True}

    # Validate article_id looks like a UUID before touching the DB
    import re as _re
    if not _re.match(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$', article_id):
        logger.warning("[GDRIVE] Invalid article_id in token: %r", article_id[:40])
        return {"ok": True}

    asyncio.create_task(asyncio.to_thread(_process_gdrive_change, article_id))
    return {"ok": True}


@app.get("/api/articles")
async def list_articles(
    channel: Optional[str] = None,
    channel_user_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """
    List saved articles. All articles are saved on generation and on every change.
    Optional: filter by channel and channel_user_id.
    """
    from src.db import list_articles as _list

    articles = _list(channel=channel, channel_user_id=channel_user_id, limit=limit, offset=offset)
    return {"articles": articles, "count": len(articles)}


@app.get("/api/articles/{article_id}/export")
async def export_article(article_id: str, format: str = "markdown"):
    """
    Export article as Markdown or plain text.
    GET /api/articles/{id}/export?format=markdown
    """
    from src.db import get_article

    article = get_article(article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")

    content = article.get("content", "")
    filename = f"article-{article_id[:8]}.md"
    if format == "markdown":
        return PlainTextResponse(
            content,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    return PlainTextResponse(content, media_type="text/plain")


class CreateGoogleDocRequest(BaseModel):
    title: Optional[str] = None


class ArticleStatusRequest(BaseModel):
    status: str  # draft, posted, in_progress


@app.patch("/api/articles/{article_id}/status")
async def update_article_status(article_id: str, req: ArticleStatusRequest):
    """Update article status: draft, posted, in_progress."""
    from src.db import set_article_status, get_article

    article = get_article(article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    try:
        updated = set_article_status(article_id, req.status)
        return updated
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/articles/{article_id}/google-doc")
async def create_google_doc_endpoint(article_id: str, req: Optional[CreateGoogleDocRequest] = Body(None)):
    """
    Create a Google Doc from the article. Requires GOOGLE_APPLICATION_CREDENTIALS.
    """
    from src.tools import google_docs_tool

    title = req.title if req else None
    result = google_docs_tool(action="create", article_id=article_id, title=title)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to create Google Doc"))
    return result


# --- Observability admin ---

@app.get("/admin/traces")
async def admin_traces(
    request: Request,
    channel: Optional[str] = None,
    channel_user_id: Optional[str] = None,
    limit: int = 50,
):
    """Admin page: list traces with responsive collapsible hierarchy."""
    from src.db import list_traces
    from src.observability import compute_trace_summary

    traces = list_traces(channel=channel, channel_user_id=channel_user_id, limit=limit)
    for t in traces:
        if isinstance(t.get("created_at"), str) and "T" in t["created_at"]:
            pass
        elif hasattr(t.get("created_at"), "isoformat"):
            t["created_at"] = t["created_at"].isoformat()
        # Compute summary stats for table view
        payload = t.get("payload") or {}
        t["summary"] = payload.get("summary") or compute_trace_summary(payload)
    return templates.TemplateResponse(
        "admin/traces.html",
        {"request": request, "traces": traces, "channel": channel, "channel_user_id": channel_user_id},
    )


@app.get("/admin")
async def admin_root():
    """Admin root - redirect to traces dashboard."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/admin/traces")


@app.get("/api/traces")
async def api_list_traces(
    channel: Optional[str] = None,
    channel_user_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """API: list traces."""
    from src.db import list_traces

    traces = list_traces(channel=channel, channel_user_id=channel_user_id, limit=limit, offset=offset)
    return {"traces": traces, "count": len(traces)}


@app.get("/api/traces/{trace_id}")
async def api_get_trace(trace_id: str):
    """API: get single trace with full payload."""
    from src.db import get_trace

    trace = get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace



# --- Admin: Agent config and prompts ---

# Provider → models: chat/completion (all text models for any agent)
PROVIDERS_MODELS_CHAT = {
    "anthropic": ["claude-opus-4-20250514", "claude-opus-4-5-20251101", "claude-opus-4-1-20250805", "claude-sonnet-4-5", "claude-4-5-sonnet", "claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
    "openai": ["gpt-4o", "gpt-4o-mini", "o3-mini", "o3", "o3-deep-research"],
    "google": ["gemini-2.0-flash", "gemini-1.5-flash"],
    "perplexity": ["sonar-pro", "sonar"],
}

# Provider → models: image generation
PROVIDERS_MODELS_IMAGE = {
    "openai": ["dall-e-3"],
    "google": ["gemini-3-pro-image-preview", "gemini-2.5-flash-image", "imagen-3.0-generate-002"],
}

MODEL_TO_PROVIDER = {}
for prov, models in PROVIDERS_MODELS_CHAT.items():
    for m in models:
        MODEL_TO_PROVIDER[m] = prov
for prov, models in PROVIDERS_MODELS_IMAGE.items():
    for m in models:
        MODEL_TO_PROVIDER[m] = prov

# Agent entity definitions
AGENTS_DEFINITION = [
    {
        "id": "master_agent",
        "name": "Master Agent",
        "description": "Main orchestrator. Handles conversation and routes to tools.",
        "config_keys": [
            {"key": "agent_model", "label": "Model", "type": "provider_model"},
            {"key": "agent_history_limit", "label": "History limit", "type": "number", "min": 5, "max": 100},
        ],
        "model_type": "chat",
        "prompts": ["master_system_core", "playbooks", "whatsapp_format"],
        "tools": ["db", "google_docs", "web_search", "send_image", "write_article", "improve_article", "generate_images", "generate_hero_image", "approve_hero_image", "generate_infographic", "approve_infographic", "clean_memory"],
        "sub_agents": ["Article Writer", "Article Improver", "Image Placement", "Infographic Analyzer", "Hero & Infographic Generator", "Generic Image Generator"],
    },
    {
        "id": "article_writer",
        "name": "Article Writer",
        "description": "PromptLayer SEO pipeline. Invoked by write_article.",
        "config_keys": [{"key": "deep_research_model", "label": "Model", "type": "provider_model"}],
        "model_type": "chat",
        "prompts": ["deep_research"],
        "tools": [],
        "sub_agents": [],
    },
    {
        "id": "article_improver",
        "name": "Article Improver",
        "description": "Edits and improves articles. Invoked by improve_article.",
        "config_keys": [{"key": "article_write_model", "label": "Model", "type": "provider_model"}],
        "model_type": "chat",
        "prompts": ["improve_article"],
        "tools": [],
        "sub_agents": [],
    },
    {
        "id": "image_placement",
        "name": "Image Placement",
        "description": "Suggests where to insert images in articles. Used by generate_images.",
        "config_keys": [{"key": "image_placement_model", "label": "Model", "type": "provider_model"}],
        "model_type": "chat",
        "prompts": ["image_placement"],
        "tools": [],
        "sub_agents": [],
    },
    {
        "id": "infographic_analyzer",
        "name": "Infographic Analyzer",
        "description": "Analyzes articles for best infographic placement and type.",
        "config_keys": [{"key": "infographic_analysis_model", "label": "Model", "type": "provider_model"}],
        "model_type": "chat",
        "prompts": ["infographic_analysis"],
        "tools": [],
        "sub_agents": [],
    },
    {
        "id": "hero_infographic_generator",
        "name": "Hero & Infographic Generator",
        "description": "Generates hero images and infographics with style references.",
        "config_keys": [{"key": "image_model_hero", "label": "Model", "type": "provider_model"}],
        "model_type": "image",
        "prompts": ["hero_image", "infographic_generation"],
        "tools": [],
        "sub_agents": [],
    },
    {
        "id": "generic_image_generator",
        "name": "Generic Image Generator",
        "description": "Generates illustrations for article body. Used by generate_images.",
        "config_keys": [{"key": "image_model_generic", "label": "Model", "type": "provider_model"}],
        "model_type": "image",
        "prompts": [],
        "tools": [],
        "sub_agents": [],
    },
]


@app.get("/admin/graph-state")
async def admin_graph_state(request: Request):
    """Admin page: active article threads (legacy table)."""
    return templates.TemplateResponse(
        "admin/graph_state.html",
        {"request": request},
    )


@app.get("/api/admin/graph-state")
async def api_get_graph_state(whatsapp_user_id: Optional[str] = None):
    """Returns active article threads from DB (legacy table, agent is now stateless)."""
    from src.db import get_client
    try:
        r = get_client().table("active_article_threads").select("*").execute()
        threads = r.data or []
    except Exception:
        threads = []
    thread = None
    if whatsapp_user_id:
        thread = next((t for t in threads if t.get("whatsapp_user_id") == whatsapp_user_id), None)
    return {
        "threads": threads,
        "thread": thread,
        "state": None,  # Graph removed; no state to show
        "next": [],
    }


@app.get("/admin/agents")
async def admin_agents(request: Request):
    """Admin page: configure agents and models."""
    return templates.TemplateResponse(
        "admin/agents.html",
        {"request": request},
    )


@app.get("/api/admin/agents")
async def api_get_agents():
    """Get all agents as entities with config, prompts, tools, sub_agents, providers_models."""
    from src.config import get_all_config
    from src.prompts_loader import get_all_prompts

    config = get_all_config()
    prompts = get_all_prompts()
    agents = []
    for a in AGENTS_DEFINITION:
        agent = dict(a)
        agent["config"] = {k["key"]: config.get(k["key"], "") for k in a["config_keys"]}
        agent["prompt_contents"] = {k: prompts.get(k, "") for k in a["prompts"]}
        mt = a.get("model_type", "chat")
        agent["providers_models"] = PROVIDERS_MODELS_IMAGE if mt == "image" else PROVIDERS_MODELS_CHAT
        agents.append(agent)
    return {
        "agents": agents,
        "model_to_provider": MODEL_TO_PROVIDER,
    }


@app.get("/api/admin/agent-config")
async def api_get_agent_config():
    """Get all agent config (models, limits)."""
    from src.config import get_all_config
    return get_all_config()


@app.patch("/api/admin/agent-config")
async def api_patch_agent_config(req: dict = Body(...)):
    """Update agent config. Expects {key: value, ...}."""
    from src.config import invalidate_config_cache
    from src.db import upsert_agent_config

    for key, value in req.items():
        if isinstance(value, (str, int, float)):
            upsert_agent_config(key, str(value))
    invalidate_config_cache()
    return {"ok": True}


@app.get("/api/admin/prompts")
async def api_get_prompts():
    """Get all prompts."""
    from src.prompts_loader import get_all_prompts
    return get_all_prompts()


@app.patch("/api/admin/prompts")
async def api_patch_prompts(req: dict = Body(...)):
    """Update prompts. Expects {key: content, ...}."""
    from src.prompts_loader import invalidate_prompts_cache
    from src.db import upsert_prompt

    for key, content in req.items():
        if isinstance(content, str):
            upsert_prompt(key, content)
    invalidate_prompts_cache()
    return {"ok": True}


@app.post("/api/admin/optimize")
async def api_run_optimization(req: dict = Body(default={})):
    """Trigger a self-optimization analysis in the background.

    Optional body: {"window_hours": 24, "max_traces": 200, "dry_run": false}
    Returns immediately. Results are saved to optimization_sessions and sent via WhatsApp.
    """
    from fastapi import BackgroundTasks

    window_hours = int(req.get("window_hours", 24))
    max_traces   = int(req.get("max_traces", 200))
    dry_run      = bool(req.get("dry_run", False))

    def _run():
        import subprocess, sys, os
        cmd = [sys.executable, "scripts/self_optimize.py", "--window", str(window_hours), "--n", str(max_traces)]
        if dry_run:
            cmd.append("--dry-run")
        cwd = Path(__file__).resolve().parent.parent
        result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logging.getLogger(__name__).error("[OPTIMIZE] Failed: %s", result.stderr[:500])
        else:
            logging.getLogger(__name__).info("[OPTIMIZE] Completed. stdout: %s", result.stdout[-500:])

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return {"ok": True, "message": f"Optimization started (window={window_hours}h, max_traces={max_traces}, dry_run={dry_run})"}


@app.get("/api/admin/optimization-sessions")
async def api_list_optimization_sessions():
    """List recent optimization sessions."""
    from src.db import list_optimization_sessions
    sessions = list_optimization_sessions(limit=10)
    return {"sessions": sessions}
