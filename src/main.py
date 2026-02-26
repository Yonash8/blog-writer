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
from fastapi.templating import Jinja2Templates

from src.observability import log_event, persist_trace

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
    from src.db import ensure_articles_schema
    try:
        ensure_articles_schema()
    except Exception as e:
        logger.warning("Schema migration on startup failed (non-fatal): %s", e)

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


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """Require X-API-Key or Bearer token on all /api/* and /admin/* routes."""
    path = request.url.path
    open_paths = {"/", "/health", "/docs", "/openapi.json", "/redoc"}
    if path in open_paths or path.startswith("/webhooks/"):
        return await call_next(request)

    web_api_key = os.getenv("WEB_API_KEY")
    if not web_api_key:
        # No key configured — allow all (local dev)
        return await call_next(request)

    auth = request.headers.get("Authorization", "")
    x_key = request.headers.get("X-API-Key", "")
    provided = auth[7:] if auth.startswith("Bearer ") else x_key

    if provided != web_api_key:
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)


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
    """Root endpoint - redirect to API docs."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


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


def _stream_chat_generator(message: str, channel_user_id: str, history: list):
    """Sync generator for SSE stream: status updates then final result."""
    from src.agent import run_agent
    from src.db import add_message_for_user
    from src.task_state import set_task, clear_task

    user_key = channel_user_id or "default"
    trace_id = str(uuid.uuid4())
    log_event(
        "request_start",
        channel="web",
        channel_user_id=user_key,
        message_preview=message[:100] + ("..." if len(message) > 100 else ""),
    )
    set_task("web", user_key, "Processing your request...")

    logger.info("[STREAM] Starting stream: message=%r, channel_user_id=%s, history_len=%d", message[:80], channel_user_id, len(history))
    status_updates = []
    result_holder = []

    def on_status(text: str) -> None:
        set_task("web", user_key, text)
        status_updates.append(text)

    error_holder: list[Exception] = []

    def run() -> None:
        req_start = time.perf_counter()
        try:
            res = run_agent(
                user_message=message,
                history=history,
                channel="web",
                channel_user_id=user_key,
                on_status=on_status,
                trace_id=trace_id,
            )
            result_holder.append(res)
            article = res.get("article")
            msg = res.get("message", "")
            if article and article.get("google_doc_url"):
                msg = f"Here's your article: {article['google_doc_url']}"
            elif article and article.get("content"):
                msg = article["content"]
            total_ms = (time.perf_counter() - req_start) * 1000
            log_event(
                "request_done",
                total_latency_ms=round(total_ms, 2),
                has_article=article is not None,
                final_message_len=len(msg),
            )
            persist_trace(msg)
        except Exception as e:
            error_holder.append(e)

    thread = threading.Thread(target=run)
    thread.start()
    last_len = 0

    while thread.is_alive() or last_len < len(status_updates):
        while last_len < len(status_updates):
            yield f"data: {json.dumps({'type': 'status', 'text': status_updates[last_len]})}\n\n"
            last_len += 1
        time.sleep(0.2)

    if error_holder:
        from src.response_interpreter import explain_to_user
        friendly = explain_to_user(str(error_holder[0]))
        clear_task("web", user_key)
        done_payload = {"type": "done", "message": friendly, "article": None, "error": True}
        yield f"data: {json.dumps(done_payload)}\n\n"
        return

    result = result_holder[0] if result_holder else {}
    clear_task("web", user_key)

    article = result.get("article")
    message_out = result.get("message", "")
    if article and article.get("google_doc_url"):
        message_out = f"Here's your article: {article['google_doc_url']}"
    elif article and article.get("content"):
        message_out = article["content"]

    add_message_for_user("web", user_key, "user", message)
    add_message_for_user("web", user_key, "assistant", message_out)

    done_payload = {
        "type": "done",
        "message": message_out,
        "article": article,
    }
    logger.info("[STREAM] Completed: has_article=%s, message_len=%d", article is not None, len(message_out))
    yield f"data: {json.dumps(done_payload)}\n\n"


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Streaming chat endpoint - sends SSE events with status updates during article writing,
    then the final result. Use this to get progress updates (deep research, Tavily, PromptLayer).
    """
    from src.db import get_messages_for_user

    channel_user_id = req.channel_user_id or "default"
    msgs = get_messages_for_user("web", channel_user_id)
    history = [{"role": m["role"], "content": m["content"]} for m in msgs]

    return StreamingResponse(
        _stream_chat_generator(req.message, channel_user_id, history),
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
        throttle_sec = int(os.getenv("CRM_PROGRESS_THROTTLE_SEC", "10"))
        now = time.monotonic()
        key = ("whatsapp", chat_id)
        with _progress_lock:
            last = _progress_last_send.get(key, 0.0)
            if now - last >= throttle_sec:
                _progress_last_send[key] = now
                try:
                    send_message_chunked(chat_id, f"_{status_text}_", quoted_message_id=None)
                except Exception as e:
                    logger.warning("[WHATSAPP] Progress message send failed: %s", e)

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
        "tools": ["db", "google_docs", "web_search", "send_image", "write_article", "improve_article", "generate_images", "generate_hero_image", "approve_hero_image", "generate_infographic", "approve_infographic"],
        "sub_agents": ["Article Writer", "Article Improver", "Image Placement", "Infographic Analyzer", "Hero & Infographic Generator", "Generic Image Generator"],
    },
    {
        "id": "article_writer",
        "name": "Article Writer",
        "description": "Deep research + Tavily + PromptLayer. Invoked by write_article.",
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
    """Admin page: live graph state viewer."""
    return templates.TemplateResponse(
        "admin/graph_state.html",
        {"request": request},
    )


@app.get("/api/admin/graph-state")
async def api_get_graph_state(whatsapp_user_id: Optional[str] = None):
    """Deprecated: LangGraph removed. Returns threads from DB for reference."""
    from src.db import get_client
    try:
        r = get_client().table("active_article_threads").select("*").execute()
        threads = r.data or []
    except Exception:
        threads = []
    return {"deprecated": "LangGraph removed — agent is now stateless", "threads": threads}


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
