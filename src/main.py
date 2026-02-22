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
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

app = FastAPI(
    title="Article-Writing Chatbot",
    description="Chatbot that writes articles via deep research, PromptLayer, and more.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    """Process incoming WhatsApp message via LangGraph workflow (runs in background)."""
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

    from src.channels.whatsapp import send_message_chunked, send_image_by_url
    from src.db import get_messages_for_user, add_message_for_user
    from src.task_state import set_task, clear_task
    from src.graph.pre_router import classify_pre_intent
    from src.graph.graph import get_graph
    from src.graph.thread_manager import get_active_thread, get_or_create_thread_id
    from src.pipeline import status_callback as pipeline_status_callback

    set_task("whatsapp", chat_id, "Processing your request...")
    # Status callback: always set_task; optionally send as WhatsApp message (CRM check-back)
    def _on_status(status_text: str) -> None:
        set_task("whatsapp", chat_id, status_text)
        if os.getenv("CRM_SEND_PROGRESS_MESSAGES", "").lower() in ("1", "true", "yes"):
            throttle_sec = int(os.getenv("CRM_PROGRESS_THROTTLE_SEC", "45"))
            now = time.monotonic()
            key = ("whatsapp", chat_id)
            with _progress_lock:
                last = _progress_last_send.get(key, 0.0)
                if now - last >= throttle_sec:
                    _progress_last_send[key] = now
                    try:
                        send_message_chunked(chat_id, status_text, quoted_message_id=quoted_message_id)
                    except Exception as e:
                        logger.warning("[WHATSAPP] Progress message send failed: %s", e)

    status_token = pipeline_status_callback.set(_on_status)

    try:
        # 1. Pre-route: general question or article-related?
        # Check active thread first so we can pass stage context to the pre-router
        active = get_active_thread(chat_id)
        active_stage = None
        _active_graph_paused = False
        if active and active.get("thread_id"):
            try:
                _g = get_graph()
                _cfg = {"configurable": {"thread_id": active["thread_id"]}}
                _gs = _g.get_state(_cfg)
                if _gs and _gs.values:
                    active_stage = (_gs.values or {}).get("stage")
                if _gs and _gs.next:
                    _active_graph_paused = True
            except Exception:
                pass

        pre_intent = classify_pre_intent(text, active_stage=active_stage)
        logger.info("[WHATSAPP] Pre-intent: %s (active_stage=%s, paused=%s)", pre_intent, active_stage, _active_graph_paused)

        # If classified as general but there's an active thread paused at interrupt,
        # the message is almost certainly a response to that interrupt
        if pre_intent == "general_question" and _active_graph_paused:
            pre_intent = "article_intent"
            logger.info("[WHATSAPP] Override to article_intent (graph paused at interrupt)")

        # Load recent chat history for conversational context
        msgs = get_messages_for_user("whatsapp", chat_id)
        recent_messages = [{"role": m["role"], "content": m["content"]} for m in msgs[-20:]]

        if pre_intent in ("help", "list_articles"):
            # Handle directly via the graph with a dedicated thread
            graph = get_graph()
            thread_id = f"{chat_id}:meta"
            config = {"configurable": {"thread_id": thread_id}}
            result = graph.invoke(
                {
                    "user_message": text,
                    "whatsapp_user_id": chat_id,
                    "recent_messages": recent_messages,
                    "intent": pre_intent,
                },
                config,
            )
            response = result.get("response_to_user", "")
        elif pre_intent == "general_question":
            # Handle general question via graph (no article thread)
            graph = get_graph()
            thread_id = f"{chat_id}:general:{trace_id}"
            config = {"configurable": {"thread_id": thread_id}}
            result = graph.invoke(
                {
                    "user_message": text,
                    "whatsapp_user_id": chat_id,
                    "recent_messages": recent_messages,
                },
                config,
            )
            response = result.get("response_to_user", "")
        else:
            # Article-related intent -> use article thread
            graph = get_graph()
            thread_id = get_or_create_thread_id(chat_id, text)
            config = {"configurable": {"thread_id": thread_id}}

            # Check if graph is paused at an interrupt
            try:
                graph_state = graph.get_state(config)
            except Exception:
                graph_state = None

            if graph_state and graph_state.next:
                # Resume from interrupt with user's message
                from langgraph.types import Command
                logger.info("[WHATSAPP] Resuming graph from interrupt, next=%s", graph_state.next)

                # Before resuming, send any pending image previews
                current_state = graph_state.values or {}
                _send_pending_images(chat_id, current_state, quoted_message_id)

                result = graph.invoke(
                    Command(resume=text),
                    config,
                )
            else:
                # Fresh invocation (CRM: confirmation_gate interrupts with "I understood..." if start_article)
                result = graph.invoke(
                    {
                        "user_message": text,
                        "whatsapp_user_id": chat_id,
                        "recent_messages": recent_messages,
                    },
                    config,
                )

            # Extract message: prefer interrupt payload (confirmation, draft approval, etc.), else response_to_user
            response = ""
            interrupts = (result.get("__interrupt__") or []) if isinstance(result, dict) else []
            first_interrupt_payload = None
            if interrupts:
                first = interrupts[0]
                payload = getattr(first, "value", first) if not isinstance(first, dict) else first
                first_interrupt_payload = payload if isinstance(payload, dict) else None
                if first_interrupt_payload and first_interrupt_payload.get("message"):
                    response = first_interrupt_payload["message"]
            if not response and isinstance(result, dict):
                response = result.get("response_to_user", "")

            # Send any image previews: prefer interrupt payload (hero/infographic approval), else state
            _send_pending_images(
                chat_id,
                result if isinstance(result, dict) else {},
                quoted_message_id,
                interrupt_payload=first_interrupt_payload,
            )

    except Exception as e:
        logger.exception("LangGraph error for WhatsApp %s: %s", chat_id, e)
        add_message_for_user("whatsapp", chat_id, "user", text)
        from src.response_interpreter import explain_to_user
        friendly = explain_to_user(str(e))
        add_message_for_user("whatsapp", chat_id, "assistant", friendly)
        _log_whatsapp(f"[WhatsApp] OUT {chat_id} (error->interpreted): {repr(friendly)}")
        try:
            send_message_chunked(
                chat_id, friendly, quoted_message_id=quoted_message_id
            )
        except Exception as send_err:
            logger.exception("Failed to send error reply: %s", send_err)
        return
    finally:
        pipeline_status_callback.reset(status_token)
        clear_task("whatsapp", chat_id)
        with _progress_lock:
            _progress_last_send.pop(("whatsapp", chat_id), None)

    logger.info("[WHATSAPP] LangGraph done: response_len=%d", len(response))
    out_preview = response[:200] + ("..." if len(response) > 200 else "")
    _log_whatsapp(f"[WhatsApp] OUT {chat_id}: {repr(out_preview)}")
    add_message_for_user("whatsapp", chat_id, "user", text)
    add_message_for_user("whatsapp", chat_id, "assistant", response)

    total_ms = (time.perf_counter() - whatsapp_start) * 1000
    log_event(
        "request_done",
        total_latency_ms=round(total_ms, 2),
        final_message_len=len(response),
    )
    persist_trace(response)

    try:
        if response:
            send_message_chunked(
                chat_id, response, quoted_message_id=quoted_message_id
            )
    except Exception as e:
        logger.exception("Failed to send WhatsApp reply to %s: %s", chat_id, e)


def _send_pending_images(
    chat_id: str,
    state: dict,
    quoted_message_id: Optional[str] = None,
    interrupt_payload: Optional[dict] = None,
) -> None:
    """Send hero/infographic image previews to WhatsApp if present in state or interrupt payload."""
    from src.channels.whatsapp import send_image_by_url

    # Prefer image_url from interrupt payload, fallback to state (LangGraph may not include url in payload)
    if interrupt_payload:
        itype = interrupt_payload.get("type")
        url = interrupt_payload.get("image_url")
        if not url and itype == "hero_approval":
            url = state.get("hero_image_url")
        if not url and itype == "infographic_approval":
            url = state.get("infographic_image_url")
        if itype in ("hero_approval", "infographic_approval") and not url:
            logger.warning(
                "[WHATSAPP] No image_url for %s: payload=%s, state_keys=%s",
                itype,
                {k: str(v)[:80] for k, v in interrupt_payload.items()},
                list(state.keys())[:20],
            )
        if url and itype == "hero_approval":
            try:
                send_image_by_url(chat_id, url, caption="Hero image preview")
                logger.info("[WHATSAPP] Sent hero preview (from interrupt): %s", url[:60])
            except Exception as e:
                logger.warning("[WHATSAPP] Failed to send hero preview: %s", e)
            return  # Sent hero, skip state-based check
        if url and itype == "infographic_approval":
            try:
                send_image_by_url(chat_id, url, caption="Infographic preview")
                logger.info("[WHATSAPP] Sent infographic preview (from interrupt): %s", url[:60])
            except Exception as e:
                logger.warning("[WHATSAPP] Failed to send infographic preview: %s", e)
            return

    # Fallback: check state (for edge cases where interrupt payload isn't used)
    for url_key, caption in [
        ("hero_image_url", "Hero image preview"),
        ("infographic_image_url", "Infographic preview"),
    ]:
        url = state.get(url_key)
        if url and state.get("stage", "").startswith("awaiting_"):
            try:
                send_image_by_url(chat_id, url, caption=caption)
                logger.info("[WHATSAPP] Sent %s preview: %s", url_key, url[:60])
            except Exception as e:
                logger.warning("[WHATSAPP] Failed to send %s preview: %s", url_key, e)


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
    ALLOWED_CHAT_ID = "972546678582@c.us"
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


@app.get("/api/last-deep-research")
async def api_last_deep_research():
    """API: get the most recent deep research (topic, content, timestamp). Available even when file save is disabled."""
    from src.pipeline import get_last_deep_research

    return get_last_deep_research()


@app.get("/admin/last-deep-research")
async def admin_last_deep_research(request: Request):
    """Admin page: view the most recent deep research."""
    from src.pipeline import get_last_deep_research

    data = get_last_deep_research()
    return templates.TemplateResponse(
        "admin/last_deep_research.html",
        {"request": request, "topic": data.get("topic", ""), "content": data.get("content", ""), "at": data.get("at", "")},
    )


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
    """Get current graph state for a user's active thread."""
    try:
        from src.graph.graph import get_graph
        from src.graph.thread_manager import get_active_thread

        if not whatsapp_user_id:
            from src.db import get_client
            try:
                r = get_client().table("active_article_threads").select("*").execute()
                threads = r.data or []
            except Exception:
                threads = []
            return {"threads": threads}

        active = get_active_thread(whatsapp_user_id)
        if not active or not active.get("thread_id"):
            return {"error": "No active thread", "thread": None, "state": None}

        # Sanitize thread dict for JSON (Supabase may return datetime etc.)
        thread_safe = {k: str(v) if hasattr(v, "isoformat") else v for k, v in active.items()}

        thread_id = active["thread_id"]
        graph = get_graph()
        config = {"configurable": {"thread_id": thread_id}}

        try:
            gs = graph.get_state(config)
        except Exception as e:
            return {"error": str(e), "thread": thread_safe, "state": None, "next": None}

        if not gs:
            return {"thread": thread_safe, "state": None, "next": None}

        # Serialize state values (strip large fields, ensure JSON-safe)
        def _to_jsonable(v):
            if hasattr(v, "isoformat"):
                return str(v)
            if isinstance(v, dict):
                return {k: _to_jsonable(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [_to_jsonable(x) for x in v]
            if isinstance(v, (str, int, float, bool, type(None))):
                return v
            return str(v)

        values = dict(gs.values) if gs.values else {}
        for k in ("research_text", "recent_messages", "response_to_user"):
            if k in values and isinstance(values[k], str) and len(values[k]) > 500:
                values[k] = values[k][:500] + f"... [{len(values[k])} chars]"
        if "recent_messages" in values and isinstance(values["recent_messages"], list):
            values["recent_messages"] = f"[{len(values['recent_messages'])} messages]"
        if "actions_log" in values and isinstance(values["actions_log"], list):
            values["actions_log"] = values["actions_log"][-10:]
        values = _to_jsonable(values)

        return {
            "thread": thread_safe,
            "state": values,
            "next": list(gs.next) if gs.next else [],
            "created_at": str(gs.created_at) if hasattr(gs, "created_at") and gs.created_at else None,
        }
    except Exception as e:
        logger.exception("api_get_graph_state failed: %s", e)
        return {"error": str(e), "thread": None, "state": None, "next": None, "threads": []}


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
