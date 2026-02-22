"""Central observability module for agent traces, tool calls, and LLM spans.

Provides:
- trace_id (contextvars) for request-scoped identification
- log_event for structured JSON logs
- observe_tool, observe_agent_call, observe_sub_agent for instrumentation
- calculate_cost for USD cost estimation from model + tokens
- compute_trace_summary for per-trace aggregation (tokens, cost, steps, duration)
"""

import json
import logging
import os
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model pricing — (input $/1M tokens, output $/1M tokens)
# ---------------------------------------------------------------------------

MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-4-20250514": (15.00, 75.00),
    "claude-opus-4-5-20251101": (5.00, 25.00),
    "claude-opus-4-1-20250805": (15.00, 75.00),
    "claude-3-opus-20240229": (15.00, 75.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-4-5-sonnet": (3.00, 15.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "o3-mini": (1.10, 4.40),
    "o3": (10.00, 40.00),
    # Google
    "gemini-3-pro-image-preview": (0.25, 1.00),
    "gemini-2.5-flash-image": (0.15, 0.60),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-flash": (0.075, 0.30),
}

# Per-image generation cost (approximate)
IMAGE_MODEL_PRICING: dict[str, float] = {
    "imagen-3.0-generate-002": 0.04,
    "dall-e-3": 0.04,
    "gemini-2.5-flash-image": 0.04,
    "gemini-3-pro-image-preview": 0.06,
}


def calculate_cost(model: str, input_tokens: int = 0, output_tokens: int = 0) -> float:
    """Calculate estimated cost in USD for a model call."""
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return 0.0
    inp_price, out_price = pricing
    return (input_tokens * inp_price / 1_000_000) + (output_tokens * out_price / 1_000_000)


def compute_trace_summary(payload: dict) -> dict:
    """Compute aggregate stats for a trace payload.

    Returns dict with: steps, total_input_tokens, total_output_tokens,
    total_tokens, total_cost_usd, duration_ms, models_used, tools_used.
    """
    events = payload.get("events", [])
    total_input = 0
    total_output = 0
    total_cost = 0.0
    steps = 0
    duration_ms = 0
    models_used: set[str] = set()
    tools_used: list[str] = []

    for ev in events:
        etype = ev.get("event_type")

        if etype in ("agent_call", "tool_result", "sub_agent"):
            steps += 1

        if etype == "agent_call":
            tokens = ev.get("tokens") or {}
            inp = tokens.get("input", 0)
            out = tokens.get("output", 0)
            total_input += inp
            total_output += out
            model = ev.get("model", "")
            if model:
                models_used.add(model)
            total_cost += calculate_cost(model, inp, out)

        elif etype == "tool_result":
            name = ev.get("tool_name", "")
            if name:
                tools_used.append(name)

        elif etype == "sub_agent":
            model = ev.get("model", "")
            if model:
                models_used.add(model)
            # If sub_agent has token info (added optionally), include it
            tokens = ev.get("tokens") or {}
            if tokens:
                inp = tokens.get("input", 0)
                out = tokens.get("output", 0)
                total_input += inp
                total_output += out
                total_cost += calculate_cost(model, inp, out)

        if etype == "request_done" and ev.get("total_latency_ms") is not None:
            duration_ms = ev["total_latency_ms"]

    # Fallback: sum up individual latencies if no request_done event
    if not duration_ms:
        for ev in events:
            if ev.get("event_type") in ("agent_call", "tool_result", "sub_agent"):
                duration_ms += ev.get("latency_ms") or ev.get("span", {}).get("duration_ms", 0)

    return {
        "steps": steps,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "total_cost_usd": round(total_cost, 6),
        "duration_ms": round(duration_ms, 1),
        "models_used": sorted(models_used),
        "tools_used": tools_used,
    }

# ---------------------------------------------------------------------------
# Trace context
# ---------------------------------------------------------------------------

_trace_id_var: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
_trace_payload_var: ContextVar[Optional[dict]] = ContextVar("trace_payload", default=None)


def set_trace_id(trace_id: Optional[str] = None) -> Optional[str]:
    """Set trace_id for the current context. Returns previous token for reset."""
    tid = trace_id or str(uuid.uuid4())
    token = _trace_id_var.set(tid)
    return token


def get_trace_id() -> Optional[str]:
    """Get current trace_id from context."""
    return _trace_id_var.get()


def reset_trace_id(token) -> None:
    """Reset trace_id to previous value (use token from set_trace_id)."""
    if token is not None:
        _trace_id_var.reset(token)


# ---------------------------------------------------------------------------
# Trace payload accumulator (for hierarchical storage)
# ---------------------------------------------------------------------------

def init_trace_payload(
    user_message: str,
    channel: Optional[str] = None,
    channel_user_id: Optional[str] = None,
) -> dict:
    """Initialize the trace payload for the current request."""
    payload = {
        "user_message": user_message,
        "channel": channel or "unknown",
        "channel_user_id": channel_user_id or "unknown",
        "root": {"type": "agent_run", "children": []},
        "events": [],
        "start_ts": datetime.now(timezone.utc).isoformat(),
    }
    token = _trace_payload_var.set(payload)
    return payload


def get_trace_payload() -> Optional[dict]:
    """Get current trace payload from context."""
    return _trace_payload_var.get()


def reset_trace_payload(token) -> None:
    """Reset trace payload (use token from init)."""
    if token is not None:
        _trace_payload_var.reset(token)


def _get_or_create_payload(user_message: str = "", channel: str = "", channel_user_id: str = "") -> dict:
    """Get existing payload or create a minimal one."""
    p = get_trace_payload()
    if p is not None:
        return p
    return init_trace_payload(user_message, channel, channel_user_id)


# ---------------------------------------------------------------------------
# Event logging
# ---------------------------------------------------------------------------

OBSERVABILITY_LEVEL = os.getenv("OBSERVABILITY_LEVEL", "default")
OBSERVABILITY_SAVE_PROMPTS = os.getenv("OBSERVABILITY_SAVE_PROMPTS", "false").lower() in ("1", "true", "yes")


def _truncate(obj: Any, max_len: int = 500) -> Any:
    """Truncate strings for safe logging."""
    if isinstance(obj, str):
        return obj[:max_len] + ("..." if len(obj) > max_len else "")
    if isinstance(obj, dict):
        return {k: _truncate(v, max_len // 2) for k, v in list(obj.items())[:20]}
    if isinstance(obj, (list, tuple)):
        return [_truncate(x, max_len // 2) for x in list(obj)[:10]]
    return obj


def _sanitize_args(args: dict) -> dict:
    """Sanitize tool args for logging (truncate, redact sensitive keys)."""
    sensitive = {"api_key", "token", "password", "secret"}
    out = {}
    for k, v in args.items():
        if any(s in k.lower() for s in sensitive):
            out[k] = "[REDACTED]"
        else:
            out[k] = _truncate(v, 200) if isinstance(v, str) else v
    return out


def log_event(event_type: str, **kwargs) -> None:
    """Emit a structured event. Appends to trace payload and logs to stderr."""
    trace_id = get_trace_id()
    event = {
        "event_type": event_type,
        "trace_id": trace_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        **kwargs,
    }
    payload = get_trace_payload()
    if payload and "events" in payload:
        payload["events"].append(event)
    if OBSERVABILITY_LEVEL in ("verbose", "debug"):
        logger.debug("[OBS] %s", json.dumps(event, default=str)[:500])


# ---------------------------------------------------------------------------
# observe_tool
# ---------------------------------------------------------------------------

def observe_tool(
    name: str,
    args: dict,
    result: Any,
    latency_ms: float,
    error: Optional[str] = None,
) -> None:
    """Log a tool call with args, result preview, and latency."""
    args_sanitized = _sanitize_args(args)
    _result_max = 1000 if OBSERVABILITY_SAVE_PROMPTS else 300
    result_preview = _truncate(str(result), _result_max) if result is not None else None
    log_event(
        "tool_result",
        tool_name=name,
        args_sanitized=args_sanitized,
        result_preview=result_preview,
        latency_ms=round(latency_ms, 2),
        success=error is None,
        error=error,
    )
    logger.info(
        "[OBS] tool_result name=%s latency_ms=%.0f success=%s",
        name, latency_ms, error is None,
    )


# ---------------------------------------------------------------------------
# observe_agent_call (Phase 2 - stub for now)
# ---------------------------------------------------------------------------

def observe_agent_call(
    name: str,
    provider: str,
    model: str,
    prompt: dict,
    response: dict,
    tokens: Optional[dict],
    span: dict,
    parent_span_id: Optional[str] = None,
) -> None:
    """Log an LLM span with full metadata."""
    span_id = str(uuid.uuid4())
    event = {
        "event_type": "agent_call",
        "trace_id": get_trace_id(),
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "name": name,
        "provider": provider,
        "model": model,
        "prompt": prompt,
        "response": response,
        "tokens": tokens,
        "span": span,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    payload = get_trace_payload()
    if payload and "events" in payload:
        payload["events"].append(event)
    if OBSERVABILITY_LEVEL in ("verbose", "debug"):
        logger.debug("[OBS] agent_call name=%s model=%s duration_ms=%s", name, model, span.get("duration_ms"))


# ---------------------------------------------------------------------------
# observe_sub_agent
# ---------------------------------------------------------------------------

def observe_sub_agent(
    name: str,
    input_keys: list[str],
    output_size: int,
    latency_ms: float,
    status: str = "success",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    tokens: Optional[dict] = None,
) -> None:
    """Log a sub-agent/pipeline step.

    Optional *model* and *tokens* params allow richer cost attribution.
    """
    extra: dict[str, Any] = {}
    if model:
        extra["model"] = model
    if tokens:
        extra["tokens"] = tokens
    log_event(
        "sub_agent",
        name=name,
        input_keys=input_keys,
        output_size=output_size,
        latency_ms=round(latency_ms, 2),
        status=status,
        provider=provider,
        **extra,
    )
    logger.info("[OBS] sub_agent name=%s latency_ms=%.0f status=%s", name, latency_ms, status)


# ---------------------------------------------------------------------------
# persist_trace (Phase 3 - save to Supabase)
# ---------------------------------------------------------------------------

def persist_trace(final_message: Optional[str] = None) -> Optional[str]:
    """Persist the current trace payload to Supabase. Returns trace_id or None.

    Computes a ``summary`` dict (tokens, cost, duration, models) and
    optionally generates a narrator summary if NARRATOR_AGENT_ENABLED=true.
    """
    trace_id = get_trace_id()
    payload = get_trace_payload()
    if not trace_id or not payload:
        return None
    payload["end_ts"] = datetime.now(timezone.utc).isoformat()
    if final_message is not None:
        payload["final_message"] = final_message

    # Compute aggregate summary ------------------------------------------
    payload["summary"] = compute_trace_summary(payload)

    # Narrator agent (optional) ------------------------------------------
    try:
        from src.narrator import narrate_trace
        narrative = narrate_trace(
            events=payload.get("events", []),
            user_message=payload.get("user_message", ""),
            final_message=payload.get("final_message", ""),
        )
        if narrative:
            payload["narrator_summary"] = narrative
    except Exception as exc:
        logger.debug("[OBS] Narrator skipped: %s", exc)

    # Persist to Supabase ------------------------------------------------
    try:
        from src.db import get_client
        client = get_client()
        client.table("observability_traces").upsert({
            "trace_id": trace_id,
            "channel": payload.get("channel", "unknown"),
            "channel_user_id": payload.get("channel_user_id", "unknown"),
            "user_message": payload.get("user_message", ""),
            "final_message": payload.get("final_message"),
            "payload": payload,
        }, on_conflict="trace_id").execute()
        logger.info("[OBS] Persisted trace %s (cost=$%.4f)", trace_id[:8], payload["summary"]["total_cost_usd"])
        return trace_id
    except Exception as e:
        logger.warning("[OBS] Failed to persist trace: %s", e)
        return None
