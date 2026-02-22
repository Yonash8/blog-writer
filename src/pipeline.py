"""Core pipeline: Deep research + Tavily enrichment + PromptLayer SEO agent."""

import json
import logging
import os
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

from src.config import get_config
from src.observability import observe_sub_agent
from src.prompts_loader import get_prompt

load_dotenv()

logger = logging.getLogger(__name__)

# In-memory cache of last deep research (available even when file save is disabled)
_last_deep_research: dict[str, Any] = {"topic": "", "content": "", "at": None}


def get_last_deep_research() -> dict[str, Any]:
    """Return the most recent deep research (topic, content, timestamp). Empty if none run yet."""
    return dict(_last_deep_research)

# Status callback for progress updates (set by run_agent before invoking tools)
status_callback: ContextVar[Optional[Callable[[str], None]]] = ContextVar(
    "status_callback", default=None
)

# Cancel check callback: returns True if the current task should be cancelled
cancel_check: ContextVar[Optional[Callable[[], bool]]] = ContextVar(
    "cancel_check", default=None
)


class TaskCancelledError(Exception):
    """Raised when a long-running pipeline task is cancelled by the user."""
    pass


def check_cancelled() -> None:
    """Check if the current task has been cancelled; raise TaskCancelledError if so."""
    fn = cancel_check.get()
    if fn and fn():
        logger.info("[PIPELINE] Task cancelled by user")
        raise TaskCancelledError("Task cancelled by user")


def emit_status(text: str) -> None:
    """Emit a status update to the callback if set."""
    cb = status_callback.get()
    if cb:
        try:
            cb(text)
        except Exception as e:
            logger.warning("Status callback error: %s", e)


def _run_deep_research_perplexity(topic: str, model: str = "sonar-pro") -> str:
    """Perplexity Sonar for deep research."""
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise ValueError("PERPLEXITY_API_KEY not set")
    prompt = get_prompt("deep_research").format(topic=topic)
    prompt += "\n\nIMPORTANT: Produce a thorough report of at least 2500-4000 words. Include detailed sections with multiple paragraphs each. Be exhaustive—cover every relevant angle. Do not summarize briefly."
    emit_status("Using Perplexity for research...")
    models_to_try = [model] if model in ("sonar-pro", "sonar") else ["sonar-pro", "sonar"]
    for try_model in models_to_try:
        check_cancelled()  # Bail out if user requested cancellation
        logger.info("[PIPELINE] Deep research: using Perplexity %s", try_model)
        try:
            # Cranked params for robust research: high max_tokens, return_citations, wide recency
            payload = {
                "model": try_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 32768 if try_model == "sonar-pro" else 16384,
                "temperature": 0.1,
                "return_citations": True,
                "return_related_questions": True,
                "search_recency_filter": "year",
            }
            if try_model == "sonar-pro":
                payload["web_search_options"] = {"search_context_size": "high"}
            with httpx.Client() as client:
                response = client.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=300.0,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                logger.info("[PIPELINE] Perplexity %s DONE: output_len=%d chars", try_model, len(content))
                return content
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (400, 402, 404) and try_model == "sonar-pro":
                logger.warning("[PIPELINE] sonar-pro failed (%s), trying sonar", e)
                continue
            raise
    raise RuntimeError("Perplexity fallback failed")


def run_deep_research(topic: str, max_tool_calls: int = 50) -> str:
    """Run deep research on the topic. Uses configured model (o3-deep-research, sonar-pro, sonar)."""
    model = get_config("deep_research_model", "o3-deep-research")
    logger.info("[PIPELINE] Deep research: topic=%r, model=%s", topic, model)
    start = time.perf_counter()

    if model in ("sonar-pro", "sonar"):
        result = _run_deep_research_perplexity(topic, model=model)
        latency_ms = (time.perf_counter() - start) * 1000
        observe_sub_agent(name="deep_research", input_keys=["topic"], output_size=len(result), latency_ms=latency_ms, status="success", provider="perplexity")
        _last_deep_research.update({"topic": topic, "content": result, "at": datetime.now(timezone.utc).isoformat()})
        return result

    try:
        result = _run_deep_research_openai(topic, max_tool_calls)
        latency_ms = (time.perf_counter() - start) * 1000
        observe_sub_agent(name="deep_research", input_keys=["topic"], output_size=len(result), latency_ms=latency_ms, status="success", provider="openai")
        _last_deep_research.update({"topic": topic, "content": result, "at": datetime.now(timezone.utc).isoformat()})
        return result
    except Exception as e:
        if os.getenv("PERPLEXITY_API_KEY"):
            logger.warning("[PIPELINE] OpenAI deep research failed (%s), falling back to Perplexity", e)
            result = _run_deep_research_perplexity(topic)
            latency_ms = (time.perf_counter() - start) * 1000
            observe_sub_agent(name="deep_research", input_keys=["topic"], output_size=len(result), latency_ms=latency_ms, status="success", provider="perplexity")
            _last_deep_research.update({"topic": topic, "content": result, "at": datetime.now(timezone.utc).isoformat()})
            return result
        latency_ms = (time.perf_counter() - start) * 1000
        observe_sub_agent(name="deep_research", input_keys=["topic"], output_size=0, latency_ms=latency_ms, status="failure", provider="openai")
        raise


def _run_deep_research_openai(topic: str, max_tool_calls: int) -> str:
    """OpenAI o3-deep-research via Responses API (background mode with polling)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("[PIPELINE] OPENAI_API_KEY not set")
        raise ValueError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key, timeout=3600)
    prompt = get_prompt("deep_research").format(topic=topic)
    logger.info("[PIPELINE] Deep research: calling o3-deep-research (background=True)")
    emit_status("Running deep research via o3...")

    response = client.responses.create(
        model="o3-deep-research",
        input=prompt,
        tools=[{"type": "web_search_preview"}],
        background=True,
        max_tool_calls=max_tool_calls,
    )
    logger.info(
        "[PIPELINE] Deep research created: id=%s, status=%s, created_at=%s, model=%s, max_tool_calls=%d",
        response.id, getattr(response, "status", "n/a"), getattr(response, "created_at", "n/a"),
        getattr(response, "model", "n/a"), max_tool_calls,
    )

    # Poll until the background task completes; emit progress every 4 min
    poll_start = time.monotonic()
    last_emit_min = 0
    poll_count = 0
    include_meta = ["web_search_call.results", "web_search_call.action.sources"]
    while response.status in ("queued", "in_progress"):
        check_cancelled()  # Bail out if user requested cancellation
        poll_count += 1
        elapsed = int(time.monotonic() - poll_start)
        # Extract all available metadata for logging
        output_items = getattr(response, "output", []) or []
        web_search_count = sum(1 for o in output_items if getattr(o, "type", None) == "web_search_call")
        message_count = sum(1 for o in output_items if getattr(o, "type", None) == "message")
        created_at = getattr(response, "created_at", None)
        model = getattr(response, "model", None)
        usage = getattr(response, "usage", None)
        usage_str = ""
        if usage is not None:
            inp = getattr(usage, "input_tokens", None)
            out = getattr(usage, "output_tokens", None)
            total = getattr(usage, "total_tokens", None)
            if inp is not None or out is not None or total is not None:
                usage_str = ", usage=%d in/%d out/%d total" % (inp or 0, out or 0, total or 0)
        logger.info(
            "[PIPELINE] Deep research polling #%d: status=%s, id=%s, elapsed=%ds, model=%s, created_at=%s, output_items=%d (web_search=%d, message=%d)%s",
            poll_count, response.status, response.id, elapsed, model, created_at,
            len(output_items), web_search_count, message_count, usage_str,
        )
        elapsed_min = int((time.monotonic() - poll_start) / 60)
        if elapsed_min > 0 and elapsed_min % 4 == 0 and elapsed_min != last_emit_min:
            last_emit_min = elapsed_min
            emit_status(f"Deep research still in progress ({elapsed_min} min)...")
        time.sleep(15)
        response = client.responses.retrieve(response.id, include=include_meta)

    if response.status != "completed":
        # Extract error details from the API response
        error_detail = "no error field"
        err = getattr(response, "error", None)
        if err is not None:
            if hasattr(err, "message"):
                error_detail = err.message
            elif hasattr(err, "code"):
                error_detail = f"code={err.code}, message={getattr(err, 'message', '')}"
            elif isinstance(err, dict):
                error_detail = str(err)
            else:
                error_detail = str(err)
        incomplete = getattr(response, "incomplete_details", None)
        if incomplete is not None:
            reason = getattr(incomplete, "reason", None) or (incomplete.get("reason") if isinstance(incomplete, dict) else None)
            if reason:
                error_detail = f"{error_detail}; incomplete_details.reason={reason}"
        logger.error(
            "[PIPELINE] Deep research FAILED: status=%s, id=%s, error=%s",
            response.status,
            response.id,
            error_detail,
        )
        raise RuntimeError(
            f"Deep research failed with status '{response.status}' (id={response.id}): {error_detail}"
        )

    research_len = len(response.output_text) if response.output_text else 0
    # Save last deep research to file for inspection (path from env or default)
    save_path = os.getenv("DEEP_RESEARCH_SAVE_PATH", "last_deep_research.md")
    if save_path and response.output_text:
        try:
            abs_path = os.path.abspath(save_path)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(response.output_text)
            logger.info("[PIPELINE] Deep research saved to %s", abs_path)
        except OSError as e:
            logger.warning("[PIPELINE] Could not save deep research to %s: %s", save_path, e)
    # Log all completion metadata
    output_items = getattr(response, "output", []) or []
    web_search_count = sum(1 for o in output_items if getattr(o, "type", None) == "web_search_call")
    message_count = sum(1 for o in output_items if getattr(o, "type", None) == "message")
    usage = getattr(response, "usage", None)
    usage_log = ""
    if usage is not None:
        inp = getattr(usage, "input_tokens", None)
        out = getattr(usage, "output_tokens", None)
        total = getattr(usage, "total_tokens", None)
        cached = getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", None)
        reasoning = getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", None)
        usage_log = ", usage: input=%d, output=%d, total=%d, cached=%s, reasoning=%s" % (
            inp or 0, out or 0, total or 0, cached if cached is not None else "n/a", reasoning if reasoning is not None else "n/a"
        )
    logger.info(
        "[PIPELINE] Deep research DONE: id=%s, output_len=%d chars, output_items=%d (web_search=%d, message=%d), created_at=%s, model=%s%s",
        response.id, research_len, len(output_items), web_search_count, message_count,
        getattr(response, "created_at", "n/a"), getattr(response, "model", "n/a"), usage_log,
    )
    return response.output_text


def run_tavily_enrichment(topic: str, max_results: int = 20) -> list[dict[str, Any]]:
    """Run Tavily search for enrichment on the topic."""
    logger.info("[PIPELINE] Tavily enrichment: topic=%r, max_results=%d", topic, max_results)
    emit_status("Getting links from Tavily...")
    start = time.perf_counter()
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.error("[PIPELINE] TAVILY_API_KEY not set")
        raise ValueError("TAVILY_API_KEY not set")

    client = TavilyClient(api_key=api_key)
    query = f"Find authoritative and relevant sources explaining {topic}. Prefer high-quality articles, engineering blogs, academic papers, postmortems, and official documentation."
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        chunks_per_source=1,
        include_raw_content=False,
        include_answer=False,
    )
    results = []
    for r in response.get("results", []):
        results.append({
            "url": r.get("url", ""),
            "title": r.get("title", ""),
            "content": r.get("content", ""),
            "raw_content": "",  # links-only mode: no full page content (avoids PromptLayer token overflow)
            "score": r.get("score", 0),
        })
    latency_ms = (time.perf_counter() - start) * 1000
    output_size = sum(len(str(r)) for r in results)
    observe_sub_agent(
        name="tavily",
        input_keys=["topic", "max_results"],
        output_size=output_size,
        latency_ms=latency_ms,
        status="success",
        provider="tavily",
    )
    logger.info("[PIPELINE] Tavily DONE: got %d results (links-only, no raw content)", len(results))
    return results


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text. Uses tiktoken for accuracy, falls back to ~4 chars/token.
    """
    if not text:
        return 0
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


def format_tavily_for_research(results: list[dict[str, Any]]) -> str:
    """
    Format Tavily results for the SEO agent. Links-only (title + URL) to keep token count
    low for PromptLayer.
    """
    # Sort by score descending (higher first)
    sorted_results = sorted(
        results,
        key=lambda r: r.get("score", 0),
        reverse=True,
    )

    parts = []
    for r in sorted_results:
        url = r.get("url", "")
        if not url:
            continue
        title = r.get("title", "Source")
        formatted = f"- [{title}]({url})"
        parts.append(formatted)

    return "\n".join(parts) if parts else ""


def truncate_research_for_promptlayer(research: str, max_tokens: Optional[int] = None) -> str:
    """
    Truncate research to fit within PromptLayer context. Uses estimate_tokens.
    If max_tokens not set, reads from PROMPTLAYER_RESEARCH_MAX_TOKENS env (default 30000).
    Note: Research may be injected multiple times in the workflow (llm_deep_research, research, etc.);
    use a conservative limit for 128k context models.
    """
    if not research:
        return research
    limit = max_tokens
    if limit is None:
        try:
            limit = int(os.getenv("PROMPTLAYER_RESEARCH_MAX_TOKENS", "30000"))
        except ValueError:
            limit = 30000
    if limit <= 0:
        return research
    tokens = estimate_tokens(research)
    if tokens <= limit:
        return research
    # Binary search by char position to land near limit
    approx_chars_per_token = len(research) / max(tokens, 1)
    target_chars = int(limit * approx_chars_per_token * 0.95)
    truncated = research[:target_chars]
    while estimate_tokens(truncated) > limit and len(truncated) > 100:
        truncated = truncated[: int(len(truncated) * 0.9)]
    suffix = "\n\n[…] truncated to fit PromptLayer context limit"
    logger.info(
        "[PIPELINE] Research truncated from %d to ~%d tokens (limit=%d)",
        tokens,
        estimate_tokens(truncated + suffix),
        limit,
    )
    return truncated.rstrip() + suffix


def _parse_token_overflow_for_retry(err_msg: str) -> tuple[int, int] | None:
    """
    Extract (got_tokens, limit_tokens) from a token overflow error.
    Returns None if not a token error. Order: "limit is 128000... resulted in 206147" -> (206147, 128000).
    """
    s = (err_msg or "").lower()
    if not any(kw in s for kw in ("token", "context length", "maximum context", "reduce the length")):
        return None
    import re
    nums = [int(x) for x in re.findall(r"\d{5,}", err_msg)]
    if len(nums) < 2:
        return None
    limit_tokens = nums[0]
    got_tokens = nums[1]
    return (got_tokens, limit_tokens)


def _parse_promptlayer_error_for_user(err_msg: str) -> dict[str, str] | None:
    """
    If the error is a PromptLayer context/token overflow, return a user-friendly
    explanation and plan. Otherwise return None.
    """
    s = (err_msg or "").lower()
    if not any(kw in s for kw in ("token", "context length", "context length", "maximum context", "reduce the length")):
        return None
    # Extract numbers: typically "maximum context length is 128000... resulted in 206147"
    import re
    nums = [int(x) for x in re.findall(r"\d{5,}", err_msg)]
    limit_tokens = nums[0] if nums else 128000
    got_tokens = nums[1] if len(nums) >= 2 else nums[0] if nums else 0
    suggested = min(30000, max(15000, limit_tokens // 4))
    return {
        "error": f"PromptLayer hit the model's context limit ({limit_tokens:,} tokens). The deep research + Tavily content was too long.",
        "plan": f"Chunk down the deep research: set PROMPTLAYER_RESEARCH_MAX_TOKENS={suggested} in your .env, restart the app, and try again with the same topic. The system will truncate research automatically.",
        "retry_hint": "Add PROMPTLAYER_RESEARCH_MAX_TOKENS=25000 to .env and try again. Or try a narrower topic.",
    }


def _truncate_for_log(obj: Any, max_str: int = 400) -> str:
    """Serialize for debug logs, truncating long strings."""

    def _trunc(o: Any, m: int) -> Any:
        if isinstance(o, str):
            return o[:m] + ("..." if len(o) > m else "")
        if isinstance(o, dict):
            return {k: _trunc(v, m) for k, v in list(o.items())[:15]}
        if isinstance(o, (list, tuple)):
            return [_trunc(x, m) for x in list(o)[:10]]
        return o

    try:
        s = json.dumps(_trunc(obj, max_str), default=str)
        return s[:1200] + ("..." if len(s) > 1200 else "")
    except Exception:
        return str(obj)[:500]


def run_promptlayer_agent(topic: str, research: str) -> str:
    """Invoke PromptLayer SEO agent with topic and research. Auto-retries with chunked research on token overflow."""
    logger.info("[PIPELINE] PromptLayer: topic=%r, research_len=%d chars", topic, len(research))
    emit_status("Writing article with PromptLayer...")
    start = time.perf_counter()
    api_key = os.getenv("PROMPTLAYER_API_KEY")
    workflow_name = os.getenv("PROMPTLAYER_WORKFLOW_NAME", "seo agent")
    if not api_key:
        logger.error("[PIPELINE] PROMPTLAYER_API_KEY not set")
        raise ValueError("PROMPTLAYER_API_KEY not set")

    try:
        current_limit = int(os.getenv("PROMPTLAYER_RESEARCH_MAX_TOKENS", "30000"))
    except ValueError:
        current_limit = 30000
    max_retries = 2
    research_base = str(research).encode("utf-8", errors="replace").decode("utf-8")

    for attempt in range(max_retries + 1):
        research_clean = truncate_research_for_promptlayer(research_base, max_tokens=current_limit)
        research_tokens = estimate_tokens(research_clean)
        input_vars = {
            "topic": topic,
            "llm_deep_research": research_clean,
            "llm_deep_reserach": research_clean,  # workflow typo: dependency node name
            "research": research_clean,
            "llm_research": research_clean,
        }
        body = json.dumps({"input_variables": input_vars, "return_all_outputs": True})
        body_tokens = estimate_tokens(body)
        logger.info(
            "[PIPELINE] PromptLayer REQUEST (attempt %d): research=%d tokens, limit=%d",
            attempt + 1, research_tokens, current_limit,
        )

        try:
            with httpx.Client() as client:
                run_response = client.post(
                    f"https://api.promptlayer.com/workflows/{workflow_name.replace(' ', '%20')}/run",
                    headers={
                        "Content-Type": "application/json",
                        "X-API-KEY": api_key,
                    },
                    content=body,
                    timeout=300.0,
                )
            run_response.raise_for_status()
            run_data = run_response.json()

            logger.info("[PIPELINE] PromptLayer RUN response: %s", _truncate_for_log(run_data))
            if not run_data.get("success"):
                logger.error("[PIPELINE] PromptLayer run failed: %s", run_data.get("message", "Unknown"))
                raise RuntimeError(f"PromptLayer run failed: {run_data.get('message', 'Unknown error')}")

            exec_id = run_data.get("workflow_version_execution_id")
            if not exec_id:
                logger.error("[PIPELINE] PromptLayer: no execution id in response")
                raise RuntimeError("No workflow_version_execution_id in response")

            logger.info("[PIPELINE] PromptLayer: exec_id=%s, polling for results...", exec_id)
            with httpx.Client() as client:
                for poll_i in range(90):  # Poll up to ~3 min
                    check_cancelled()  # Bail out if user requested cancellation
                    results_response = client.get(
                        "https://api.promptlayer.com/workflow-version-execution-results",
                        headers={"X-API-KEY": api_key},
                        params={"workflow_version_execution_id": exec_id, "return_all_outputs": True},
                        timeout=30.0,
                    )
                    if results_response.status_code == 202:
                        if poll_i % 5 == 0:
                            logger.info("[PIPELINE] PromptLayer: poll #%d still running (202)", poll_i + 1)
                        time.sleep(15)
                        continue
                    if results_response.status_code != 200:
                        logger.error("[PIPELINE] PromptLayer results failed: %s", results_response.text[:200])
                        raise RuntimeError(f"Failed to get execution results: {results_response.text}")

                    results_data = results_response.json()
                    # DEBUG: log ALL outputs for troubleshooting (structure + node statuses)
                    logger.info("[PIPELINE] PromptLayer RESULTS (all outputs): %s", _truncate_for_log(results_data, max_str=500))

                    # Check for FAILURE/FAILED status before returning any value (PL uses both)
                    for node_name, node_data in list(results_data.items()):
                        if isinstance(node_data, dict):
                            status = node_data.get("status", "").upper()
                            err = node_data.get("error_message") or node_data.get("raw_error_message")
                            err_msg = err.get("raw", err) if isinstance(err, dict) else err
                            if status in ("FAILURE", "FAILED") and err_msg:
                                logger.error("[PIPELINE] PromptLayer node %r FAILED: %s", node_name, err_msg)
                                raise RuntimeError(f"PromptLayer node {node_name!r} failed: {err_msg}")

                    # Response is keyed by node name. We need the ARTICLE, not research.
                    # - Prefer nodes with is_output_node=True (official workflow output)
                    # - Exclude "Research Agent" - it outputs research, not the final article
                    RESEARCH_NODE_NAMES = {"Research Agent", "research agent"}  # blocklist
                    output_node_name = os.getenv("PROMPTLAYER_OUTPUT_NODE")  # e.g. "Write Article"

                    def _extract_val(nd: dict) -> str | None:
                        v = nd.get("value")
                        if v is not None and isinstance(v, str) and len(v) > 100:
                            return v
                        if isinstance(v, dict) and "value" in v:
                            s = str(v["value"])
                            return s if len(s) > 100 else None
                        return None

                    # 1) Use explicitly configured output node if set
                    if output_node_name and output_node_name in results_data:
                        nd = results_data.get(output_node_name)
                        if isinstance(nd, dict) and nd.get("status", "SUCCESS").upper() == "SUCCESS":
                            out = _extract_val(nd)
                            if out:
                                latency_ms = (time.perf_counter() - start) * 1000
                                observe_sub_agent(
                                    name="promptlayer_seo",
                                    input_keys=["topic", "research"],
                                    output_size=len(out),
                                    latency_ms=latency_ms,
                                    status="success",
                                    provider="promptlayer",
                                )
                                logger.info("[PIPELINE] PromptLayer DONE: node=%r (configured) len=%d", output_node_name, len(out))
                                return out
                    # 2) Prefer nodes marked as output node (is_output_node=True)
                    for node_name, node_data in results_data.items():
                        if node_name in RESEARCH_NODE_NAMES or not isinstance(node_data, dict):
                            continue
                        if not node_data.get("is_output_node"):
                            continue
                        if node_data.get("status", "SUCCESS").upper() != "SUCCESS":
                            continue
                        out = _extract_val(node_data)
                        if out:
                            latency_ms = (time.perf_counter() - start) * 1000
                            observe_sub_agent(
                                name="promptlayer_seo",
                                input_keys=["topic", "research"],
                                output_size=len(out),
                                latency_ms=latency_ms,
                                status="success",
                                provider="promptlayer",
                            )
                            logger.info("[PIPELINE] PromptLayer DONE: node=%r (is_output_node) len=%d", node_name, len(out))
                            return out
                    # 3) Fallback: any non-research node with long output
                    for node_name, node_data in results_data.items():
                        if node_name in RESEARCH_NODE_NAMES:
                            continue
                        if not isinstance(node_data, dict):
                            continue
                        if node_data.get("status", "SUCCESS").upper() != "SUCCESS":
                            continue
                        out = _extract_val(node_data)
                        if out:
                            latency_ms = (time.perf_counter() - start) * 1000
                            observe_sub_agent(
                                name="promptlayer_seo",
                                input_keys=["topic", "research"],
                                output_size=len(out),
                                latency_ms=latency_ms,
                                status="success",
                                provider="promptlayer",
                            )
                            logger.info("[PIPELINE] PromptLayer DONE: node=%r len=%d", node_name, len(out))
                            return out
                    if isinstance(results_data, dict) and "final_output" in results_data:
                        fo = results_data["final_output"]
                        if isinstance(fo, str):
                            latency_ms = (time.perf_counter() - start) * 1000
                            observe_sub_agent(
                                name="promptlayer_seo",
                                input_keys=["topic", "research"],
                                output_size=len(fo),
                                latency_ms=latency_ms,
                                status="success",
                                provider="promptlayer",
                            )
                            return fo
                        if isinstance(fo, dict):
                            for k, v in fo.items():
                                if k in RESEARCH_NODE_NAMES:
                                    continue
                                if isinstance(v, dict):
                                    st = (v.get("status") or "SUCCESS").upper()
                                    if st != "SUCCESS":
                                        continue
                                    if "value" in v:
                                        out = str(v["value"])
                                        if len(out) > 100:
                                            latency_ms = (time.perf_counter() - start) * 1000
                                            observe_sub_agent(
                                                name="promptlayer_seo",
                                                input_keys=["topic", "research"],
                                                output_size=len(out),
                                                latency_ms=latency_ms,
                                                status="success",
                                                provider="promptlayer",
                                            )
                                            return out
                    time.sleep(15)

            latency_ms = (time.perf_counter() - start) * 1000
            observe_sub_agent(
                name="promptlayer_seo",
                input_keys=["topic", "research"],
                output_size=0,
                latency_ms=latency_ms,
                status="failure",
                provider="promptlayer",
            )
            logger.error("[PIPELINE] PromptLayer: timed out after 90 polls")
            raise RuntimeError("PromptLayer execution timed out or returned no output")
        except RuntimeError as e:
            err_str = str(e)
            parsed = _parse_token_overflow_for_retry(err_str)
            if parsed and attempt < max_retries:
                got_tokens, limit_tokens = parsed
                new_limit = int(current_limit * (limit_tokens / got_tokens) * 0.9)
                new_limit = max(new_limit, 5000)
                if new_limit >= current_limit:
                    new_limit = int(current_limit * 0.5)
                logger.warning("[PIPELINE] Token overflow: got=%d limit=%d -> chunking to %d tokens and retrying", got_tokens, limit_tokens, new_limit)
                emit_status(f"Chunking down research ({got_tokens}→{new_limit} tokens) and retrying...")
                current_limit = new_limit
                continue
            raise


def write_article_from_research(topic: str, research: str) -> str:
    """
    Run only PromptLayer SEO agent with given research (no deep research, no Tavily).
    Use when you have existing content (e.g. a draft to rewrite) as the research input.
    Returns Markdown article.
    """
    logger.info("[PIPELINE] write_article_from_research: topic=%r, research_len=%d", topic, len(research))
    return run_promptlayer_agent(topic, research)


def write_article(topic: str, include_tavily: bool = True, tavily_max_results: int = 20) -> str:
    """
    Full pipeline: Deep research + Tavily enrichment + PromptLayer SEO agent.
    Returns Markdown article.
    """
    logger.info("[PIPELINE] write_article START: topic=%r, include_tavily=%s, tavily_max_results=%d", topic, include_tavily, tavily_max_results)
    research = run_deep_research(topic)
    logger.info("[PIPELINE] Step 1/3 (deep research) complete, research_len=%d", len(research))

    if include_tavily:
        tavily_results = run_tavily_enrichment(topic, max_results=tavily_max_results)
        tavily_text = format_tavily_for_research(tavily_results)
        combined_research = f"{research}\n\n## Additional Sources (Tavily Enrichment)\n\n{tavily_text}"
        logger.info("[PIPELINE] Step 2/3 (Tavily) complete, sources=%d, combined_len=%d", tavily_max_results, len(combined_research))
    else:
        combined_research = research
        logger.info("[PIPELINE] Step 2/3 (Tavily) skipped")

    article = run_promptlayer_agent(topic, combined_research)
    logger.info("[PIPELINE] write_article DONE: article_len=%d chars", len(article))
    return article
