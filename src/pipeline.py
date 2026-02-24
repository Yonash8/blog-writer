from __future__ import annotations
"""Core pipeline: Tavily enrichment + PromptLayer SEO agent."""

import json
import logging
import os
import time
from contextvars import ContextVar
from typing import Any, Callable, Optional

import httpx
from dotenv import load_dotenv
from tavily import TavilyClient

from src.observability import observe_sub_agent

load_dotenv()

logger = logging.getLogger(__name__)

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


def run_promptlayer_agent(topic: str, research: str) -> dict[str, Any]:
    """Invoke PromptLayer SEO agent with topic and research.

    Returns {"article": str, "metadata": Any}.
    Article comes from the "add links from research" node (PROMPTLAYER_ARTICLE_NODE).
    Metadata comes from the "metadata agent" node (PROMPTLAYER_METADATA_NODE).
    Auto-retries with chunked research on token overflow.
    """
    logger.info("[PIPELINE] PromptLayer: topic=%r, research_len=%d chars", topic, len(research))
    emit_status("Writing article with PromptLayer...")
    start = time.perf_counter()
    api_key = os.getenv("PROMPTLAYER_API_KEY")
    workflow_name = os.getenv("PROMPTLAYER_WORKFLOW_NAME", "seo agent")
    article_node_name = os.getenv("PROMPTLAYER_ARTICLE_NODE", "add links from research")
    metadata_node_name = os.getenv("PROMPTLAYER_METADATA_NODE", "metadata agent")
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
                        elapsed_min = (poll_i + 1) * 15 // 60
                        if poll_i > 0 and poll_i % 4 == 0:
                            emit_status(f"Still writing... ({elapsed_min}m)")
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

                    def _extract_val(nd: dict) -> str | None:
                        v = nd.get("value")
                        if v is not None and isinstance(v, str) and len(v) > 100:
                            return v
                        if isinstance(v, dict) and "value" in v:
                            s = str(v["value"])
                            return s if len(s) > 100 else None
                        return None

                    def _extract_metadata(nd: dict) -> Any:
                        """Extract metadata node value as-is (dict, str, or None)."""
                        v = nd.get("value")
                        if isinstance(v, dict) and "value" in v:
                            return v["value"]
                        return v

                    # --- Extract article from named node ---
                    article_out: str | None = None
                    metadata_out: Any = None

                    article_nd = results_data.get(article_node_name)
                    if isinstance(article_nd, dict) and article_nd.get("status", "SUCCESS").upper() == "SUCCESS":
                        article_out = _extract_val(article_nd)
                        if article_out:
                            logger.info("[PIPELINE] PromptLayer: article from node %r, len=%d", article_node_name, len(article_out))

                    # --- Extract metadata from named node ---
                    metadata_nd = results_data.get(metadata_node_name)
                    if isinstance(metadata_nd, dict) and metadata_nd.get("status", "SUCCESS").upper() == "SUCCESS":
                        metadata_out = _extract_metadata(metadata_nd)
                        logger.info("[PIPELINE] PromptLayer: metadata from node %r", metadata_node_name)

                    # --- Fallback: any is_output_node if article not found yet ---
                    if not article_out:
                        SKIP_NODES = {article_node_name, metadata_node_name, "Research Agent", "research agent"}
                        for node_name, node_data in results_data.items():
                            if node_name in SKIP_NODES or not isinstance(node_data, dict):
                                continue
                            if not node_data.get("is_output_node"):
                                continue
                            if node_data.get("status", "SUCCESS").upper() != "SUCCESS":
                                continue
                            article_out = _extract_val(node_data)
                            if article_out:
                                logger.info("[PIPELINE] PromptLayer: article fallback from is_output_node %r, len=%d", node_name, len(article_out))
                                break

                    if article_out:
                        latency_ms = (time.perf_counter() - start) * 1000
                        observe_sub_agent(
                            name="promptlayer_seo",
                            input_keys=["topic", "research"],
                            output_size=len(article_out),
                            latency_ms=latency_ms,
                            status="success",
                            provider="promptlayer",
                        )
                        return {"article": article_out, "metadata": metadata_out}

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
            logger.error("[PIPELINE] PromptLayer: timed out after 90 polls (article_node=%r, metadata_node=%r)", article_node_name, metadata_node_name)
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


def run_metadata_agent(article_content: str) -> Any:
    """Call PromptLayer metadata workflow with the approved article content.

    Returns the raw metadata output (dict, str, etc.) from the workflow.
    Uses PROMPTLAYER_METADATA_WORKFLOW env (default "metadata agent").
    """
    logger.info("[PIPELINE] Metadata agent: content_len=%d chars", len(article_content))
    emit_status("Generating SEO metadata...")
    start = time.perf_counter()
    api_key = os.getenv("PROMPTLAYER_API_KEY")
    workflow_name = os.getenv("PROMPTLAYER_METADATA_WORKFLOW", "metadata agent")
    if not api_key:
        raise ValueError("PROMPTLAYER_API_KEY not set")

    with httpx.Client() as client:
        r = client.post(
            f"https://api.promptlayer.com/workflows/{workflow_name.replace(' ', '%20')}/run",
            headers={"Content-Type": "application/json", "X-API-KEY": api_key},
            json={
                "input_variables": {"article": article_content},
                "return_all_outputs": True,
            },
            timeout=300.0,
        )
        r.raise_for_status()
        data = r.json()
        if not data.get("success"):
            raise RuntimeError(f"PromptLayer metadata run failed: {data.get('message', 'Unknown error')}")

        exec_id = data.get("workflow_version_execution_id")
        if not exec_id:
            raise RuntimeError("No workflow_version_execution_id in metadata response")

        logger.info("[PIPELINE] Metadata agent: exec_id=%s, polling...", exec_id)
        for poll_i in range(60):  # up to ~15 min
            check_cancelled()
            res = client.get(
                "https://api.promptlayer.com/workflow-version-execution-results",
                headers={"X-API-KEY": api_key},
                params={"workflow_version_execution_id": exec_id, "return_all_outputs": True},
                timeout=30.0,
            )
            if res.status_code == 202:
                if poll_i > 0 and poll_i % 4 == 0:
                    elapsed_min = (poll_i + 1) * 15 // 60
                    emit_status(f"Generating metadata... ({elapsed_min}m)")
                time.sleep(15)
                continue
            if res.status_code != 200:
                raise RuntimeError(f"Metadata agent results failed: {res.text}")

            results = res.json()
            logger.info("[PIPELINE] Metadata agent RESULTS: %s", _truncate_for_log(results))

            # Check for failures first
            for node_name, node_data in list(results.items()):
                if isinstance(node_data, dict):
                    status = node_data.get("status", "").upper()
                    if status in ("FAILURE", "FAILED"):
                        err = node_data.get("error_message") or node_data.get("raw_error_message")
                        err_msg = err.get("raw", err) if isinstance(err, dict) else err
                        raise RuntimeError(f"Metadata agent node {node_name!r} failed: {err_msg}")

            # Extract output node value — prefer is_output_node: true, then fall back
            def _emit_and_return(node_name, v):
                if isinstance(v, dict) and "value" in v:
                    v = v["value"]
                latency_ms = (time.perf_counter() - start) * 1000
                observe_sub_agent(
                    name="promptlayer_metadata",
                    input_keys=["article"],
                    output_size=len(str(v)),
                    latency_ms=latency_ms,
                    status="success",
                    provider="promptlayer",
                )
                logger.info("[PIPELINE] Metadata agent DONE: node=%r, type=%s", node_name, type(v).__name__)
                return v, exec_id

            # First pass: prefer the designated output node
            for node_name, node_data in results.items():
                if not isinstance(node_data, dict):
                    continue
                if node_data.get("status", "").upper() != "SUCCESS":
                    continue
                if not node_data.get("is_output_node"):
                    continue
                v = node_data.get("value")
                if v is not None:
                    return _emit_and_return(node_name, v)

            # Second pass: fallback to first node with any value
            for node_name, node_data in results.items():
                if not isinstance(node_data, dict):
                    continue
                if node_data.get("status", "").upper() != "SUCCESS":
                    continue
                v = node_data.get("value")
                if v is not None:
                    return _emit_and_return(node_name, v)

            time.sleep(15)

    latency_ms = (time.perf_counter() - start) * 1000
    observe_sub_agent(
        name="promptlayer_metadata",
        input_keys=["article"],
        output_size=0,
        latency_ms=latency_ms,
        status="failure",
        provider="promptlayer",
    )
    raise RuntimeError("Metadata agent timed out or returned no output")


def run_orchestrator_workflow(input_variables: Optional[dict] = None) -> str:
    """Run the PromptLayer orchestrator workflow to get the master system prompt.

    Workflow name is configured via PROMPTLAYER_ORCHESTRATOR_WORKFLOW env
    (default: 'master_system_core').
    Returns the workflow output as a string.
    """
    api_key = os.getenv("PROMPTLAYER_API_KEY")
    workflow_name = os.getenv("PROMPTLAYER_ORCHESTRATOR_WORKFLOW", "blog_writer_master_system_core")
    if not api_key:
        raise ValueError("PROMPTLAYER_API_KEY not set")

    logger.info("[PIPELINE] Orchestrator workflow: name=%r", workflow_name)
    start = time.perf_counter()

    with httpx.Client() as client:
        r = client.post(
            f"https://api.promptlayer.com/workflows/{workflow_name.replace(' ', '%20')}/run",
            headers={"Content-Type": "application/json", "X-API-KEY": api_key},
            json={"input_variables": input_variables or {}, "return_all_outputs": True},
            timeout=60.0,
        )
        r.raise_for_status()
        data = r.json()

        if not data.get("success"):
            raise RuntimeError(f"PromptLayer orchestrator run failed: {data.get('message', 'Unknown')}")

        exec_id = data.get("workflow_version_execution_id")
        if not exec_id:
            raise RuntimeError("No workflow_version_execution_id in orchestrator response")

        logger.info("[PIPELINE] Orchestrator workflow: exec_id=%s, polling...", exec_id)
        for poll_i in range(20):  # up to ~100 seconds
            time.sleep(5)
            res = client.get(
                "https://api.promptlayer.com/workflow-version-execution-results",
                headers={"X-API-KEY": api_key},
                params={"workflow_version_execution_id": exec_id, "return_all_outputs": True},
                timeout=30.0,
            )
            if res.status_code == 202:
                logger.info("[PIPELINE] Orchestrator workflow: poll #%d still running", poll_i + 1)
                continue
            if res.status_code != 200:
                raise RuntimeError(f"Orchestrator workflow results failed: {res.text}")

            results = res.json()
            logger.info("[PIPELINE] Orchestrator workflow RESULTS: %s", _truncate_for_log(results))

            # Check for node failures first
            for node_name, node_data in list(results.items()):
                if isinstance(node_data, dict):
                    status = node_data.get("status", "").upper()
                    if status in ("FAILURE", "FAILED"):
                        err = node_data.get("error_message") or node_data.get("raw_error_message")
                        err_msg = err.get("raw", err) if isinstance(err, dict) else err
                        raise RuntimeError(f"Orchestrator workflow node {node_name!r} failed: {err_msg}")

            # Extract output: prefer is_output_node, then fallback to any node with a string value
            def _get_str(nd: dict) -> Optional[str]:
                v = nd.get("value")
                if isinstance(v, dict) and "value" in v:
                    v = v["value"]
                return v if isinstance(v, str) and len(v) > 10 else None

            for node_name, node_data in results.items():
                if not isinstance(node_data, dict):
                    continue
                if node_data.get("status", "").upper() != "SUCCESS":
                    continue
                if not node_data.get("is_output_node"):
                    continue
                v = _get_str(node_data)
                if v:
                    latency_ms = (time.perf_counter() - start) * 1000
                    logger.info("[PIPELINE] Orchestrator workflow DONE: node=%r, len=%d, latency=%.0fms", node_name, len(v), latency_ms)
                    return v

            # Fallback: first successful node with a non-trivial string
            for node_name, node_data in results.items():
                if not isinstance(node_data, dict):
                    continue
                if node_data.get("status", "").upper() != "SUCCESS":
                    continue
                v = _get_str(node_data)
                if v:
                    latency_ms = (time.perf_counter() - start) * 1000
                    logger.info("[PIPELINE] Orchestrator workflow DONE (fallback): node=%r, len=%d", node_name, len(v))
                    return v

    raise RuntimeError("Orchestrator workflow timed out or returned no output")


def write_article_from_research(topic: str, research: str) -> dict[str, Any]:
    """
    Run only PromptLayer SEO agent with given research (no Tavily).
    Use when you have existing content (e.g. a draft to rewrite) as the research input.
    Returns {"article": str, "metadata": Any}.
    """
    logger.info("[PIPELINE] write_article_from_research: topic=%r, research_len=%d", topic, len(research))
    return run_promptlayer_agent(topic, research)


def write_article(topic: str, include_tavily: bool = False, tavily_max_results: int = 20) -> dict[str, Any]:
    """
    Pipeline: optional Tavily enrichment + PromptLayer SEO agent (research is done inside PL).
    Returns {"article": str, "metadata": Any}.
    """
    logger.info("[PIPELINE] write_article START: topic=%r, include_tavily=%s, tavily_max_results=%d", topic, include_tavily, tavily_max_results)

    if include_tavily:
        tavily_results = run_tavily_enrichment(topic, max_results=tavily_max_results)
        tavily_text = format_tavily_for_research(tavily_results)
        research = f"## Sources (Tavily Enrichment)\n\n{tavily_text}"
        logger.info("[PIPELINE] Tavily complete, sources=%d, research_len=%d", len(tavily_results), len(research))
    else:
        research = ""
        logger.info("[PIPELINE] Tavily skipped")

    result = run_promptlayer_agent(topic, research)
    article = result.get("article", "")
    logger.info("[PIPELINE] write_article DONE: article_len=%d chars", len(article))
    return result
