"""Narrator agent: observes workflow events and produces a human-readable summary.

Controlled via NARRATOR_AGENT_ENABLED env var (default: false).
Uses gpt-4o-mini for low cost (~$0.0002 per narration).
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

NARRATOR_ENABLED = os.getenv("NARRATOR_AGENT_ENABLED", "false").lower() in (
    "1",
    "true",
    "yes",
)


def narrate_trace(
    events: list[dict],
    user_message: str = "",
    final_message: str = "",
) -> Optional[str]:
    """Generate a human-readable narrative of what happened during a workflow.

    Returns None if the narrator is disabled or if generation fails.
    """
    if not NARRATOR_ENABLED:
        return None
    if not events:
        return None

    # Build a concise step summary for the LLM
    steps: list[str] = []
    for ev in events:
        etype = ev.get("event_type")
        if etype == "agent_call":
            model = ev.get("model", "unknown")
            tokens = ev.get("tokens") or {}
            dur = ev.get("span", {}).get("duration_ms", 0)
            tool_calls = (ev.get("response") or {}).get("tool_calls") or []
            tools_str = (
                ", ".join(tc.get("name", "?") for tc in tool_calls)
                if tool_calls
                else "no tools called"
            )
            steps.append(
                f"LLM call ({model}, {tokens.get('total', 0)} tokens, {dur:.0f}ms) → {tools_str}"
            )
        elif etype == "tool_result":
            name = ev.get("tool_name", "?")
            success = ev.get("success", True)
            dur = ev.get("latency_ms", 0)
            steps.append(f"Tool: {name} ({'OK' if success else 'FAILED'}, {dur:.0f}ms)")
        elif etype == "sub_agent":
            name = ev.get("name", "?")
            provider = ev.get("provider", "?")
            model = ev.get("model", "")
            dur = ev.get("latency_ms", 0)
            model_str = f", model={model}" if model else ""
            steps.append(
                f"Sub-agent: {name} via {provider}{model_str} ({dur:.0f}ms)"
            )

    if not steps:
        return None

    prompt = (
        "You are a concise workflow narrator for a blog-writing AI agent system. "
        "Summarize what happened in this workflow in 2-4 sentences. "
        "Be specific about which models, tools, and sub-agents were used. "
        "Mention total duration and noteworthy events (failures, retries, image generation). "
        "Write in past tense, third person.\n\n"
        f'User asked: "{user_message[:300]}"\n'
        f'Assistant replied: "{final_message[:300]}"\n\n'
        "Steps:\n"
        + "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(steps))
        + "\n\nNarrative summary:"
    )

    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.3,
        )
        narrative = (response.choices[0].message.content or "").strip()
        logger.info("[NARRATOR] Generated narrative (%d chars)", len(narrative))
        return narrative
    except Exception as e:
        logger.warning("[NARRATOR] Failed to generate narrative: %s", e)
        return None
