#!/usr/bin/env python3
"""
Dev mode: Chat with the agent via terminal. No WhatsApp, no web server.
Uses channel="dev", channel_user_id="com" so history is isolated.
Usage: python scripts/dev_chat.py   (or: make dev-chat)
"""
import logging
import sys
import warnings
from pathlib import Path

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Show logs in terminal (agent, pipeline, tools, etc.)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    force=True,
    stream=sys.stderr,
)
# Keep only concise structured trace lines from the agent internals.
logging.getLogger("src.agent").setLevel(logging.WARNING)
logging.getLogger("devchat.trace").setLevel(logging.INFO)
# Reduce noisy network/client logs in terminal dev chat.
for noisy_logger in (
    "httpx",
    "httpcore",
    "urllib3",
    "postgrest",
    "supabase",
    "gotrue",
    "storage3",
    "realtime",
    "src.message_cache",
    "src.db",
    "src.observability",
    "src.prompts_loader",
):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")

from src.agent import run_agent
from src.db import add_message_for_user, get_messages_for_user

CHANNEL = "dev"
CHANNEL_USER_ID = "com"


def main():
    print("Dev chat — agent via terminal (channel=dev, user=com)")
    print("Commands: /quit or /exit to stop, /clear to reset session\n")

    while True:
        try:
            msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not msg:
            continue
        if msg.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Bye.")
            break
        if msg.lower() == "/clear":
            # Clear messages for this dev user (would need a DB call; for now just say ok)
            print("(Session clear not implemented; start fresh in next run)\n")
            continue

        def on_status(text: str):
            low = text.lower()
            if "memory" in low and ("load" in low or "loading" in low):
                return
            logging.getLogger("devchat.trace").info("status | %s", text)

        try:
            msgs = get_messages_for_user(CHANNEL, CHANNEL_USER_ID)
            history = [{"role": m["role"], "content": m["content"]} for m in msgs]
            result = run_agent(
                user_message=msg,
                channel=CHANNEL,
                channel_user_id=CHANNEL_USER_ID,
                history=history,
                format_for_whatsapp=False,
                on_status=on_status,
            )
            out = result.get("message", "").strip()
            add_message_for_user(CHANNEL, CHANNEL_USER_ID, "user", msg)
            add_message_for_user(CHANNEL, CHANNEL_USER_ID, "assistant", out)
            if out:
                print(f"\nAgent: {out}\n")
            else:
                print("\nAgent: (no message)\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
