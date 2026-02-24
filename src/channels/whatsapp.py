"""WhatsApp channel via Green API."""

import os
import re
from typing import Optional

import httpx


def format_for_whatsapp(text: str) -> str:
    """
    Convert markdown to WhatsApp format.
    Official WhatsApp/Green API formatting (https://green-api.com/docs/faq/how-to-format-messages):
    - Bold: *text* | Italic: _text_ | Strikethrough: ~text~ | Monospace: ```text```
    - To escape literal symbols: double them (e.g. __ for underscore)
    """
    if not text:
        return text
    # **bold** and __bold__ (markdown) -> *bold* (WhatsApp)
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)
    text = re.sub(r"__(.+?)__", r"*\1*", text)
    # ~~strikethrough~~ (markdown) -> ~strikethrough~ (WhatsApp)
    text = re.sub(r"~~(.+?)~~", r"~\1~", text)
    # ## Header -> *Header*
    text = re.sub(r"^#+\s*(.+)$", r"*\1*", text, flags=re.MULTILINE)
    # --- horizontal rule -> remove
    text = re.sub(r"\n-{3,}\n", "\n\n", text)
    text = re.sub(r"^-{3,}\n?", "", text)
    # Markdown bullets - or * at line start -> bullet char
    text = re.sub(r"^\s*[-*]\s+", "• ", text, flags=re.MULTILINE)
    return text.strip()


# Green API / WhatsApp message length limit (chars)
_MSG_MAX_LEN = 20000


def send_article_single(
    chat_id: str, content: str, quoted_message_id: Optional[str] = None
) -> None:
    """
    Send the full article as a single WhatsApp message.
    Formats for WhatsApp and truncates at API limit (20k chars) if needed.
    """
    text = format_for_whatsapp(content)
    if len(text) > _MSG_MAX_LEN:
        text = text[:_MSG_MAX_LEN] + "\n\n[… truncated for WhatsApp limit]"
    send_whatsapp_message(chat_id, text, quoted_message_id=quoted_message_id)


def send_message_chunked(
    chat_id: str,
    message: str,
    max_chunk: int = 3000,
    quoted_message_id: Optional[str] = None,
) -> None:
    """
    Send message in chunks. Splits at paragraph boundaries when possible,
    batching consecutive short paragraphs to avoid flooding the chat.
    If quoted_message_id is provided, only the first chunk will be sent as a quoted reply.
    """
    message = format_for_whatsapp(message)
    if len(message) <= max_chunk:
        send_whatsapp_message(chat_id, message, quoted_message_id=quoted_message_id)
        return
    # Split by double newline first (paragraphs), then by single newline, then by space
    parts = re.split(r"\n\n+", message)
    # Batch short paragraphs together so we don't flood the chat with tiny messages
    batched: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}\n\n{part}".strip() if current else part
        if len(candidate) <= max_chunk:
            current = candidate
        else:
            if current:
                batched.append(current)
            current = part
    if current:
        batched.append(current)
    parts = batched

    chunks = []
    for part in parts:
        if len(part) <= max_chunk:
            chunks.append(part)
        else:
            lines = part.split("\n")
            current = ""
            for line in lines:
                if len(line) > max_chunk:
                    if current:
                        chunks.append(current)
                        current = ""
                    for i in range(0, len(line), max_chunk):
                        chunks.append(line[i : i + max_chunk])
                elif len(current) + len(line) + 1 <= max_chunk:
                    current = f"{current}\n{line}".strip() if current else line
                else:
                    if current:
                        chunks.append(current)
                    current = line
            if current:
                chunks.append(current)
    for i, chunk in enumerate(chunks):
        if chunk.strip():
            qid = quoted_message_id if (i == 0 and quoted_message_id) else None
            send_whatsapp_message(chat_id, chunk, quoted_message_id=qid)


def send_image_by_url(
    chat_id: str,
    image_url: str,
    filename: str = "image.png",
    caption: Optional[str] = None,
    quoted_message_id: Optional[str] = None,
) -> dict:
    """
    Send an image via Green API's sendFileByUrl endpoint.
    The image must be a publicly accessible URL (e.g. Supabase Storage public URL).
    Returns the API response (e.g. idMessage) or raises on error.
    """
    instance_id = os.getenv("GREEN_API_INSTANCE_ID")
    token = os.getenv("GREEN_API_TOKEN")
    base_url = os.getenv("GREEN_API_BASE_URL", "https://api.green-api.com")

    if not instance_id or not token:
        raise ValueError(
            "GREEN_API_INSTANCE_ID and GREEN_API_TOKEN must be set in environment"
        )
    url = (
        f"{base_url.rstrip('/')}/waInstance{instance_id}"
        f"/sendFileByUrl/{token}"
    )
    payload: dict = {
        "chatId": chat_id,
        "urlFile": image_url,
        "fileName": filename,
    }
    if caption:
        payload["caption"] = caption[:20000]
    if quoted_message_id:
        payload["quotedMessageId"] = quoted_message_id

    with httpx.Client(timeout=30) as client:
        resp = client.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()


def send_whatsapp_message(
    chat_id: str, message: str, quoted_message_id: Optional[str] = None
) -> dict:
    """
    Send a text message via Green API.
    If quoted_message_id is provided, the message will be sent as a reply quoting that message.
    Returns the API response (e.g. idMessage) or raises on error.
    """
    instance_id = os.getenv("GREEN_API_INSTANCE_ID")
    token = os.getenv("GREEN_API_TOKEN")
    base_url = os.getenv("GREEN_API_BASE_URL", "https://api.green-api.com")

    if not instance_id or not token:
        raise ValueError(
            "GREEN_API_INSTANCE_ID and GREEN_API_TOKEN must be set in environment"
        )
    url = (
        f"{base_url.rstrip('/')}/waInstance{instance_id}"
        f"/sendMessage/{token}"
    )
    payload: dict = {"chatId": chat_id, "message": message[:20000]}
    if quoted_message_id:
        payload["quotedMessageId"] = quoted_message_id
    with httpx.Client(timeout=30) as client:
        resp = client.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()


def parse_incoming_text_webhook(payload: dict) -> Optional[dict]:
    """
    Parse Green API webhook payload.
    Returns dict with chatId, textMessage, and optionally quotedMessage.
    Handles textMessage and quotedMessage (reply) types.
    """
    if payload.get("typeWebhook") != "incomingMessageReceived":
        return None
    msg_data = payload.get("messageData") or {}
    msg_type = msg_data.get("typeMessage")

    text = None
    quoted = None

    if msg_type == "textMessage":
        text_data = msg_data.get("textMessageData") or {}
        text = text_data.get("textMessage")
    elif msg_type == "quotedMessage":
        ext_data = msg_data.get("extendedTextMessageData") or {}
        text = ext_data.get("text")
        quoted_msg = msg_data.get("quotedMessage") or {}
        quoted = quoted_msg.get("textMessage")

    if not text or not isinstance(text, str):
        return None
    sender = payload.get("senderData") or {}
    chat_id = sender.get("chatId")
    if not chat_id:
        return None
    result = {"chatId": chat_id, "textMessage": text.strip()}
    if quoted:
        result["quotedMessage"] = quoted.strip()
    # idMessage of the incoming message - use as quotedMessageId when replying
    msg_id = payload.get("idMessage")
    if msg_id:
        result["idMessage"] = msg_id
    return result
