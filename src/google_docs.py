"""Create Google Docs from article markdown. Uses OAuth (user) or service account credentials."""

import os
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from project root (parent of src/)
_load_env = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_load_env)

SCOPES = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.file",
]
_TOKEN_PATH = Path(__file__).resolve().parent.parent / "secrets" / "google-token.json"


def _use_oauth() -> bool:
    """True if OAuth credentials (client_id + client_secret) are configured."""
    return bool(os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET"))


def _get_oauth_creds():
    """Get OAuth credentials. Runs browser flow on first use, then uses saved token."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    if _TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(_TOKEN_PATH), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Build client config from env
            client_id = os.getenv("GOOGLE_CLIENT_ID")
            client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
            if not client_id or not client_secret:
                raise ValueError(
                    "GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET required for OAuth. "
                    "Create OAuth Desktop credentials in Google Cloud Console."
                )
            flow = InstalledAppFlow.from_client_config(
                {
                    "installed": {
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "redirect_uris": ["http://localhost"],
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                    }
                },
                SCOPES,
            )
            creds = flow.run_local_server(port=0)

        _TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_TOKEN_PATH, "w") as f:
            f.write(creds.to_json())

    return creds


def _get_service_account_creds():
    """Get service account credentials."""
    from google.oauth2 import service_account

    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("GOOGLE_CREDENTIALS_PATH")
    if creds_path and not os.path.isabs(creds_path):
        creds_path = str(Path(__file__).resolve().parent.parent / creds_path)
    if not creds_path or not Path(creds_path).exists():
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS must point to a service account JSON file.")
    return service_account.Credentials.from_service_account_file(creds_path, scopes=SCOPES)


def _get_creds():
    """Get credentials: OAuth (user) if configured, else service account."""
    if _use_oauth():
        return _get_oauth_creds()
    return _get_service_account_creds()


def _get_docs_service():
    """Build Google Docs API service."""
    from googleapiclient.discovery import build
    return build("docs", "v1", credentials=_get_creds())


def _get_drive_service():
    """Build Google Drive API service."""
    from googleapiclient.discovery import build
    return build("drive", "v3", credentials=_get_creds())


def _parse_inline_formatting(text: str) -> tuple[str, list[tuple[int, int, str, Optional[str]]]]:
    """
    Parse **bold**, *italic*, and [anchor](url) from text in a single left-to-right pass.
    Returns (plain_text, [(start, end, style, url?)]) where positions are in plain-text coordinates.
    """
    token_re = re.compile(
        r'\[([^\]]*)\]\(([^)]+)\)'        # group 1=anchor, group 2=url  (link)
        r'|\*\*([^*]+)\*\*'               # group 3=inner               (bold **)
        r'|__([^_]+)__'                    # group 4=inner               (bold __)
        r'|(?<!\*)\*([^*\n]+)\*(?!\*)'     # group 5=inner               (italic *)
        r'|(?<!_)_([^_\n]+)_(?!_)'         # group 6=inner               (italic _)
    )
    styles: list[tuple[int, int, str, Optional[str]]] = []
    result: list[str] = []
    pos = 0
    for m in token_re.finditer(text):
        result.append(text[pos:m.start()])
        plain_start = sum(len(p) for p in result)
        if m.group(1) is not None:       # link
            anchor = m.group(1)
            result.append(anchor)
            styles.append((plain_start, plain_start + len(anchor), "link", m.group(2)))
        elif m.group(3) is not None:     # bold **
            inner = m.group(3)
            result.append(inner)
            styles.append((plain_start, plain_start + len(inner), "bold", None))
        elif m.group(4) is not None:     # bold __
            inner = m.group(4)
            result.append(inner)
            styles.append((plain_start, plain_start + len(inner), "bold", None))
        elif m.group(5) is not None:     # italic *
            inner = m.group(5)
            result.append(inner)
            styles.append((plain_start, plain_start + len(inner), "italic", None))
        elif m.group(6) is not None:     # italic _
            inner = m.group(6)
            result.append(inner)
            styles.append((plain_start, plain_start + len(inner), "italic", None))
        pos = m.end()
    result.append(text[pos:])
    return "".join(result), styles


def _parse_markdown_blocks(markdown: str) -> list[dict]:
    """
    Parse markdown into blocks: {type, content, level?, url?, alt?, is_list?, inline_styles?}.
    Inline styles: [(start, end, 'bold'|'italic'|'link', url?)] for paragraphs.
    """
    blocks = []
    image_re = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
    raw_blocks = re.split(r"\n\s*\n", markdown)

    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue

        # Entire block is a single image
        img_match = image_re.fullmatch(block)
        if img_match:
            blocks.append({"type": "image", "alt": img_match.group(1), "url": img_match.group(2)})
            continue

        # Block may contain text + inline images
        parts = image_re.split(block)
        if len(parts) > 1:
            i = 0
            while i < len(parts):
                if i % 3 == 0:
                    text = parts[i].strip()
                    if text:
                        if text.startswith("#"):
                            level = len(text) - len(text.lstrip("#"))
                            heading_text = text.lstrip("# ").strip()
                            plain_h, inline_h = _parse_inline_formatting(heading_text)
                            blocks.append({"type": "heading", "content": plain_h, "level": min(level, 3), "inline_styles": inline_h})
                        elif text.startswith("- ") or text.startswith("* "):
                            content = text[2:].strip()
                            plain, inline = _parse_inline_formatting(content)
                            blocks.append({"type": "list_item", "content": plain, "inline_styles": inline})
                        else:
                            plain, inline = _parse_inline_formatting(text)
                            blocks.append({"type": "paragraph", "content": plain, "inline_styles": inline})
                elif i % 3 == 1 and i + 1 < len(parts):
                    blocks.append({"type": "image", "alt": parts[i], "url": parts[i + 1]})
                i += 1
            continue

        if block.startswith("#"):
            level = len(block) - len(block.lstrip("#"))
            heading_text = block.lstrip("# ").strip()
            plain_h, inline_h = _parse_inline_formatting(heading_text)
            blocks.append({"type": "heading", "content": plain_h, "level": min(level, 3), "inline_styles": inline_h})
        else:
            # Split into lines; consecutive "- " or "* " lines = list items
            lines = block.split("\n")
            i = 0
            while i < len(lines):
                line = lines[i]
                stripped = line.strip()
                if stripped.startswith("- ") or stripped.startswith("* "):
                    content = stripped[2:].strip()
                    plain, inline = _parse_inline_formatting(content)
                    blocks.append({"type": "list_item", "content": plain, "inline_styles": inline})
                    i += 1
                else:
                    # Collect non-list lines into one paragraph
                    para_lines = []
                    while i < len(lines) and lines[i].strip() and not (
                        lines[i].strip().startswith("- ") or lines[i].strip().startswith("* ")
                    ):
                        para_lines.append(lines[i].strip())
                        i += 1
                    if para_lines:
                        para_text = " ".join(para_lines)
                        plain, inline = _parse_inline_formatting(para_text)
                        blocks.append({"type": "paragraph", "content": plain, "inline_styles": inline})
                if i < len(lines) and not lines[i].strip():
                    i += 1

    return blocks


def _utf16_len(s: str) -> int:
    """Google Docs API uses UTF-16 code units for indices."""
    return len(s.encode("utf-16-le")) // 2


def create_doc_from_markdown(
    markdown: str,
    title: str = "Article",
    folder_id: Optional[str] = None,
) -> dict:
    """
    Create a Google Doc from markdown content. Images (![alt](url)) must use public URLs.
    If folder_id is set, the doc is created in that folder (must be shared with the service account).
    Returns {document_id, document_url, error?}.
    """
    docs_service = _get_docs_service()
    drive_service = _get_drive_service()

    # Create document: use Drive API in user's folder to avoid service account quota, else Docs API
    folder_id = folder_id or os.getenv("GOOGLE_DOCS_FOLDER_ID")
    if folder_id:
        file_meta = {"name": title[:300], "mimeType": "application/vnd.google-apps.document", "parents": [folder_id]}
        file = drive_service.files().create(body=file_meta, fields="id").execute()
        document_id = file.get("id")
    else:
        body = {"title": title[:300]}
        doc = docs_service.documents().create(body=body).execute()
        document_id = doc.get("documentId")
    if not document_id:
        raise RuntimeError("Failed to create document")

    # Get initial document to find endIndex of body content
    doc = docs_service.documents().get(documentId=document_id).execute()
    content = doc.get("body", {}).get("content", [])
    # Find the last index (end of body). Index must be < segment endIndex, so use endIndex-1 when at boundary.
    idx = 1
    for el in content:
        if "endIndex" in el:
            idx = el["endIndex"]
    # Drive-created docs can have minimal structure (endIndex=2); insert must be < endIndex, so use 1.
    if idx <= 2:
        idx = 1

    blocks = _parse_markdown_blocks(markdown)
    insert_requests = []
    style_requests = []

    is_first_block = True
    for b in blocks:
        if b["type"] == "paragraph":
            text = b["content"] + "\n"
            insert_requests.append({"insertText": {"location": {"index": idx}, "text": text}})
            _add_text_styles(idx, b["content"], b.get("inline_styles"), style_requests)
            idx += _utf16_len(text)
        elif b["type"] == "list_item":
            bullet_content = "- " + b["content"] + "\n"
            insert_requests.append({"insertText": {"location": {"index": idx}, "text": bullet_content}})
            end = idx + _utf16_len(bullet_content)
            _add_text_styles(idx + 2, b["content"], b.get("inline_styles"), style_requests)
            # Add spacing below each bullet so items aren't condensed
            style_requests.append({
                "updateParagraphStyle": {
                    "range": {"startIndex": idx, "endIndex": end},
                    "paragraphStyle": {
                        "spaceBelow": {"magnitude": 6, "unit": "PT"},
                    },
                    "fields": "spaceBelow",
                },
            })
            idx = end
        elif b["type"] == "heading":
            text = b["content"] + "\n"
            insert_requests.append({"insertText": {"location": {"index": idx}, "text": text}})
            end = idx + _utf16_len(text)
            level = b.get("level", 1)
            style_requests.append({
                "updateParagraphStyle": {
                    "range": {"startIndex": idx, "endIndex": end},
                    "paragraphStyle": {"namedStyleType": f"HEADING_{min(level, 3)}"},
                    "fields": "namedStyleType",
                },
            })
            _add_text_styles(idx, b["content"], b.get("inline_styles"), style_requests)
            idx = end
        elif b["type"] == "image":
            url = b.get("url", "").strip()
            if not url or not (url.startswith("http://") or url.startswith("https://")):
                is_first_block = False
                continue
            if not is_first_block:
                insert_requests.append({"insertText": {"location": {"index": idx}, "text": "\n"}})
                idx += 1
            # Hero image (first image in doc, or alt contains "hero"): render full-width
            alt = b.get("alt", "").lower()
            is_hero = (is_first_block and b["type"] == "image") or "hero" in alt
            if is_hero:
                img_size = {"height": {"magnitude": 260, "unit": "PT"}, "width": {"magnitude": 468, "unit": "PT"}}
            else:
                img_size = {"height": {"magnitude": 200, "unit": "PT"}, "width": {"magnitude": 300, "unit": "PT"}}
            insert_requests.append({
                "insertInlineImage": {
                    "location": {"index": idx},
                    "uri": url,
                    "objectSize": img_size,
                },
            })
            idx += 1
            insert_requests.append({"insertText": {"location": {"index": idx}, "text": nl}})
            idx += 1
        is_first_block = False

    all_requests = insert_requests + style_requests

    if all_requests:
        docs_service.documents().batchUpdate(documentId=document_id, body={"requests": all_requests}).execute()

    # Make document viewable by anyone with link
    try:
        drive_service.permissions().create(
            fileId=document_id,
            body={"type": "anyone", "role": "reader"},
        ).execute()
    except Exception:
        pass  # Doc is still created; sharing may fail if permissions restricted

    doc_url = f"https://docs.google.com/document/d/{document_id}/edit"
    return {"document_id": document_id, "document_url": doc_url}


def _document_id_from_url(url: str) -> Optional[str]:
    """Extract document ID from a Google Docs URL."""
    if not url or not isinstance(url, str):
        return None
    # https://docs.google.com/document/d/DOC_ID/edit
    match = re.search(r"/document/d/([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None


def _read_paragraph_element(element: dict) -> str:
    """Extract text from a ParagraphElement."""
    text_run = element.get("textRun")
    if not text_run:
        return ""
    return text_run.get("content", "")


def _read_structural_elements(elements: list) -> str:
    """Recursively extract markdown-like text from structural elements.

    Tables are exported as markdown tables so downstream markdown consumers
    (like Ghost lexical conversion) can render them as native table blocks.
    """
    parts = []
    for item in elements or []:
        if "paragraph" in item:
            for elem in item["paragraph"].get("elements", []):
                parts.append(_read_paragraph_element(elem))
        elif "table" in item:
            rows: list[list[str]] = []
            for row in item["table"].get("tableRows", []):
                cells: list[str] = []
                for cell in row.get("tableCells", []):
                    cell_text = _read_structural_elements(cell.get("content", []))
                    cell_text = " ".join(cell_text.replace("\n", " ").split()).strip()
                    cells.append(cell_text)
                if cells:
                    rows.append(cells)
            if rows:
                col_count = max(len(r) for r in rows)
                normalized_rows = [r + ([""] * (col_count - len(r))) for r in rows]
                header = normalized_rows[0]
                separator = ["---"] * col_count
                table_lines = [
                    "| " + " | ".join(header) + " |",
                    "| " + " | ".join(separator) + " |",
                ]
                for body_row in normalized_rows[1:]:
                    table_lines.append("| " + " | ".join(body_row) + " |")
                parts.append("\n" + "\n".join(table_lines) + "\n")
        elif "tableOfContents" in item:
            parts.append(_read_structural_elements(item["tableOfContents"].get("content", [])))
    return "".join(parts)


def fetch_doc_content(document_id_or_url: str) -> dict:
    """
    Fetch and extract text content from a Google Doc.

    Args:
        document_id_or_url: Document ID (e.g. 1u058OVd-53jK6ky4Mf0JT3SLKosQJ6eeOUTcTXEa274)
            or full Google Docs URL.

    Returns:
        {success, content, title, document_id, error?}
        The document must be viewable by the service account or OAuth user.
    """
    document_id = _document_id_from_url(document_id_or_url) or document_id_or_url
    if not document_id or len(document_id) < 10:
        return {"success": False, "error": "Invalid document ID or URL. Use a Google Docs URL or document ID."}

    try:
        docs_service = _get_docs_service()
        doc = docs_service.documents().get(documentId=document_id).execute()

        title = doc.get("title", "Untitled")
        content_elements = doc.get("body", {}).get("content", [])
        text = _read_structural_elements(content_elements).strip()

        return {
            "success": True,
            "content": text,
            "title": title,
            "document_id": document_id,
        }
    except Exception as e:
        err_str = str(e).lower()
        if "403" in err_str or "forbidden" in err_str or "not found" in err_str:
            return {
                "success": False,
                "error": (
                    "Cannot access this document. Share it with the service account email "
                    "(if using service account) or ensure the OAuth user has access."
                ),
            }
        return {"success": False, "error": str(e)}


def _add_text_styles(block_idx: int, content: str, inline_styles: Optional[list], target: list) -> None:
    """Apply bold, italic, and link styles to a text block. Shared by create and update."""
    for start, end, style, url in inline_styles or []:
        if start >= end:
            continue
        s_utf16 = _utf16_len(content[:start])
        e_utf16 = _utf16_len(content[:end])
        r_start, r_end = block_idx + s_utf16, block_idx + e_utf16
        if r_start >= r_end:
            continue
        if style == "bold":
            target.append({
                "updateTextStyle": {
                    "range": {"startIndex": r_start, "endIndex": r_end},
                    "textStyle": {"bold": True},
                    "fields": "bold",
                },
            })
        elif style == "italic":
            target.append({
                "updateTextStyle": {
                    "range": {"startIndex": r_start, "endIndex": r_end},
                    "textStyle": {"italic": True},
                    "fields": "italic",
                },
            })
        elif style == "link" and url:
            url_str = url.strip()
            if not url_str.startswith("http://") and not url_str.startswith("https://"):
                url_str = "https://" + url_str
            target.append({
                "updateTextStyle": {
                    "range": {"startIndex": r_start, "endIndex": r_end},
                    "textStyle": {"link": {"url": url_str}},
                    "fields": "link",
                },
            })


def inject_links_into_doc(document_id: str, link_placements: list) -> dict:
    """
    Inject hyperlinks into a Google Doc by finding phrases and applying link styles.
    Does NOT rewrite the document — purely applies updateTextStyle to existing text runs.

    Args:
        document_id: Google Doc ID
        link_placements: [{"phrase": "exact verbatim text to linkify", "url": "https://..."}]

    Returns:
        {"success": True, "placed": [...], "not_placed": [...]}
    """
    docs_service = _get_docs_service()
    doc = docs_service.documents().get(documentId=document_id).execute()
    body_content = doc.get("body", {}).get("content", [])

    requests = []
    placed = []
    not_placed = []

    for placement in link_placements:
        phrase = (placement.get("phrase") or "").strip()
        url = (placement.get("url") or "").strip()
        if not phrase or not url:
            not_placed.append({"phrase": phrase, "url": url, "reason": "empty phrase or url"})
            continue
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url

        found = False
        for item in body_content:
            if "paragraph" not in item:
                continue
            for element in item["paragraph"].get("elements", []):
                text_run = element.get("textRun", {})
                content = text_run.get("content", "")
                start_idx = element.get("startIndex", 0)

                char_pos = content.find(phrase)
                if char_pos >= 0:
                    # Google Docs API uses UTF-16 code unit indices
                    link_start = start_idx + _utf16_len(content[:char_pos])
                    link_end = link_start + _utf16_len(phrase)
                    requests.append({
                        "updateTextStyle": {
                            "range": {"startIndex": link_start, "endIndex": link_end},
                            "textStyle": {"link": {"url": url}},
                            "fields": "link",
                        }
                    })
                    placed.append({"phrase": phrase, "url": url})
                    found = True
                    break
            if found:
                break

        if not found:
            not_placed.append({"phrase": phrase, "url": url, "reason": "phrase not found in doc"})

    if requests:
        docs_service.documents().batchUpdate(
            documentId=document_id, body={"requests": requests}
        ).execute()

    return {"success": True, "placed": placed, "not_placed": not_placed}


def update_doc_from_markdown(document_id: str, markdown: str) -> dict:
    """
    Replace the content of an existing Google Doc with new markdown.
    Preserves inline formatting: bold, italic, and links.
    Returns {document_id, document_url}.
    """
    docs_service = _get_docs_service()
    doc = docs_service.documents().get(documentId=document_id).execute()
    content = doc.get("body", {}).get("content", [])
    end_idx = 1
    for el in content:
        if "endIndex" in el:
            end_idx = el["endIndex"]
    if end_idx <= 2:
        end_idx = 2
    requests = []
    if end_idx > 2:
        requests.append({"deleteContentRange": {"range": {"startIndex": 1, "endIndex": end_idx - 1}}})
    blocks = _parse_markdown_blocks(markdown)
    insert_requests = []
    style_requests = []
    idx = 1
    is_first_block = True
    for b in blocks:
        if b["type"] == "paragraph":
            text = b["content"] + "\n"
            insert_requests.append({"insertText": {"location": {"index": idx}, "text": text}})
            _add_text_styles(idx, b["content"], b.get("inline_styles"), style_requests)
            idx += _utf16_len(text)
        elif b["type"] == "list_item":
            bullet_content = "- " + b["content"] + "\n"
            insert_requests.append({"insertText": {"location": {"index": idx}, "text": bullet_content}})
            end = idx + _utf16_len(bullet_content)
            _add_text_styles(idx + 2, b["content"], b.get("inline_styles"), style_requests)
            style_requests.append({
                "updateParagraphStyle": {
                    "range": {"startIndex": idx, "endIndex": end},
                    "paragraphStyle": {"spaceBelow": {"magnitude": 6, "unit": "PT"}},
                    "fields": "spaceBelow",
                },
            })
            idx = end
        elif b["type"] == "heading":
            text = b["content"] + "\n"
            insert_requests.append({"insertText": {"location": {"index": idx}, "text": text}})
            end = idx + _utf16_len(text)
            level = b.get("level", 1)
            style_requests.append({
                "updateParagraphStyle": {
                    "range": {"startIndex": idx, "endIndex": end},
                    "paragraphStyle": {"namedStyleType": f"HEADING_{min(level, 3)}"},
                    "fields": "namedStyleType",
                },
            })
            _add_text_styles(idx, b["content"], b.get("inline_styles"), style_requests)
            idx = end
        elif b["type"] == "image":
            url = b.get("url", "").strip()
            if not url or not (url.startswith("http://") or url.startswith("https://")):
                is_first_block = False
                continue
            if not is_first_block:
                insert_requests.append({"insertText": {"location": {"index": idx}, "text": "\n"}})
                idx += 1
            alt = b.get("alt", "").lower()
            is_hero = (is_first_block and b["type"] == "image") or "hero" in alt
            if is_hero:
                img_size = {"height": {"magnitude": 260, "unit": "PT"}, "width": {"magnitude": 468, "unit": "PT"}}
            else:
                img_size = {"height": {"magnitude": 200, "unit": "PT"}, "width": {"magnitude": 300, "unit": "PT"}}
            insert_requests.append({
                "insertInlineImage": {
                    "location": {"index": idx},
                    "uri": url,
                    "objectSize": img_size,
                },
            })
            idx += 1
            insert_requests.append({"insertText": {"location": {"index": idx}, "text": "\n"}})
            idx += 1
        is_first_block = False
    # Execute inserts first, then style updates
    all_requests = requests + insert_requests + style_requests
    docs_service.documents().batchUpdate(documentId=document_id, body={"requests": all_requests}).execute()
    doc_url = f"https://docs.google.com/document/d/{document_id}/edit"
    return {"document_id": document_id, "document_url": doc_url}


def insert_image_into_doc_at_start(document_id: str, image_url: str) -> dict:
    """Surgically insert a hero image at the very beginning of an existing Google Doc.

    Does NOT rewrite the document — only prepends the image at position 1.
    Returns {document_id, document_url}.
    """
    docs_service = _get_docs_service()
    img_size = {"height": {"magnitude": 260, "unit": "PT"}, "width": {"magnitude": 468, "unit": "PT"}}
    # Insert image at position 1 (before all existing content), then add a newline separator.
    requests = [
        {"insertInlineImage": {"location": {"index": 1}, "uri": image_url, "objectSize": img_size}},
        # After the image is at index 1, insert a newline at index 2 to separate it from the title.
        {"insertText": {"location": {"index": 2}, "text": "\n"}},
    ]
    docs_service.documents().batchUpdate(documentId=document_id, body={"requests": requests}).execute()
    return {"document_id": document_id, "document_url": f"https://docs.google.com/document/d/{document_id}/edit"}


def insert_image_into_doc_after_paragraph(document_id: str, position_after: str, image_url: str) -> dict:
    """Surgically insert an infographic after the paragraph matching position_after.

    Does NOT rewrite the document — only inserts the image at the correct position.
    Falls back to appending at the end if no matching paragraph is found.
    Returns {document_id, document_url}.
    """
    docs_service = _get_docs_service()
    doc = docs_service.documents().get(documentId=document_id).execute()
    body_content = doc.get("body", {}).get("content", [])

    snippet = (position_after or "").strip()[:60]
    insert_idx = None
    last_end_idx = 1

    for item in body_content:
        if "endIndex" in item:
            last_end_idx = item["endIndex"]
        if "paragraph" not in item:
            continue
        para_text = "".join(
            el.get("textRun", {}).get("content", "")
            for el in item["paragraph"].get("elements", [])
        ).replace("\n", " ").strip()
        if snippet and (snippet in para_text or para_text.startswith(snippet[:30])):
            insert_idx = item["endIndex"]
            break

    if insert_idx is None:
        # Fallback: append at end of document (before the final sentinel index)
        insert_idx = max(1, last_end_idx - 1)

    img_size = {"height": {"magnitude": 200, "unit": "PT"}, "width": {"magnitude": 300, "unit": "PT"}}
    requests = [
        {"insertInlineImage": {"location": {"index": insert_idx}, "uri": image_url, "objectSize": img_size}},
        {"insertText": {"location": {"index": insert_idx + 1}, "text": "\n"}},
    ]
    docs_service.documents().batchUpdate(documentId=document_id, body={"requests": requests}).execute()
    return {"document_id": document_id, "document_url": f"https://docs.google.com/document/d/{document_id}/edit"}


def register_gdrive_watch(document_id: str, article_id: str, webhook_base_url: str) -> dict:
    """Register a Google Drive push notification watch for a document.

    Drive will POST to <webhook_base_url>/webhooks/gdrive-changes when the doc changes.
    The article_id is passed as the channel token so the webhook can identify which article changed.

    Returns {channel_id, resource_id, expiry_ms}.
    Raises on failure — caller should handle gracefully.

    Note: The webhook URL domain must be verified in Google Search Console, or you can use a
    custom domain configured on your Fly.io app via WEBHOOK_BASE_URL env var.
    """
    import uuid as _uuid

    channel_id = str(_uuid.uuid4())
    webhook_url = webhook_base_url.rstrip("/") + "/webhooks/gdrive-changes"

    drive_service = _get_drive_service()
    body = {
        "id": channel_id,
        "type": "web_hook",
        "address": webhook_url,
        "token": article_id,  # Returned as X-Goog-Channel-Token in every notification
        "params": {"ttl": "604800"},  # Max 7 days in seconds
    }
    response = drive_service.files().watch(fileId=document_id, body=body).execute()
    return {
        "channel_id": response["id"],
        "resource_id": response["resourceId"],
        "expiry_ms": int(response.get("expiration", 0)),
        "document_id": document_id,
    }


def stop_gdrive_watch(channel_id: str, resource_id: str) -> None:
    """Stop an active Drive push notification watch channel."""
    drive_service = _get_drive_service()
    drive_service.channels().stop(body={
        "id": channel_id,
        "resourceId": resource_id,
    }).execute()


def ensure_gdrive_watch(document_id: str, article_id: str) -> Optional[dict]:
    """Ensure a valid GDrive watch exists for a document. Renews if expiring within 24 hours.

    Reads WEBHOOK_BASE_URL from environment. Returns None if not configured.
    Stores watch info in article metadata via DB helpers.
    """
    import os
    import time as _time

    webhook_base_url = os.getenv("WEBHOOK_BASE_URL", "").rstrip("/")
    if not webhook_base_url:
        return None

    from src.db import get_gdrive_watch, set_gdrive_watch

    existing = get_gdrive_watch(article_id)
    if existing:
        expiry_ms = existing.get("expiry_ms", 0)
        # Keep the watch if it still has > 24 hours left
        if expiry_ms - int(_time.time() * 1000) > 86_400_000:
            return existing
        # Expiring soon — stop the old channel and re-register
        try:
            stop_gdrive_watch(existing["channel_id"], existing["resource_id"])
        except Exception:
            pass  # Best effort; the old channel will expire anyway

    watch_info = register_gdrive_watch(document_id, article_id, webhook_base_url)
    set_gdrive_watch(article_id, watch_info)
    return watch_info


def create_google_doc_from_article(article_id: str, title: Optional[str] = None, folder_id: Optional[str] = None) -> dict:
    """
    Create a Google Doc from an article in the database.
    If GOOGLE_DOCS_FOLDER_ID is set (or folder_id passed), creates in that folder—avoids service account quota.
    Returns {document_id, document_url, article_id, error?}.
    """
    from src.db import get_article

    article = get_article(article_id)
    if not article:
        return {"error": f"Article {article_id} not found"}

    content = article.get("content", "")
    doc_title = title or article.get("title") or "Article"
    return create_doc_from_markdown(content, title=doc_title, folder_id=folder_id)
