from __future__ import annotations
"""Image generation: placement analysis (Anthropic), Gemini for image generation, injection, Supabase upload.

Includes hero image and infographic generation with style reference support.
"""

import base64
import json
import logging
import os
import time
from datetime import datetime, timezone
import random
import re
import uuid
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from src.config import get_config
from src.observability import OBSERVABILITY_SAVE_PROMPTS, observe_agent_call
from src.prompts_loader import get_prompt, get_prompt_llm_kwargs

logger = logging.getLogger(__name__)

# OpenAI used only for DALL-E fallback; placement analysis uses Anthropic
_openai: OpenAI | None = None


def _get_openai() -> OpenAI:
    global _openai
    if _openai is None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not set")
        _openai = OpenAI(api_key=key)
    return _openai


def _get_supabase_client():
    """Get a Supabase client for storage operations."""
    from supabase import create_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
    return create_client(url, key)


# ---------------------------------------------------------------------------
# Reference image fetching from Supabase Storage buckets
# ---------------------------------------------------------------------------


def fetch_random_references(bucket_name: str, count: int = 3) -> list[bytes]:
    """Fetch `count` random images from a Supabase Storage bucket.

    Returns a list of image bytes. If the bucket has fewer images than
    requested, returns all available images.
    """
    client = _get_supabase_client()
    storage = client.storage.from_(bucket_name)

    # List all files in the bucket root
    files = storage.list()
    # Filter to image files only
    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif"}
    image_files = [
        f for f in files
        if isinstance(f, dict)
        and any(f.get("name", "").lower().endswith(ext) for ext in image_extensions)
    ]

    if not image_files:
        raise RuntimeError(
            f"No reference images found in bucket '{bucket_name}'. "
            "Upload style reference images via the Supabase Dashboard."
        )

    selected = random.sample(image_files, min(count, len(image_files)))
    logger.info(
        "[IMAGES] Fetched %d/%d reference images from '%s': %s",
        len(selected), len(image_files), bucket_name,
        [f["name"] for f in selected],
    )

    result = []
    for f in selected:
        data = storage.download(f["name"])
        result.append(data)
    return result


# ---------------------------------------------------------------------------
# Gemini 2.5 Flash Image: generation with style references
# ---------------------------------------------------------------------------


def generate_with_gemini_references(
    prompt: str,
    reference_images: list[bytes],
    aspect_ratio: Optional[str] = None,
    prompt_key: str = "hero_image",
) -> bytes:
    """Generate an image using Gemini 2.5 Flash Image with reference images.

    Accepts up to 3 reference images alongside the text prompt.
    Returns the generated image as PNG bytes.
    """
    from google import genai
    from google.genai import types

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set")

    client = genai.Client(api_key=api_key)

    # Build multimodal contents: reference images first, then the text prompt
    contents: list[Any] = []
    for i, img_bytes in enumerate(reference_images[:3]):
        # Detect MIME type from bytes (default to PNG)
        mime = "image/png"
        if img_bytes[:3] == b"\xff\xd8\xff":
            mime = "image/jpeg"
        elif img_bytes[:4] == b"RIFF":
            mime = "image/webp"
        contents.append(
            types.Part.from_bytes(data=img_bytes, mime_type=mime)
        )
    contents.append(prompt)

    config = types.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"],
    )

    logger.info(
        "[IMAGES] Calling Gemini 2.5 Flash Image with %d refs, prompt: %.120s...",
        len(reference_images[:3]), prompt,
    )
    GEMINI_IMAGE_MODELS = {"gemini-2.5-flash-image", "gemini-3-pro-image-preview"}
    model = (
        get_prompt_llm_kwargs(prompt_key).get("model")
        or get_config("image_model_hero", "gemini-2.5-flash-image")
    )
    if model not in GEMINI_IMAGE_MODELS:
        logger.warning(
            "[IMAGES] image_model_hero=%r is not an image-capable model, using gemini-2.5-flash-image",
            model,
        )
        model = "gemini-2.5-flash-image"
    start_ts = datetime.now(timezone.utc).isoformat()
    start_perf = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    duration_ms = (time.perf_counter() - start_perf) * 1000
    end_ts = datetime.now(timezone.utc).isoformat()

    # Extract image bytes from response parts
    img_bytes = None
    for part in response.candidates[0].content.parts:
        if part.inline_data and part.inline_data.mime_type.startswith("image/"):
            img_bytes = part.inline_data.data
            break

    if not img_bytes:
        raise RuntimeError("No image returned from Gemini 2.5 Flash Image")

    # Observability: record the Gemini image generation (prompt + model + duration)
    usage = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
    tokens = None
    if usage:
        inp = getattr(usage, "prompt_token_count", None) or getattr(usage, "input_tokens", None) or 0
        out = getattr(usage, "candidates_token_count", None) or getattr(usage, "output_tokens", None) or 0
        if inp or out:
            tokens = {"input": inp, "output": out, "total": (inp or 0) + (out or 0)}
    _preview_len = 800 if OBSERVABILITY_SAVE_PROMPTS else 400
    observe_agent_call(
        name="gemini_image_generation",
        provider="google",
        model=model,
        prompt={
            "prompt_len": len(prompt),
            "prompt_preview": prompt[:_preview_len] + ("..." if len(prompt) > _preview_len else ""),
            "reference_images_count": len(reference_images[:3]),
        },
        response={
            "output_type": "image",
            "size_bytes": len(img_bytes),
            "content_preview": f"Generated image ({len(img_bytes)} bytes)",
        },
        tokens=tokens,
        span={"start_ts": start_ts, "end_ts": end_ts, "duration_ms": round(duration_ms, 2)},
    )
    return img_bytes


# ---------------------------------------------------------------------------
# Hero image generation
# ---------------------------------------------------------------------------

def _download_image(url: str) -> bytes:
    """Download an image from a URL and return its bytes."""
    import httpx

    with httpx.Client() as client:
        r = client.get(url, timeout=30.0, follow_redirects=True)
        r.raise_for_status()
        return r.content


def generate_hero_image(
    description: str,
    feedback: Optional[str] = None,
    previous_image_url: Optional[str] = None,
) -> tuple[bytes, str]:
    """Generate a hero image using style references from the 'heros' bucket.

    Args:
        description: What the hero image should depict (e.g. "walking into a shop").
        feedback: Optional refinement instructions to append to the prompt.
        previous_image_url: URL of the previously generated image to use as
            reference when refining. When provided (typically alongside feedback),
            the previous image is used as the primary reference so Gemini can see
            what it is modifying, plus 1 random style reference for palette/style.

    Returns:
        (image_bytes, prompt_used) tuple.
    """
    hero_template = get_prompt("hero_image")

    if feedback and previous_image_url:
        # Refinement: use the previous image as primary reference so the model
        # can see what to change, plus 1 style ref for palette consistency.
        try:
            previous_bytes = _download_image(previous_image_url)
            style_refs = fetch_random_references("heros", count=1)
            refs = [previous_bytes] + style_refs
            logger.info("[IMAGES] Hero refinement: using previous image + 1 style ref")
        except Exception as e:
            logger.warning("[IMAGES] Failed to download previous image (%s), falling back to 3 random refs", e)
            refs = fetch_random_references("heros", count=3)
    else:
        refs = fetch_random_references("heros", count=3)

    prompt = hero_template.format(description=description)
    if feedback:
        prompt += f"\n\nAdditional instructions: {feedback}"

    img_bytes = generate_with_gemini_references(prompt, refs)
    logger.info("[IMAGES] Hero image generated (%d bytes)", len(img_bytes))
    return img_bytes, prompt


# ---------------------------------------------------------------------------
# Infographic analysis (Claude Sonnet)
# ---------------------------------------------------------------------------

def analyze_infographic_placement(article_markdown: str) -> dict[str, Any]:
    """Use Claude Sonnet to analyze the article and determine the best infographic opportunity.

    Returns a dict with: position_after, infographic_type, title, description, section_name.
    """
    from anthropic import Anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    analysis_prompt = get_prompt("infographic_analysis")
    model = (
        get_prompt_llm_kwargs("infographic_analysis").get("model")
        or get_config("infographic_analysis_model", "claude-sonnet-4-5")
    )
    if not model.startswith("claude"):
        logger.warning("[IMAGES] infographic_analysis_model=%r is not an Anthropic model, using claude-sonnet-4-5", model)
        model = "claude-sonnet-4-5"

    client = Anthropic(api_key=api_key)
    prompt = analysis_prompt.format(article=article_markdown[:15000])

    start_ts = datetime.now(timezone.utc).isoformat()
    start_perf = time.perf_counter()
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    end_ts = datetime.now(timezone.utc).isoformat()
    duration_ms = (time.perf_counter() - start_perf) * 1000

    usage = getattr(response, "usage", None)
    tokens = None
    if usage:
        tokens = {
            "input": getattr(usage, "input_tokens", None),
            "output": getattr(usage, "output_tokens", None),
            "total": (getattr(usage, "input_tokens", 0) or 0) + (getattr(usage, "output_tokens", 0) or 0),
        }
    content = response.content[0].text.strip()
    observe_agent_call(
        name="analyze_infographic_placement",
        provider="anthropic",
        model=model,
        prompt={"prompt_len": len(prompt), "messages_count": 1},
        response={"content_preview": content[:500] + ("..." if len(content) > 500 else "")},
        tokens=tokens,
        span={"start_ts": start_ts, "end_ts": end_ts, "duration_ms": round(duration_ms, 2)},
    )

    # Strip markdown code fences if present
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    try:
        result = json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse infographic analysis: {e}\nRaw: {content[:500]}") from e

    required = {"position_after", "infographic_type", "title", "description"}
    missing = required - set(result.keys())
    if missing:
        raise RuntimeError(f"Infographic analysis missing fields: {missing}")

    logger.info(
        "[IMAGES] Infographic analysis: type=%s, title=%s, section=%s",
        result.get("infographic_type"), result.get("title"), result.get("section_name"),
    )
    return result


# ---------------------------------------------------------------------------
# Infographic generation
# ---------------------------------------------------------------------------

def generate_infographic(
    article_markdown: str,
    feedback: Optional[str] = None,
    infographic_type: Optional[str] = None,
    previous_image_url: Optional[str] = None,
) -> tuple[bytes, str, dict[str, Any]]:
    """Generate an infographic for an article using style references from the 'infographics' bucket.

    1. Analyze the article to find the best position and type (unless overridden).
    2. Generate the infographic image using Gemini with reference images.

    Args:
        article_markdown: The article content in markdown.
        feedback: Optional refinement instructions.
        infographic_type: Override the auto-detected type.
        previous_image_url: URL of the previously generated infographic to use as
            reference when refining. When provided (typically alongside feedback),
            the previous image is used as the primary reference so Gemini can see
            what it is modifying, plus 1 random style reference for consistency.

    Returns:
        (image_bytes, prompt_used, analysis) where analysis contains position info.
    """
    # Step 1: Analyze article for best infographic placement
    analysis = analyze_infographic_placement(article_markdown)

    # Allow type override
    if infographic_type:
        analysis["infographic_type"] = infographic_type

    # Step 2: Fetch references — use previous image when refining
    if feedback and previous_image_url:
        try:
            previous_bytes = _download_image(previous_image_url)
            style_refs = fetch_random_references("infographics", count=1)
            refs = [previous_bytes] + style_refs
            logger.info("[IMAGES] Infographic refinement: using previous image + 1 style ref")
        except Exception as e:
            logger.warning("[IMAGES] Failed to download previous infographic (%s), falling back to 3 random refs", e)
            refs = fetch_random_references("infographics", count=3)
    else:
        refs = fetch_random_references("infographics", count=3)

    gen_template = get_prompt("infographic_generation")

    # Step 3: Build prompt
    prompt = gen_template.format(
        infographic_type=analysis["infographic_type"],
        title=analysis["title"],
        description=analysis["description"],
    )
    if feedback:
        prompt += f"\n\nAdditional instructions: {feedback}"

    # Step 4: Generate
    img_bytes = generate_with_gemini_references(prompt, refs, prompt_key="infographic_generation")
    logger.info("[IMAGES] Infographic generated (%d bytes)", len(img_bytes))
    return img_bytes, prompt, analysis


# ---------------------------------------------------------------------------
# Inject hero image into markdown (above title)
# ---------------------------------------------------------------------------


def inject_hero_into_markdown(markdown: str, image_url: str, alt_text: str = "Hero image") -> str:
    """Insert a hero image as the very first element in the markdown, above the title."""
    hero_md = f"![{alt_text}]({image_url})"
    return f"{hero_md}\n\n{markdown}"


def inject_infographic_into_markdown(
    markdown: str,
    position_after: str,
    image_url: str,
    alt_text: str = "Infographic",
) -> str:
    """Insert an infographic image after the paragraph matching position_after snippet.

    Uses surgical string insertion to preserve original formatting and links.
    """
    snippet = (position_after or "").strip()[:60]
    if not snippet:
        return markdown + f"\n\n![{alt_text}]({image_url})"

    sep_matches = list(re.finditer(r"\n\s*\n", markdown))
    blocks: list[tuple[int, int, str]] = []
    for i in range(len(sep_matches) + 1):
        start = sep_matches[i - 1].end() if i > 0 else 0
        end = sep_matches[i].start() if i < len(sep_matches) else len(markdown)
        blocks.append((start, end, markdown[start:end]))

    for block_idx, (_, end, content) in enumerate(blocks):
        block_clean = content.replace("\n", " ").strip()
        normalized = re.sub(r"\[([^\]]*)\]\([^)]+\)", r"\1", block_clean)
        if snippet in block_clean or snippet in normalized or block_clean.startswith(snippet[:30]) or normalized.startswith(snippet[:30]):
            pos = sep_matches[block_idx].end() if block_idx < len(sep_matches) else len(markdown)
            img_md = f"![{alt_text}]({image_url})"
            return markdown[:pos] + f"{img_md}\n\n" + markdown[pos:]

    logger.warning("[IMAGES] Could not find position_after snippet, appended infographic at end")
    return markdown + f"\n\n![{alt_text}]({image_url})"


def analyze_image_placements(article_markdown: str, max_images: int = 4) -> list[dict[str, Any]]:
    """Use Anthropic to suggest where to place images in the article."""
    from anthropic import Anthropic as _Anthropic
    placement_prompt = get_prompt("image_placement")
    model = (
        get_prompt_llm_kwargs("image_placement").get("model")
        or get_config("image_placement_model", "claude-haiku-4-5-20251001")
    )
    if not model.startswith("claude"):
        logger.warning(
            "[IMAGES] image_placement_model=%r is not Anthropic, using claude-haiku-4-5-20251001",
            model,
        )
        model = "claude-haiku-4-5-20251001"
    max_tokens = get_prompt_llm_kwargs("image_placement").get("max_tokens", 1024)

    anthropic_client = _Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    prompt = f"""{placement_prompt}

Article:
---
{article_markdown[:12000]}
---"""

    response = anthropic_client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.content[0].text.strip()
    # Extract JSON from response (handle markdown code blocks)
    if "```" in content:
        match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", content, re.DOTALL)
        if match:
            content = match.group(1)
    try:
        placements = json.loads(content)
    except json.JSONDecodeError:
        return []
    if not isinstance(placements, list):
        return []
    return placements[:max_images]


def generate_image_with_imagen(prompt: str, aspect_ratio: str = "3:4") -> bytes:
    """Generate image using Google Imagen via Gemini API."""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        img_config = types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio=aspect_ratio,
        )
        response = client.models.generate_images(
            model="imagen-3.0-generate-002",
            prompt=prompt,
            config=img_config,
        )
        if response.generated_images:
            img = response.generated_images[0]
            if hasattr(img, "image") and hasattr(img.image, "image_bytes"):
                return img.image.image_bytes
            if hasattr(img, "image_bytes"):
                return img.image_bytes
    except Exception as e:
        raise RuntimeError(f"Imagen generation failed: {e}") from e
    raise RuntimeError("No image returned from Imagen")


def generate_image_with_dalle(prompt: str, size: str = "1024x1024") -> bytes:
    """Fallback: Generate image using OpenAI DALL-E 3."""
    client = _get_openai()
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=size,
        quality="standard",
        n=1,
    )
    url = response.data[0].url
    if not url:
        raise RuntimeError("No image URL from DALL-E")
    import httpx
    with httpx.Client() as http:
        r = http.get(url)
        r.raise_for_status()
        return r.content


def generate_image_with_gemini_text_only(prompt: str, model: Optional[str] = None) -> bytes:
    """Generate image using Gemini image model from text only (no references)."""
    from google import genai
    from google.genai import types

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set")
    client = genai.Client(api_key=api_key)
    if model is None:
        model = get_config("image_model_hero", "gemini-2.5-flash-image")
    GEMINI_IMAGE_MODELS = {"gemini-2.5-flash-image", "gemini-3-pro-image-preview"}
    if model not in GEMINI_IMAGE_MODELS:
        logger.warning(
            "[IMAGES] image_model_hero=%r is not image-capable, using gemini-2.5-flash-image",
            model,
        )
        model = "gemini-2.5-flash-image"
    config = types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"])
    response = client.models.generate_content(model=model, contents=prompt, config=config)
    for part in response.candidates[0].content.parts:
        if part.inline_data and part.inline_data.mime_type.startswith("image/"):
            return part.inline_data.data
    raise RuntimeError("No image returned from Gemini")


def generate_image(prompt: str, aspect_ratio: str = "3:4") -> bytes:
    """Generate image based on configured image_model_generic."""
    model = get_config("image_model_generic", "imagen-3.0-generate-002")
    if model in ("gemini-2.5-flash-image", "gemini-3-pro-image-preview"):
        return generate_image_with_gemini_text_only(prompt, model=model)
    if model == "dall-e-3":
        return generate_image_with_dalle(prompt)
    try:
        return generate_image_with_imagen(prompt, aspect_ratio)
    except Exception:
        return generate_image_with_dalle(prompt)


def upload_to_supabase(image_bytes: bytes, filename: str, bucket: str = "article-images") -> str:
    """Upload image to Supabase Storage and return public URL."""
    client = _get_supabase_client()
    path = f"{uuid.uuid4()}_{filename}"

    # Detect content type
    content_type = "image/png"
    if image_bytes[:3] == b"\xff\xd8\xff":
        content_type = "image/jpeg"
    elif image_bytes[:4] == b"RIFF":
        content_type = "image/webp"

    client.storage.from_(bucket).upload(
        path,
        image_bytes,
        file_options={"content-type": content_type},
    )
    public_url = client.storage.from_(bucket).get_public_url(path)
    # Ensure https (some Supabase client versions return http)
    if public_url.startswith("http://"):
        public_url = public_url.replace("http://", "https://", 1)
    return public_url


def inject_images_into_markdown(
    markdown: str,
    placements: list[dict],
    image_urls: list[str],
) -> str:
    """Insert image markdown at the specified positions.

    Uses surgical string insertion: finds paragraph boundaries, locates matching blocks,
    and inserts image markdown at exact positions. Does NOT split/join the document,
    preserving all original formatting (links, code blocks, extra newlines, etc.).
    """
    if len(placements) != len(image_urls):
        raise ValueError("Placements and image_urls length mismatch")

    # Find paragraph boundaries without altering the markdown
    sep_matches = list(re.finditer(r"\n\s*\n", markdown))
    blocks: list[tuple[int, int, str]] = []
    for i in range(len(sep_matches) + 1):
        start = sep_matches[i - 1].end() if i > 0 else 0
        end = sep_matches[i].start() if i < len(sep_matches) else len(markdown)
        content = markdown[start:end]
        blocks.append((start, end, content))

    # Map: block_index -> insert_position (after separator following this block)
    def insert_pos_for_block(block_idx: int) -> int:
        if block_idx < len(sep_matches):
            return sep_matches[block_idx].end()
        return len(markdown)

    # Find (insert_position, image_md) for each placement
    insertions: list[tuple[int, str]] = []
    placement_idx = 0

    for block_idx, (_, _, content) in enumerate(blocks):
        if placement_idx >= len(placements):
            break
        p = placements[placement_idx]
        snippet = (p.get("paragraph_after") or "").strip()[:60]
        alt = p.get("alt_text", "Image")
        block_clean = content.replace("\n", " ").strip()
        # Match: exact substring, or relaxed match (snippet without link syntax)
        normalized = re.sub(r"\[([^\]]*)\]\([^)]+\)", r"\1", block_clean)
        matches = (
            snippet
            and (
                snippet in block_clean
                or snippet in normalized
                or block_clean.startswith(snippet[:30])
                or normalized.startswith(snippet[:30])
            )
        )
        if matches:
            pos = insert_pos_for_block(block_idx)
            img_md = f"![{alt}]({image_urls[placement_idx]})"
            insertions.append((pos, img_md))
            placement_idx += 1

    # Insert from end to start so indices stay valid
    insertions.sort(key=lambda x: x[0], reverse=True)
    result = markdown
    for pos, img_md in insertions:
        # pos is already past the paragraph's \n\n; insert image before next block
        result = result[:pos] + f"{img_md}\n\n" + result[pos:]

    # Append any unplaced images at end
    while placement_idx < len(image_urls):
        p = placements[placement_idx]
        result += f"\n\n![{p.get('alt_text', 'Image')}]({image_urls[placement_idx]})"
        placement_idx += 1

    return result


def generate_and_inject_images(
    article_markdown: str,
    max_images: int = 4,
    use_supabase: bool = True,
    inject: bool = True,
) -> tuple[str, list[dict]]:
    """
    Full pipeline: analyze placements, generate images, upload, optionally inject.
    If inject=False (approval mode), returns (original_markdown, image_records) - no injection.
    User can then call inject_approved_image for each approved image.
    Returns (updated_markdown, list of image records for DB).
    """
    placements = analyze_image_placements(article_markdown, max_images)
    if not placements:
        return article_markdown, []

    image_records = []
    image_urls = []

    for i, p in enumerate(placements):
        prompt = p.get("image_prompt", "Professional illustration")
        img_bytes = generate_image(prompt)
        if use_supabase:
            url = upload_to_supabase(img_bytes, f"img_{i}.png")
        else:
            # For testing without Supabase: use data URL (large but works)
            b64 = base64.b64encode(img_bytes).decode()
            url = f"data:image/png;base64,{b64}"
        image_urls.append(url)
        image_records.append({
            "position": i,
            "url": url,
            "alt_text": p.get("alt_text"),
            "prompt_used": prompt,
        })

    if inject:
        updated = inject_images_into_markdown(article_markdown, placements, image_urls)
        return updated, image_records
    return article_markdown, image_records


def inject_approved_image_at_position(
    markdown: str,
    position: int,
    image_url: str,
    alt_text: str = "Image",
) -> str:
    """
    Inject a single approved image at a specific placement position.
    Reuses placement analysis to find where to insert; position is 0-based index.
    Returns updated markdown.
    """
    placements = analyze_image_placements(markdown, max_images=position + 1)
    if position >= len(placements):
        raise ValueError(
            f"Position {position} out of range. Article has {len(placements)} suggested image placements (0-based)."
        )
    single_placement = [placements[position]]
    single_url = [image_url]
    return inject_images_into_markdown(markdown, single_placement, single_url)
