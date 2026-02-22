"""Quick test for image generation (Gemini). Run: python scripts/test_image_gen.py [model]
   Optional: pass gemini-2.5-flash-image or gemini-3-pro-image-preview to force model."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def main():
    model_override = sys.argv[1] if len(sys.argv) > 1 else None
    print("Testing image generation (Gemini)...")
    if not os.getenv("GOOGLE_API_KEY"):
        print("FAIL: GOOGLE_API_KEY not set in .env")
        sys.exit(1)
    if model_override:
        print(f"  Model override: {model_override}")

    from src.images import generate_image_with_gemini_text_only

    prompt = "A simple illustration of a coffee cup on a wooden table, minimalist flat design"
    print(f"Calling generate_image_with_gemini_text_only (prompt: {prompt[:50]}...)...")
    try:
        img_bytes = generate_image_with_gemini_text_only(
            prompt, model=model_override if model_override else None
        )
        out_path = Path(__file__).parent.parent / "test_output_image.png"
        out_path.write_bytes(img_bytes)
        print(f"  OK - Generated {len(img_bytes)} bytes, saved to {out_path}")
    except Exception as e:
        print(f"  FAIL: {e}")
        raise


if __name__ == "__main__":
    main()
