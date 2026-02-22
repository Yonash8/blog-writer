"""Create the 'heros' and 'infographics' reference image buckets in Supabase Storage.

Run after supabase db push. Upload style reference images to these buckets
via the Supabase dashboard or the Storage API.

- heros: PromptLayer Bot mascot / monochromatic blue palette style references
- infographics: Line weight, font style, and color tone style examples
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv

load_dotenv()

BUCKETS = [
    ("heros", "Style references for hero image generation (mascot / blue palette)"),
    ("infographics", "Style references for infographic generation (line weight, fonts, colors)"),
]


def main():
    from supabase import create_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
    if not url or not key:
        print("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env")
        sys.exit(1)

    client = create_client(url, key)

    for bucket_name, description in BUCKETS:
        try:
            client.storage.create_bucket(
                bucket_name,
                options={"public": True, "file_size_limit": 10 * 1024 * 1024},  # 10 MB
            )
            print(f"Created bucket: {bucket_name} — {description}")
        except Exception as e:
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                print(f"Bucket '{bucket_name}' already exists")
            else:
                print(f"Error creating bucket '{bucket_name}': {e}")
                raise

    print("\nDone. Upload reference images via Supabase Dashboard -> Storage.")


if __name__ == "__main__":
    main()
