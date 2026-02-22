"""Create the article-images storage bucket. Run after supabase db push."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

def main():
    from supabase import create_client
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
    if not url or not key:
        print("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env")
        sys.exit(1)
    client = create_client(url, key)
    try:
        client.storage.create_bucket("article-images", options={"public": True})
        print("Created bucket: article-images")
    except Exception as e:
        if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
            print("Bucket article-images already exists")
        else:
            raise

if __name__ == "__main__":
    main()
