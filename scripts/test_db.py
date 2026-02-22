"""Quick test: DB write with service role key. Run: python scripts/test_db.py"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

def main():
    print("Testing Supabase connection...")
    print("  SUPABASE_URL:", "set" if os.getenv("SUPABASE_URL") else "MISSING")
    print("  SUPABASE_SERVICE_ROLE_KEY:", "set" if os.getenv("SUPABASE_SERVICE_ROLE_KEY") else "MISSING")
    if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_SERVICE_ROLE_KEY"):
        print("FAIL: Add SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY to .env")
        sys.exit(1)

    from src.db import create_topic, get_topics

    # Write test
    t = create_topic(title="Test topic from script", description="DB write test")
    print(f"  Created topic: {t['id']}")

    # Read test
    topics = get_topics(limit=5)
    print(f"  Read {len(topics)} topics")
    for t in topics[:2]:
        print(f"    - {t.get('title', '?')}")

    print("OK - DB read/write works with service role key")

if __name__ == "__main__":
    main()
