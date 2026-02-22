"""Quick test for Create Google Doc. Run from project root: python scripts/test_google_doc.py"""

import os
import sys

# Ensure project root is on path and cwd
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
os.chdir(_project_root)

from dotenv import load_dotenv

load_dotenv()

def main():
    from src.db import create_article, get_or_create_topic_for_article
    from src.tools import google_docs_tool

    print("Creating test article...")
    topic_rec = get_or_create_topic_for_article("Test Article")
    art = create_article(
        channel="test",
        channel_user_id="script",
        topic_id=topic_rec["id"],
        content="# Test Article\n\nThis is a short test to verify Google Doc creation.\n\nIt has **bold**, [links](https://google.com), and bullets.",
    )
    article_id = art["id"]
    print(f"Article ID: {article_id}")

    print("Creating Google Doc...")
    result = google_docs_tool(action="create", article_id=article_id)
    if not result.get("success"):
        print(f"Error: {result.get('error')}")
        sys.exit(1)
    print(f"Done! Document URL: {result['document_url']}")
    return result["document_url"]

if __name__ == "__main__":
    main()
