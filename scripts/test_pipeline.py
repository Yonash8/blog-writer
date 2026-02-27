"""Test the core pipeline: topic -> research -> article."""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()


def main():
    """Run pipeline with a test topic."""
    from src.pipeline import write_article

    topic = os.environ.get("TEST_TOPIC", "Why LLMs fail intermittently in production")
    print(f"Testing pipeline with topic: {topic}")
    print("-" * 60)

    try:
        result = write_article(topic)
        article = result.get("article", "")
        print("SUCCESS - Article generated:")
        print("-" * 60)
        print(article[:2000] + ("..." if len(article) > 2000 else ""))
    except Exception as e:
        print(f"FAILED: {e}")
        raise


if __name__ == "__main__":
    main()
