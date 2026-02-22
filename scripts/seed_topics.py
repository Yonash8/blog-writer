"""Seed the topics table with initial article ideas."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()


DEFAULT_TOPICS = [
    {"title": "Why LLMs fail intermittently in production", "description": "Explore common causes of LLM reliability issues and how to debug them.", "keywords": ["llm", "reliability", "production", "debugging"]},
    {"title": "Building a RAG pipeline from scratch", "description": "Step-by-step guide to implementing retrieval-augmented generation.", "keywords": ["rag", "embeddings", "vector-db", "llm"]},
    {"title": "Prompt engineering best practices", "description": "Techniques for writing effective prompts that improve output quality.", "keywords": ["prompts", "llm", "best-practices"]},
    {"title": "Cost optimization for AI applications", "description": "Strategies to reduce API costs when building AI-powered products.", "keywords": ["cost", "openai", "anthropic", "optimization"]},
    {"title": "Evaluating LLM outputs at scale", "description": "How to build evaluation pipelines for production ML systems.", "keywords": ["evaluation", "llm", "mlops"]},
]


def main():
    """Seed topics into the database."""
    from src.db import create_topic, get_topics

    existing = get_topics(limit=100)
    existing_titles = {t["title"].lower() for t in existing}

    added = 0
    for topic in DEFAULT_TOPICS:
        if topic["title"].lower() in existing_titles:
            print(f"Skip (exists): {topic['title']}")
            continue
        create_topic(
            title=topic["title"],
            description=topic.get("description"),
            keywords=topic.get("keywords"),
        )
        print(f"Added: {topic['title']}")
        added += 1

    print(f"\nDone. Added {added} topics.")


if __name__ == "__main__":
    main()
