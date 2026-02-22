"""Call Tavily with the topic and save full response to JSON. Run: python scripts/tavily_debug.py"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv()

def main():
    from tavily import TavilyClient

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("ERROR: TAVILY_API_KEY not set in .env")
        sys.exit(1)

    topic = "How do you debug agentic workflows?"
    query = f"Find authoritative and relevant sources explaining {topic}. Prefer high-quality articles, engineering blogs, academic papers, postmortems, and official documentation."

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=20,
        chunks_per_source=3,
        include_raw_content=True,
        include_answer=False,
    )

    # Save full raw response
    out_path = Path(__file__).resolve().parent.parent / "tavily_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(response, f, indent=2, ensure_ascii=False)

    n = len(response.get("results", []))
    print(f"Saved {n} results to {out_path}")

if __name__ == "__main__":
    main()
