"""Seed the topics table with LLM eval topics (cluster = group)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

# Cluster -> group. Status column omitted (not in schema).
TOPICS = [
    ("Cross-Functional & No-Code Participation", "How do PMs contribute to LLM evals without touching code?"),
    ("Eval Dataset Design, Ownership & Governance", "Dataset versioning vs experiment tracking"),
    ("Eval Dataset Design, Ownership & Governance", "How do large orgs standardize LLM benchmarks?"),
    ("Eval Dataset Design, Ownership & Governance", "How do teams audit changes to evaluation data?"),
    ("Eval Dataset Design, Ownership & Governance", "How often should LLM eval datasets be updated?"),
    ("Eval Dataset Design, Ownership & Governance", "How to design good LLM evaluation datasets"),
    ("Eval Dataset Design, Ownership & Governance", "Versioning datasets for machine learning best practices"),
    ("Eval Dataset Design, Ownership & Governance", "Who should own LLM evaluation datasets?"),
    ("Eval Dataset Design, Ownership & Governance", "Why do LLM evals break down across teams?"),
    ("Failure Detection, Debugging & Root Cause Analysis", "How do teams identify failure cases in production LLM systems?"),
    ("Failure Detection, Debugging & Root Cause Analysis", "How do you debug agentic workflows?"),
    ("Failure Detection, Debugging & Root Cause Analysis", "How do you trace multi-step LLM workflows?"),
    ("Failure Detection, Debugging & Root Cause Analysis", "How to collect bad LLM outputs automatically"),
    ("Failure Detection, Debugging & Root Cause Analysis", "Which step in the chain caused the failure?"),
    ("Failure Detection, Debugging & Root Cause Analysis", "Why is this LLM failing intermittently?"),
    ("Foundations of LLM Evaluation & \"Ground Truth\"", "How do teams evaluate LLM performance reliably?"),
    ("Foundations of LLM Evaluation & \"Ground Truth\"", "How many examples do you need for an LLM evaluation?"),
    ("Foundations of LLM Evaluation & \"Ground Truth\"", "How many examples do you need for LLM evals?"),
    ("Foundations of LLM Evaluation & \"Ground Truth\"", "Questions practitioners ask before they trust their evals"),
    ("Foundations of LLM Evaluation & \"Ground Truth\"", "Reproducible LLM evaluations"),
    ("Foundations of LLM Evaluation & \"Ground Truth\"", "What does 'ground truth' even mean for generative AI?"),
    ("Foundations of LLM Evaluation & \"Ground Truth\"", "What makes an eval dataset statistically meaningful?"),
    ("Foundations of LLM Evaluation & \"Ground Truth\"", "Why are our eval results not reproducible?"),
    ("Labeling, Judgment & Disagreement", "How do teams label LLM outputs at scale?"),
    ("Labeling, Judgment & Disagreement", "How do teams validate synthetic examples?"),
    ("Labeling, Judgment & Disagreement", "How do you deal with disagreement in LLM output labeling?"),
    ("Labeling, Judgment & Disagreement", "Human vs LLM-as-judge: when does each make sense?"),
    ("Labeling, Judgment & Disagreement", "LLM-as-judge best practices"),
    ("Observability, Logging & Tracing", "Best practices for redacting LLM traces"),
    ("Observability, Logging & Tracing", "How do you observe LLM systems in production?"),
    ("Observability, Logging & Tracing", "Logging and tracing LLM failures best practices"),
    ("Observability, Logging & Tracing", "Logging LLM inputs without storing PII"),
    ("Observability, Logging & Tracing", "Tracing vs logging for LLMs"),
    ("Observability, Logging & Tracing", "What should we log from LLM calls?"),
    ("Privacy, Security & Sensitive Data Handling", "Can we use production data for evals safely?"),
    ("Privacy, Security & Sensitive Data Handling", "Handling sensitive data in LLM evaluation datasets"),
    ("Privacy, Security & Sensitive Data Handling", "Logging LLM inputs without storing PII"),
    ("Production Monitoring, Drift & Long-Term Quality", "Dataset drift in LLM systems"),
    ("Production Monitoring, Drift & Long-Term Quality", "How do teams monitor LLM quality over time?"),
    ("Production Monitoring, Drift & Long-Term Quality", "How do teams prevent overfitting to eval datasets?"),
    ("Production Monitoring, Drift & Long-Term Quality", "Why do LLM eval scores stop correlating with prod performance?"),
    ("Production Monitoring, Drift & Long-Term Quality", "Why do our LLM eval scores look good but prod feels worse?"),
    ("Synthetic Data vs Real Data in Evals", "Does synthetic data actually improve LLM performance?"),
    ("Synthetic Data vs Real Data in Evals", "Risks of synthetic data in LLM evaluation"),
    ("Synthetic Data vs Real Data in Evals", "When should you use synthetic vs real data for evals?"),
]

# Dedupe: same title -> use first occurrence (first group wins)
_seen = set()
DEDUPED = []
for group, title in TOPICS:
    key = title.strip().lower()
    if key not in _seen:
        _seen.add(key)
        DEDUPED.append((group, title))


def main():
    """Seed eval topics into the database."""
    from src.db import create_topic, get_topics

    existing = get_topics(limit=500)
    existing_titles = {t["title"].strip().lower() for t in existing}

    added = 0
    for group, title in DEDUPED:
        if title.strip().lower() in existing_titles:
            print(f"Skip (exists): {title}")
            continue
        create_topic(title=title.strip(), group=group)
        print(f"Added: [{group}] {title}")
        added += 1

    print(f"\nDone. Added {added} topics.")


if __name__ == "__main__":
    main()
