#!/usr/bin/env python3
"""Inspect the raw PromptLayer metadata agent output.

Runs the metadata agent with a minimal test article and dumps the FULL raw
PromptLayer response so we can see exactly what field names and nesting the
workflow returns.

Usage:
    python scripts/inspect_metadata_payload.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

import httpx

API_KEY = os.getenv("PROMPTLAYER_API_KEY")
WORKFLOW_NAME = os.getenv("PROMPTLAYER_METADATA_WORKFLOW", "metadata agent")

SAMPLE_ARTICLE = """# How Solar Panels Work

Solar panels convert sunlight into electricity using the photovoltaic effect.

## The Photovoltaic Effect

When photons from sunlight hit a silicon cell, they knock electrons loose,
creating a flow of electricity. This direct current (DC) is then converted
to alternating current (AC) by an inverter.

## Key Benefits

- Reduces electricity bills by up to 70%
- Zero emissions during operation
- Low maintenance requirements
- 25-30 year lifespan

## Installation Considerations

A typical home installation requires 20-25 panels rated at 400W each.
South-facing roofs with minimal shading yield the best results.
"""


def main():
    if not API_KEY:
        print("ERROR: PROMPTLAYER_API_KEY not set")
        sys.exit(1)

    headers = {"X-API-KEY": API_KEY, "Content-Type": "application/json"}
    print(f"Running metadata agent workflow: '{WORKFLOW_NAME}'")
    print("=" * 60)

    with httpx.Client(timeout=60.0) as client:
        # Trigger the workflow
        r = client.post(
            f"https://api.promptlayer.com/workflows/{WORKFLOW_NAME.replace(' ', '%20')}/run",
            headers=headers,
            json={"input_variables": {"article": SAMPLE_ARTICLE}, "return_all_outputs": True},
            timeout=60.0,
        )
        print(f"Trigger response: HTTP {r.status_code}")
        if r.status_code not in (200, 201, 202):
            print("ERROR:", r.text[:1000])
            sys.exit(1)

        data = r.json()
        exec_id = data.get("workflow_version_execution_id")
        if not exec_id:
            print("ERROR: No exec_id in response:", data)
            sys.exit(1)
        print(f"exec_id: {exec_id}")
        print("Polling for results...")

        # Poll
        for poll_i in range(40):
            time.sleep(10)
            res = client.get(
                "https://api.promptlayer.com/workflow-version-execution-results",
                headers=headers,
                params={"workflow_version_execution_id": exec_id, "return_all_outputs": True},
                timeout=30.0,
            )
            if res.status_code == 202:
                print(f"  poll #{poll_i + 1}: still running...")
                continue
            if res.status_code != 200:
                print(f"ERROR polling: HTTP {res.status_code} {res.text[:500]}")
                sys.exit(1)

            results = res.json()
            print("\n=== RAW FULL RESPONSE ===")
            print(json.dumps(results, indent=2, default=str))

            print("\n=== PER-NODE SUMMARY ===")
            for node_name, node_data in results.items():
                if not isinstance(node_data, dict):
                    continue
                status = node_data.get("status", "?")
                is_output = node_data.get("is_output_node", False)
                value = node_data.get("value")
                err = node_data.get("error_message") or node_data.get("raw_error_message") or ""
                print(f"\n  Node: {node_name!r}")
                print(f"    status={status}, is_output_node={is_output}")
                if err:
                    print(f"    ERROR: {str(err)[:300]}")
                if value is not None:
                    if isinstance(value, dict):
                        print(f"    value type: dict, keys={list(value.keys())}")
                        # If there's a nested "value", show its keys too
                        inner = value.get("value")
                        if isinstance(inner, dict):
                            print(f"    value['value'] keys: {list(inner.keys())}")
                            print(f"    value['value'] content: {json.dumps(inner, indent=6, default=str)[:800]}")
                        elif isinstance(inner, str):
                            print(f"    value['value'] (str): {inner[:400]}")
                        else:
                            print(f"    value content: {json.dumps(value, indent=6, default=str)[:800]}")
                    elif isinstance(value, str):
                        print(f"    value type: str, len={len(value)}")
                        print(f"    value[:600]: {value[:600]}")
                    else:
                        print(f"    value type: {type(value).__name__}, value: {value!r}")
            break
        else:
            print("Timed out after 40 polls")


if __name__ == "__main__":
    main()
