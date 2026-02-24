#!/usr/bin/env python3
"""Inspect PromptLayer workflow and prompt template to see expected input variables."""

import json
import os
import sys

# Load .env from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv

load_dotenv()

import httpx

API_KEY = os.getenv("PROMPTLAYER_API_KEY")
WORKFLOW_NAME = os.getenv("PROMPTLAYER_ORCHESTRATOR_WORKFLOW", "blog_writer_master_system_core")
PROMPT_NAME = "blog_writer/master_system_core"


def main():
    if not API_KEY:
        print("ERROR: PROMPTLAYER_API_KEY not set")
        sys.exit(1)

    headers = {"X-API-KEY": API_KEY, "Content-Type": "application/json"}

    with httpx.Client(timeout=15.0) as client:
        # 1. List workflows to find our workflow and get its ID
        print("=== Listing Workflows ===")
        r = client.get(
            "https://api.promptlayer.com/workflows",
            headers=headers,
            params={"per_page": 100},
        )
        if r.status_code != 200:
            print(f"List workflows failed: {r.status_code} {r.text[:500]}")
        else:
            data = r.json()
            items = data.get("items", [])
            for w in items:
                if WORKFLOW_NAME in w.get("name", "") or "master_system" in w.get("name", "").lower():
                    print(json.dumps(w, indent=2))
            if not any("master_system" in w.get("name", "").lower() for w in items):
                print("(No matching workflow found in list)")

        # 2. Get prompt template structure - REST endpoint our app uses
        print("\n=== Get Prompt Template (blog_writer/master_system_core) ===")
        r2 = client.get(
            "https://api.promptlayer.com/rest/get-prompt-template",
            headers=headers,
            params={"prompt_name": PROMPT_NAME},
        )
        if r2.status_code != 200:
            print(f"Get prompt failed: {r2.status_code} {r2.text[:500]}")
        else:
            data2 = r2.json()
            # Show structure without full content
            template = data2.get("prompt_template") or data2.get("template") or {}
            messages = template.get("messages") or []
            print(f"Template keys: {list(template.keys())}")
            print(f"Template input_variables: {template.get('input_variables')}")
            print(f"Messages count: {len(messages)}")
            for i, m in enumerate(messages):
                role = m.get("role", m.get("type", "?"))
                prompt_obj = m.get("prompt") or {}
                content = prompt_obj.get("template") or m.get("content") or ""
                # Extract variable names: {{ var }} or { var }
                import re
                jinja_vars = set(re.findall(r"\{\{\s*(\w+)\s*\}\}", content))
                fstr_vars = set(re.findall(r"\{(\w+)\}", content))
                vars_found = jinja_vars or fstr_vars
                preview = (content[:200] + "…") if len(content) > 200 else content
                print(f"  Message {i}: role={role}, vars={vars_found}")
                print(f"    Preview: {preview!r}")

        # 3. Try run with minimal inputs, then with PLAYBOOKS included
        print("\n=== Test Run A: user, context, history (no PLAYBOOKS) ===")
        test_vars_a = {
            "user": "Hello",
            "context": "(no additional context)",
            "history": "(no prior messages)",
        }
        r3 = client.post(
            f"https://api.promptlayer.com/workflows/{WORKFLOW_NAME.replace(' ', '%20')}/run",
            headers=headers,
            json={"input_variables": test_vars_a, "return_all_outputs": True},
            timeout=60.0,
        )
        print(f"Run response: {r3.status_code}")
        exec_id = None
        if r3.status_code in (201, 202):
            d = r3.json()
            exec_id = d.get("workflow_version_execution_id")
            print(f"  exec_id: {exec_id}")

        # Poll for results (like our pipeline does)
        if exec_id:
            import time
            print("  Polling for results (up to 30s)...")
            for _ in range(6):
                time.sleep(5)
                r4 = client.get(
                    "https://api.promptlayer.com/workflow-version-execution-results",
                    headers=headers,
                    params={"workflow_version_execution_id": exec_id, "return_all_outputs": True},
                    timeout=30.0,
                )
                if r4.status_code == 202:
                    print("    Still running...")
                    continue
                if r4.status_code == 200:
                    results = r4.json()
                    for node_name, node_data in results.items():
                        if isinstance(node_data, dict):
                            status = node_data.get("status", "?")
                            err = node_data.get("error_message") or ""
                            print(f"    Node {node_name}: status={status}")
                            if err:
                                print(f"      ERROR: {err[:300]}")
                    break

        # 4. Test with PLAYBOOKS from our prompts_loader (template expects it)
        print("\n=== Test Run B: user, context, history, PLAYBOOKS (from get_prompt) ===")
        from src.prompts_loader import get_prompt
        playbooks_content = get_prompt("playbooks") or "## Playbooks\n(no playbooks loaded)"
        test_vars_b = {
            "user": "Hello",
            "context": "(no additional context)",
            "history": "(no prior messages)",
            "PLAYBOOKS": playbooks_content,
        }
        r5 = client.post(
            f"https://api.promptlayer.com/workflows/{WORKFLOW_NAME.replace(' ', '%20')}/run",
            headers=headers,
            json={"input_variables": test_vars_b, "return_all_outputs": True},
            timeout=60.0,
        )
        print(f"Run response: {r5.status_code}")
        if r5.status_code in (201, 202):
            exec_id2 = r5.json().get("workflow_version_execution_id")
            print("  Polling...")
            import time
            for _ in range(6):
                time.sleep(5)
                r6 = client.get(
                    "https://api.promptlayer.com/workflow-version-execution-results",
                    headers=headers,
                    params={"workflow_version_execution_id": exec_id2, "return_all_outputs": True},
                    timeout=30.0,
                )
                if r6.status_code == 200:
                    results = r6.json()
                    for node_name, node_data in results.items():
                        if isinstance(node_data, dict):
                            status = node_data.get("status", "?")
                            val = node_data.get("value")
                            print(f"  Node {node_name}: status={status}")
                            if status in ("SUCCESS", "FAILURE", "FAILED") and node_data.get("error_message"):
                                print(f"    ERROR: {str(node_data.get('error_message'))[:400]}")
                    break
                elif r6.status_code == 202:
                    print("    Still running...")


if __name__ == "__main__":
    main()
