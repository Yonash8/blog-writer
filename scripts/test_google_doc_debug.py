"""Debug script: verify Google credentials and try both Docs and Drive API."""

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
os.chdir(_project_root)

from dotenv import load_dotenv

load_dotenv()

def main():
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path:
        print("ERROR: GOOGLE_APPLICATION_CREDENTIALS not set")
        return
    if not os.path.isabs(creds_path):
        creds_path = os.path.abspath(creds_path)
    if not os.path.exists(creds_path):
        print(f"ERROR: File not found: {creds_path}")
        return

    import json
    with open(creds_path) as f:
        key_data = json.load(f)
    project_id = key_data.get("project_id", "?")
    client_email = key_data.get("client_email", "?")
    print(f"Project: {project_id}")
    print(f"Service account: {client_email}")
    print()

    scopes = [
        "https://www.googleapis.com/auth/documents",
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/drive.file",
    ]
    creds = service_account.Credentials.from_service_account_file(creds_path, scopes=scopes)

    # Try 1: Docs API - create blank doc
    print("Trying Docs API (documents.create)...")
    try:
        docs = build("docs", "v1", credentials=creds)
        doc = docs.documents().create(body={"title": "Test"}).execute()
        print(f"  SUCCESS! Doc ID: {doc.get('documentId')}")
        print(f"  URL: https://docs.google.com/document/d/{doc.get('documentId')}/edit")
    except Exception as e:
        print(f"  FAILED: {e}")
        if hasattr(e, "resp") and e.resp:
            print(f"  Response: {e.resp}")

    # Try 2: Drive API - create doc (alternative method)
    print("\nTrying Drive API (files.create with doc mimetype)...")
    try:
        drive = build("drive", "v3", credentials=creds)
        file = drive.files().create(
            body={"name": "Test from Drive API", "mimeType": "application/vnd.google-apps.document"},
            fields="id,webViewLink",
        ).execute()
        print(f"  SUCCESS! File ID: {file.get('id')}")
        print(f"  URL: {file.get('webViewLink', 'N/A')}")
    except Exception as e:
        print(f"  FAILED: {e}")
        if hasattr(e, "resp") and e.resp:
            print(f"  Response: {e.resp}")

if __name__ == "__main__":
    main()
