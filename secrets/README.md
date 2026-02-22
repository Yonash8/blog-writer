# Secrets folder

Place sensitive credential files here. **This folder is git-ignored** so they are never committed.

Example:
- `secrets/google-service-account.json` — Google Docs/Drive API key

Then in `.env`:
```
GOOGLE_APPLICATION_CREDENTIALS=secrets/google-service-account.json
```
