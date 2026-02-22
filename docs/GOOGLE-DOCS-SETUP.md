# Google Docs API Setup

To use **Create Google Doc** from approved drafts, you need a Google Cloud service account and its JSON key.

## Steps

1. **Go to [Google Cloud Console](https://console.cloud.google.com/)** and create or select a project.

2. **Enable APIs**
   - APIs & Services → Library
   - Search and enable:
     - **Google Docs API**
     - **Google Drive API**

3. **Create a service account**
   - APIs & Services → Credentials → Create Credentials → Service Account
   - Name it (e.g. `blog-writer-docs`) → Create
   - Optional: add role (e.g. Editor) → Done

4. **Download a JSON key**
   - APIs & Services → Credentials → click the service account you created
   - Open the **Keys** tab
   - Add Key → Create new key → choose **JSON** → Create
   - A `.json` file downloads to your computer (e.g. `project-name-abc123.json`)

5. **Configure your app**
   - Put the JSON file in the project’s `secrets/` folder (git-ignored)
   - Add to `.env`:
   ```env
   GOOGLE_APPLICATION_CREDENTIALS=secrets/your-service-account-key.json
   ```

## Notes

- Docs are created in the service account’s Drive. The link is public (“anyone with link can view”) so you can open and copy the doc to your Drive.
- Keep the JSON key private and do not commit it to version control.

## Security

Downloaded service account keys can be a security risk if leaked. For production or CI, consider using [Workload Identity Federation](https://cloud.google.com/iam/docs/workload-identity-federation) instead of storing JSON keys. The JSON key approach above is simpler for local development only.

---

## If you get "403 - caller does not have permission"

Do this **from scratch** in a new project:

### A. Create a new project

1. [Google Cloud Console](https://console.cloud.google.com/)
2. Top bar: click the project dropdown → **New Project**
3. Name it `blog-writer` (or anything)
4. Click **Create** → wait for it to finish
5. **Select that new project** in the dropdown (important)

### B. Enable both APIs

1. Left menu: **APIs & Services** → **Library**
2. Search: `Google Docs API` → open it → **Enable**
3. Back to Library, search: `Google Drive API` → open it → **Enable**

### C. Create service account and key

1. **APIs & Services** → **Credentials**
2. **Create Credentials** → **Service Account**
3. Name: `blog-writer-docs` → **Create and Continue** → **Done**
4. Click the new service account in the list
5. **Keys** tab → **Add Key** → **Create new key**
6. Choose **JSON** → **Create** → file downloads

### D. Wire it up

1. Move the downloaded JSON into your project’s `secrets/` folder
2. In `.env` set:
   ```
   GOOGLE_APPLICATION_CREDENTIALS=secrets/your-downloaded-filename.json
   ```
3. Run the test:
   ```
   python scripts/test_google_doc.py
   ```

### E. Share a folder (fixes "storage quota exceeded")

The service account has no Drive space. Create a folder in **your** Google Drive and share it:

1. [Google Drive](https://drive.google.com) → New → Folder → name it `Blog Writer Docs`
2. Right‑click the folder → Share
3. Add: `blog-writer-docs@blog-writer-487316.iam.gserviceaccount.com` (your service account email from the JSON)
4. Give it **Editor** access → Share
5. Open the folder → copy the folder ID from the URL:  
   `https://drive.google.com/drive/folders/`**`FOLDER_ID_HERE`**
6. Add to `.env`:
   ```
   GOOGLE_DOCS_FOLDER_ID=FOLDER_ID_HERE
   ```

If it still fails, the project may need **Billing** enabled (free tier is enough). Go to **Billing** in the left menu and link a payment method.
