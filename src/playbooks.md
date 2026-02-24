## Workflows

These describe what each workflow does. You decide the exact steps based on context.

### Write Article
User gives a topic. Call write_article → get article_id + google_doc_url. Share the doc link.
After the article is ready, you can offer to generate a hero image and/or infographic.

If write_article fails with a token/context limit error, the tool returns `error`, `plan`, and `retry_hint`. Tell the user what went wrong and explain the plan: set PROMPTLAYER_RESEARCH_MAX_TOKENS in .env to chunk down the deep research, then try again.

### Improve / Edit Article
User gives feedback on an existing article. Resolve article_id from context, call improve_article. For full rewrites, use use_promptlayer=True. Article auto-syncs to Google Doc.

### Add Citations / Links
Use web_search to find relevant URLs, then improve_article with the links parameter.

### Import Article from Google Doc
User shares a Google Docs URL. Use google_docs(action="fetch") to read it, create a topic and article in DB.

### Add Topic
Check for duplicates via db sql query, then insert into topics table.

### Generate Images (hero + infographic)
When the user asks for images/photos/visuals, this means a **hero image** and an **infographic**.

Generate → send_image the preview → user responds → act:
- **Approve** → call approve tool (embeds in article + syncs to Google Doc)
- **Refine** ("make it darker", "add more data") → call generate again with `feedback`
- **New idea** ("try a timeline instead", "do a flowchart") → call generate again with the new `infographic_type` or `description` (no `feedback` — this is a fresh generation, not a refinement)

The generate tools handle references automatically: refinements use the previous image as reference; fresh generations use random style references.

### Generate Images (generic)
Only when user explicitly asks for "generic illustrations" or "placeholder images in the body". Call generate_images.

### Data Questions
Use db(action="sql") for counts, aggregations, statistics. Scope to current user.

### Google Doc Sync
Resolve article_id, check if google_doc_url exists. Create or update accordingly.

### List Articles / Browse Topics
Query the db and format results for the user.

### Push to Ghost
When user says "push to ghost", "send to ghost", "ghost draft", or "publish":
Call push_to_ghost(article_id=...) immediately — do NOT narrate, just call the tool.
Returns a Ghost editor URL. Share it with the user.

### SEO Metadata
After article is approved, call generate_seo_metadata(article_id=...) to generate and save SEO fields.
This is called automatically by push_to_ghost if not already done.
