# Deploy workflow

**Never deploy to prod directly.** Do not run `flyctl deploy`, `fly deploy`, or `make deploy` against the prod `blog-writer` app.

The flow is:
1. Make changes on a branch.
2. Open a PR — the `.github/workflows/pr-preview.yml` workflow will create `blog-writer-pr-{N}.fly.dev` automatically. Use that preview to verify the change end-to-end.
3. The user merges the PR via `gh pr merge` themselves. That's the only path to prod.

If a user asks you to "deploy" or "push to prod," interpret it as: push the branch, open a PR, comment the preview URL. Do not bypass.
