.PHONY: deploy push fly

# Commit, push to GitHub, and deploy to Fly.io in one command
# Usage: make deploy m="your commit message"
deploy:
	@if [ -z "$(m)" ]; then echo "Usage: make deploy m=\"your commit message\""; exit 1; fi
	git add -A
	git commit -m "$(m)"
	git push
	flyctl deploy

# Push to GitHub only (no Fly deploy)
push:
	git add -A
	git commit -m "$(m)"
	git push

# Deploy to Fly only (no git commit)
fly:
	flyctl deploy
