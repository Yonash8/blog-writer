.PHONY: deploy push fly dev dev-chat dev-all

# Run the app locally (web server, API, admin UI). Terminal 1.
dev:
	python3 -m uvicorn src.main:app --reload --host 127.0.0.1 --port 8080

# Chat with the agent via terminal (no WhatsApp). Terminal 2.
dev-chat:
	python3 scripts/dev_chat.py

# Run both in separate terminals: Terminal 1: make dev. Terminal 2: make dev-chat.
dev-all:
	@echo "Open TWO terminals. Run in parallel:"
	@echo "  Terminal 1: make dev        (web app + logs)"
	@echo "  Terminal 2: make dev-chat   (chat + agent logs)"

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
