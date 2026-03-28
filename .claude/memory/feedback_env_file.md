---
name: API keys live in .env — always check there first
description: Project uses .env file loaded by bootstrap.py via python-dotenv; never test API key availability outside the app bootstrap path
type: feedback
---

API keys (OPENAI_API_KEY, etc.) are stored in `.env` at the project root. The app loads them via `load_dotenv()` in `app/bootstrap.py` at startup. All CLI commands (`pixi run ingest`, `pixi run build-graph`, etc.) call `bootstrap_app()` which loads `.env` automatically.

**Why:** I caused a false alarm by testing `os.environ.get('OPENAI_API_KEY')` in a bare Python process outside the bootstrap path, which doesn't load `.env`. This made the user think their money would be wasted again.

**How to apply:** When checking if an API key or env var is available, always test through the app's bootstrap path (e.g. `pixi run python -c "from dotenv import load_dotenv; ..."`) or just trust that `.env` exists and bootstrap loads it. Never run a bare `python -c` to test env vars for this project.
