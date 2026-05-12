# CLAUDE.md

Sandbox for learning agentic AI patterns. Python 3.12, uv for everything.

## Commands
- `uv add <pkg>` — add a dependency (edits pyproject.toml + uv.lock)
- `uv run python <script>.py` — run a script in the project venv
- `uv run pytest` — run tests (once we add them)
- `uv sync` — install deps from lockfile (after a fresh clone)

## Conventions
- Type hints on all functions
- Prefer stdlib + small deps; do NOT introduce LangChain/LlamaIndex unless I explicitly ask
- One concept per file while learning — don't merge experiments

## Secrets
- `.env` holds `ANTHROPIC_API_KEY`. It is gitignored. Never commit it.
- Load secrets via `python-dotenv`, not from os.environ directly in scripts.

## When unsure
- Anthropic SDK / API behaviour → fetch from https://docs.claude.com rather than guessing
- MCP details → https://modelcontextprotocol.io