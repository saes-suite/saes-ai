# CLAUDE.md — SAES-AI (Intelligence & Agent Suite for SAES)

> This file defines the repository conventions, architecture, and quality gates for SAES-AI.

## Project Identity

**What**: SAES-AI — Conversational statistical assistant, MCP server, and LLM integrations for SAES.

## Repository Layout

```
saes-ai/
├── .agents/
│   └── rules/
│       └── code-guidelines.md
├── scripts/
│   └── release_stamp.js
├── src/
│   └── saes_ai/
│       ├── chat/          ← Statistical assistant, prompts & context builders
│       ├── mcp/           ← Model Context Protocol server & tool specifications
│       ├── plugins/       ← Claude & IDE plugin manifests
│       └── server/        ← FastAPI routes & schemas for chat & streaming
├── tests/                 ← Unit & integration tests
├── CHANGELOG.md
├── Makefile
├── pyproject.toml
├── README.md
└── version.txt
```

## Non-Negotiable Architecture Rules

1. **Python API & REST Parity**: Every AI feature exposed in `saes-ui` must exist as a programmatic Python API in `saes_ai` and be served via `saes_ai.server` (`:8001`). The UI does not perform analytical or prompt-building calculations locally.
2. **Changelog & Versioning**: Update `CHANGELOG.md` under `## [Unreleased]`; never edit `version.txt` or `pyproject.toml` version manually (use `make release-stamp`).

## Mandatory Quality Gate

Before completing any task, run:
```bash
make ci
```
This runs:
- `uv run ruff check .`
- `uv run mypy src`
- `uv run pytest`
