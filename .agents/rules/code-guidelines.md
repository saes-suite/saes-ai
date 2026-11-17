---
trigger: always_on
---

# SAES-AI — Intelligence & Agent Layer for SAES

> This file is read automatically by Antigravity and Claude on every session.
> It defines the project identity, architecture, and conventions.

## Project Identity

**What**: SAES-AI — Conversational statistical assistant, Model Context Protocol (MCP) server, and agent toolkit for SAES.

**Goal**: Provide AI interpretation of benchmark results, multi-provider LLM integration (Ollama, OpenAI, Anthropic, Gemini, offline fallback), MCP tools for external agents, and Claude plugins.

---

## Tech Stack

| Layer | Technology | Notes |
|-------|------------|-------|
| Language | Python >= 3.11 | Modern Python standard |
| Core Engine | `saes` | Statistical computation dependency |
| Package Manager | `uv` | Dependency resolution & environment |
| LLM Providers | Ollama, Anthropic, OpenAI, Gemini | Multi-provider intelligence |
| Protocol | Model Context Protocol (MCP) | stdio & SSE for external tools |
| Web API | FastAPI, Uvicorn | Streaming and REST endpoints |
| Quality | Ruff, MyPy | Strict linting & typing |
| Testing | Pytest, Pytest-Cov | Unit & integration tests |

---

## Non-Negotiable Architecture Rules (API-First & Python Parity)

1. **Python Programmatic API First**: Every AI, chat, or agent capability exposed to `saes-ui` MUST be implemented as a clean programmatic Python API in `saes_ai` and exposed via FastAPI (`src/saes_ai/server/`) and/or MCP (`src/saes_ai/mcp/`).
2. **Backend-Driven Logic**: `saes-ui` must never construct domain prompts, statistical context, or analytical interpretations locally; it must consume `saes-ai` REST endpoints (`:8001`).
3. **Mandatory Quality Gate**: Before finishing any task or suggesting a commit, **you MUST run `make ci`** (`uv run ruff check .`, `uv run mypy src`, `uv run pytest`).
4. **Versioning & Changelog**: Update `CHANGELOG.md` under `## [Unreleased]`, never edit `version.txt` or `pyproject.toml` version manually (use `make release-stamp`), and follow Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`).
