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
