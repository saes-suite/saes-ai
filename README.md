<p align="center">
  <img src="https://raw.githubusercontent.com/saes-suite/saes-ai/main/assets/saes-logo.svg" alt="SAES Logo" width="128" />
</p>

# SAES-AI

> AI Agent, Chat Assistant, and Model Context Protocol (MCP) server for the Stochastic Algorithm Evaluation Suite (SAES).

## Overview

`saes-ai` extends `saesV2` with intelligent conversational capabilities:
- **Statistical Chat Assistant**: Ask natural-language questions about ANOVA results, normality violations, Welch tests, and effect sizes.
- **Multi-Provider Support**: Integrates with local Ollama, Anthropic Claude, OpenAI GPT, Google Gemini, plus deterministic offline fallbacks.
- **Model Context Protocol (MCP)**: Run `saes-mcp` to expose SAES statistical routines directly to Claude Desktop, Cursor, and compliant agents.
- **Modular FastAPI Router**: Seamlessly mounted into the unified SAES server or run standalone on `:8070`.

## Quickstart

```bash
# Install with uv
uv add saes-ai

# Or with dev tools
make install

# Run standalone REST API server (:8070)
make start-api

# Run MCP server
saes-mcp
```
