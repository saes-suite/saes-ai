# SAES-AI

> AI Agent, Chat Assistant, and Model Context Protocol (MCP) server for the Stochastic Algorithm Evaluation Suite (SAES).

## Overview

`saes-ai` extends `saes` with intelligent conversational capabilities:
- **Statistical Chat Assistant**: Ask natural-language questions about ANOVA results, normality violations, Welch tests, and effect sizes.
- **Multi-Provider Support**: Integrates with local Ollama, Anthropic Claude, OpenAI GPT, Google Gemini, plus deterministic offline fallbacks.
- **Model Context Protocol (MCP)**: Run `saes-mcp` to expose SAES statistical routines directly to Claude Desktop, Cursor, and compliant agents.
- **Modular FastAPI Router**: Seamlessly mounted into the unified SAES server or run standalone.

## Quickstart

```bash
# Install with uv
uv add saes-ai

# Or with dev tools
make install

# Run MCP server
saes-mcp
```
