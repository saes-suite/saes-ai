# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial standalone release of `saes-ai` extracted from SAES core.
- Conversational statistical assistant (`StatisticalChatAgent`) with multi-provider LLM support (Ollama, OpenAI, Anthropic, Google Gemini, and offline rule engine).
- Model Context Protocol (MCP) server (`saes-mcp`) exposing SAES statistical routines to external agents.
- Claude Desktop plugin definitions and manifests.
- FastAPI REST & SSE streaming endpoints for AI chat, Ollama daemon management, and model probing.
- `run_ranking_analysis` MCP tool and ranking plot support (`critical-distance`, `bayesian-rank`) in `generate_distribution_plot`.
- Global Friedman + Nemenyi CD and Bayesian posterior ranking summaries injected into `build_analysis_context`.

## [Released]
