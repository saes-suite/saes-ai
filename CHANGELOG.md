# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [Released]

## [0.3.0] — 2026-11-19 (@rorro6787)

### Changed
- Made `StatisticalChatAgent` (`prompts.py`, `agent.py`, `tools.py`) a natural, general-purpose conversational assistant that answers questions directly without unsolicited tool calls, dataset mutations, or UI navigation, and removed legacy `active_dataset` / `select_dataset` state assumptions.

### Fixed
- Resolved `__version__` dynamically from package metadata (`importlib.metadata`) with `version.txt` fallback across `saes_ai`, FastAPI `/health`, and `SAESMCPServer`.
- Updated `scripts/release_stamp.js` and `.github/workflows/release.yml` to automatically bump `src/saes_ai/plugins/claude/plugin.json` during releases.
- Added SAES logo and updated `README.md` and `Makefile` `.PHONY` targets.

## [0.2.0] — 2026-11-18 (@rorro6787)

### Fixed
- Local Ollama agent (`StatisticalChatAgent._agent_local`) no longer stops silently mid-thought: increased default `num_ctx` to `16384` (configurable via `SAES_OLLAMA_NUM_CTX`), switched `/api/chat` to streaming NDJSON accumulation to prevent total-request timeouts, handled unclosed `<think>` blocks, and added explicit recovery/warning messages when `done_reason == "length"` or when thinking ends without content or tool calls.

### Added
- Structured `DEBUG` console logging (`SAES_AI_LOG_LEVEL=DEBUG`, `make start-api-debug` / `make start-ai-debug`) across FastAPI request middleware, chat routes, agent dispatch, Ollama streaming steps, and tool calls/results.
- `generate_plot` agent tool in `saes_ai.chat.tools` generating interactive Plotly specifications and static PNG previews for all `saesV2` plot/diagram types (`boxplot`, `violin`, `critical-distance`, `bayesian-rank`, `barplot`, `lineplot`, `ecdf`, `scatter`).
- Inline chat artifact streaming (`artifact` NDJSON event in `/api/v1/chat/stream` and `artifacts` field in `ChatResponse`) emitted by `generate_plot` and `generate_statistical_table` so diagrams, images, and tables render directly in chat with download metadata.
- Initial standalone release of `saes-ai` extracted from SAES core.
- Conversational statistical assistant (`StatisticalChatAgent`) with multi-provider LLM support (Ollama, OpenAI, Anthropic, Google Gemini, and offline rule engine).
- Model Context Protocol (MCP) server (`saes-mcp`) exposing SAES statistical routines to external agents.
- Claude Desktop plugin definitions and manifests.
- FastAPI REST & SSE streaming endpoints for AI chat, Ollama daemon management, and model probing.
- `run_ranking_analysis` MCP tool and ranking plot support (`critical-distance`, `bayesian-rank`) in `generate_distribution_plot`.
- `generate_benchmark_table` and `explore_dataset_distributions` MCP tools for ANOVA/Friedman/Wilcoxon table generation and normality/test recommendation profiling.
- Global Friedman + Nemenyi CD and Bayesian posterior ranking summaries injected into `build_analysis_context`.

