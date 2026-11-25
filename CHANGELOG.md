# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [Released]

## [0.4.1] — 2026-11-24 (@rorro6787)

### Changed
- Updated `.github/workflows/release.yml` to authenticate release stamp pushes and GitHub Releases via the `saes-release-bot` GitHub App (`actions/create-github-app-token@v1`) so protected `main` rulesets remain enforced.

### Fixed
- Automatically route default `localhost:11434` Ollama requests to `OLLAMA_HOST` / `http://host.docker.internal:11434` when running inside Docker (`SAES_IN_DOCKER=1` or `/.dockerenv`), and return an accurate host-connectivity status instead of `"Ollama is not installed"` when unreachable from a container.

## [0.4.0] — 2026-11-24 (@rorro6787)

### Changed
- Migrated core statistical dependency from `saesV2` to canonical **`saes[viz]>=2.0.0`** (`import saes`), updating `pyproject.toml`, `Dockerfile`, release workflows, and all agent/server/test modules.

### Added
- Exposed full `saes` 2.0 statistical and visualization parity in `saes_ai.chat.tools`, `prompts.py`, and `mcp/tools.py`: new agent tools `run_posthoc_analysis` (Tukey HSD, Games–Howell, Dunnett, Dunn, Conover–Iman), `run_tost_equivalence` (Schuirmann TOST), `run_bayesian_correlated_ttest` (Nadeau–Bengio CV correlation), and `compute_effect_size_forest` (95% CIs); added `plackett-luce` and `effect-size-forest` to `generate_plot`; added Friedman Aligned-Ranks, Quade, RM-ANOVA (GG $\varepsilon$), and Plackett–Luce outputs to `run_ranking_analysis`; added explicit `paired=False` support, all 13 multiplicity adjustments, MDES sensitivity summaries, and Anderson–Darling / design-capability fields.

## [0.3.2] — 2026-11-20 (@rorro6787)

### Added
- Production `Dockerfile` (`python:3.12-slim` + `uv`) and `.dockerignore` exposing port `8070`, automatically upgrading `saesV2` from PyPI on build.
- Automatic git tag (`vX.Y.Z`) creation and GitHub Release publishing in `.github/workflows/release.yml`, plus `--upgrade-package saesV2` lockfile sync on release.

## [0.3.1] — 2026-11-20 (@rorro6787)

### Changed

- Removed redundant `ImportError` fallback for `saesV2.server.storage.get_datasets_dir` in `saes_ai/server/routes.py`.

### Fixed

- Use absolute `raw.githubusercontent.com` URL for the SAES logo in `README.md` so it renders properly on PyPI.

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
