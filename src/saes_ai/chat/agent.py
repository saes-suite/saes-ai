"""Agentic statistical consultant for SAES-AI with native tool calling."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
from saesV2.app.pipeline import AnalysisReport
from saesV2.domain.models import Dataset

from saes_ai.chat.context import build_analysis_context
from saes_ai.chat.prompts import STATISTICAL_SYSTEM_PROMPT
from saes_ai.chat.tools import SAES_AGENT_TOOLS, ToolContext, execute_tool

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 8
MAX_TOOL_RESULT_CHARS = 12000
DEFAULT_OLLAMA_NUM_CTX = 16384


def _get_ollama_num_ctx() -> int:
    """Return configured Ollama context window size, defaulting to DEFAULT_OLLAMA_NUM_CTX."""
    raw = os.getenv("SAES_OLLAMA_NUM_CTX", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return DEFAULT_OLLAMA_NUM_CTX


def _sanitize_error_message(exc: Exception) -> str:
    """Redact API keys from exception strings before displaying to users."""
    return re.sub(r"([?&]key=)[^&\s'\"]+", r"\1***", str(exc))


def _extract_think_blocks(text: str) -> tuple[str | None, str]:
    """Extract <think>...</think> reasoning blocks from model text, returning (thought, clean_text)."""
    thoughts: list[str] = []

    def _collect(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        if inner:
            thoughts.append(inner)
        return ""

    clean = re.sub(r"<think>(.*?)</think>", _collect, text, flags=re.DOTALL)
    # Also extract an unclosed trailing <think> block (e.g. when truncated mid-thought).
    clean = re.sub(r"<think>(.*)$", _collect, clean, flags=re.DOTALL).strip()
    return ("\n\n".join(thoughts) if thoughts else None), clean


def _post_with_retry(
    client: httpx.Client,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json_payload: dict[str, Any],
    max_retries: int = 2,
) -> httpx.Response:
    """POST with brief retry on transient 429/503 provider overload errors."""
    is_test = bool(os.getenv("PYTEST_CURRENT_TEST"))
    last_res: httpx.Response | None = None
    for attempt in range(max_retries + 1):
        res = client.post(url, headers=headers, json=json_payload)
        last_res = res
        if res.status_code not in (429, 503) or attempt == max_retries:
            res.raise_for_status()
            return res
        if not is_test:
            time.sleep(0.8 * (attempt + 1))
    assert last_res is not None
    last_res.raise_for_status()
    return last_res


def _default_datasets_dir() -> Path:
    custom_dir = os.getenv("SAES_DATASETS_DIR")
    if custom_dir:
        p = Path(custom_dir).resolve()
    else:
        p = Path.cwd() / "datasets"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _truncate_tool_result(result: dict[str, Any]) -> str:
    raw = json.dumps(result, default=str)
    if len(raw) <= MAX_TOOL_RESULT_CHARS:
        return raw
    return raw[: MAX_TOOL_RESULT_CHARS - 32] + '... [truncated for brevity]"}'


def _summarize_tool_result(name: str, result: dict[str, Any]) -> str:
    if not result.get("ok", True):
        return f"Error: {result.get('error', 'unknown error')}"
    if name == "list_datasets":
        return f"Found {result.get('datasets_count', 0)} dataset(s)"
    if name == "inspect_dataset":
        status = "valid" if result.get("is_valid") else f"{result.get('error_count', 0)} error(s)"
        return f"{result.get('filename', 'dataset')}: {result.get('row_count', 0)} rows ({status})"
    if name == "edit_dataset":
        status = "valid" if result.get("is_valid") else f"{result.get('error_count', 0)} error(s)"
        return f"Edited {result.get('filename', 'dataset')} ({status})"
    if name == "create_dataset":
        return f"Created {result.get('filename', 'dataset')} ({result.get('row_count', 0)} rows)"
    if name == "get_dataset_kpis":
        return f"KPIs for {result.get('filename', 'dataset')} ({result.get('total_rows', 0)} rows)"
    if name == "explore_dataset":
        return f"Normality passed {result.get('normality_passed_count', 0)}/{result.get('normality_total_checks', 0)}"
    if name == "run_statistical_analysis":
        return f"{result.get('significant_cells_count', 0)}/{result.get('cells_count', 0)} significant cells"
    if name == "navigate_ui":
        act = result.get("action", {})
        tab = f"/{act['tab']}" if act.get("tab") else ""
        return f"Navigated UI to {act.get('view', 'visualize')}{tab}"
    return "Completed"


class StatisticalChatAgent:
    """Agentic conversational assistant for SAES dataset management, analysis, and UI control."""

    def __init__(
        self,
        report: AnalysisReport | None = None,
        dataset: Dataset | None = None,
        system_prompt: str = STATISTICAL_SYSTEM_PROMPT,
        provider: str = "local",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        tool_context: ToolContext | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> None:
        self.report = report
        self.dataset = dataset
        self.system_prompt = system_prompt
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.tool_context = tool_context or ToolContext(datasets_dir=_default_datasets_dir())
        self.history: list[dict[str, str]] = list(history) if history else []

    def set_context(self, report: AnalysisReport | None, dataset: Dataset | None = None) -> None:
        """Update current statistical context."""
        self.report = report
        if dataset is not None:
            self.dataset = dataset

    def _build_runtime_context(self) -> str:
        parts: list[str] = []
        stats_ctx = build_analysis_context(self.report, self.dataset)
        if stats_ctx and not stats_ctx.startswith("No statistical analysis"):
            parts.append(stats_ctx)

        ctx = self.tool_context
        state_lines = ["### Active Workspace State"]
        if ctx.active_dataset_name:
            state_lines.append(f"- **Active Dataset**: `{ctx.active_dataset_name}`")
        else:
            state_lines.append("- **Active Dataset**: None selected")

        try:
            ctx.datasets_dir.mkdir(parents=True, exist_ok=True)
            stored_files = sorted(p.name for p in ctx.datasets_dir.glob("*.csv") if p.is_file())
            if stored_files:
                state_lines.append(
                    f"- **Datasets Available in Storage ({len(stored_files)})**: "
                    + ", ".join(f"`{name}`" for name in stored_files[:20])
                )
            else:
                state_lines.append("- **Datasets Available in Storage**: None")
        except Exception:
            pass

        if ctx.staged_csv is not None:
            state_lines.append(
                f"- **Staged Chat Upload**: `{ctx.staged_filename or 'uploaded_dataset.csv'}` "
                f"({len(ctx.staged_csv)} chars) — call `inspect_dataset(use_staged=true)` to inspect it."
            )
        if ctx.current_view:
            view_str = ctx.current_view + (f" (tab: `{ctx.current_tab}`)" if ctx.current_tab else "")
            state_lines.append(f"- **Current UI View**: `{view_str}`")
        state_lines.append(f"- **Significance Level (alpha)**: {ctx.alpha}")
        parts.append("\n".join(state_lines))
        return "\n\n".join(parts)

    def _run_tool_and_emit(self, name: str, args: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Execute a tool and yield tool_call, ui_action, artifact, and tool_result events."""
        logger.debug("Tool call --> %s args=%s", name, args)
        t0 = time.perf_counter()
        yield {"type": "tool_call", "tool": name, "arguments": args}
        prev_ui_count = len(self.tool_context.ui_actions)
        prev_art_count = len(self.tool_context.artifacts)
        result = execute_tool(self.tool_context, name, args)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        summary = _summarize_tool_result(name, result)
        ok = bool(result.get("ok", True))
        new_ui = self.tool_context.ui_actions[prev_ui_count:]
        new_arts = self.tool_context.artifacts[prev_art_count:]
        logger.debug(
            "Tool result <-- %s ok=%s elapsed=%.1fms ui_actions=%d artifacts=%d summary=%s",
            name,
            ok,
            elapsed_ms,
            len(new_ui),
            len(new_arts),
            summary,
        )
        self.tool_context.tool_events.append(
            {
                "tool": name,
                "arguments": args,
                "ok": ok,
                "summary": summary,
            }
        )
        for action in new_ui:
            yield {"type": "ui_action", "action": action}
        for artifact in new_arts:
            yield {"type": "artifact", "artifact": artifact}
        yield {
            "type": "tool_result",
            "tool": name,
            "ok": ok,
            "summary": summary,
            "result": result,
        }

    def ask(self, query: str) -> str:
        """Process user query synchronously and return final assistant text."""
        chunks: list[str] = []
        for event in self.stream_events(query):
            if event.get("type") == "token":
                chunks.append(str(event.get("token", "")))
        return "".join(chunks)

    def stream_ask(self, query: str) -> Iterator[str]:
        """Process user query and yield response text tokens iteratively."""
        for event in self.stream_events(query):
            if event.get("type") == "token":
                token = str(event.get("token", ""))
                if token:
                    yield token

    def stream_events(self, query: str) -> Iterator[dict[str, Any]]:
        """Run the agentic loop and yield structured events (tool_call, ui_action, token)."""
        context_str = self._build_runtime_context()
        provider = (self.provider or "local").lower().strip()
        chunks: list[str] = []

        def _dispatch() -> Iterator[dict[str, Any]]:
            if provider == "local":
                yield from self._agent_local(query, context_str)
                return
            if provider == "openai":
                key = self.api_key or os.getenv("OPENAI_API_KEY")
                if key:
                    yield from self._agent_openai(query, context_str, api_key=key)
                else:
                    yield {
                        "type": "token",
                        "token": (
                            "⚠️ **OpenAI API key missing.** "
                            "Please configure your OpenAI API key in **AI Settings**."
                        ),
                    }
                return
            if provider == "anthropic":
                key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
                if key:
                    yield from self._agent_anthropic(query, context_str, api_key=key)
                else:
                    yield {
                        "type": "token",
                        "token": (
                            "⚠️ **Anthropic API key missing.** "
                            "Please configure your Anthropic API key in **AI Settings**."
                        ),
                    }
                return
            if provider in ("gemini", "google"):
                key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                if key:
                    yield from self._agent_gemini(query, context_str, api_key=key)
                else:
                    yield {
                        "type": "token",
                        "token": (
                            "⚠️ **Google Gemini API key missing.** "
                            "Please configure your Gemini API key in **AI Settings**."
                        ),
                    }
                return

            # Fallback auto-detection when an unrecognized provider is passed
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            openai_key = os.getenv("OPENAI_API_KEY")
            if anthropic_key:
                yield from self._agent_anthropic(query, context_str, api_key=anthropic_key)
            elif openai_key:
                yield from self._agent_openai(query, context_str, api_key=openai_key)
            else:
                yield {
                    "type": "token",
                    "token": (
                        f"⚠️ **Unsupported AI provider `{provider}`.** "
                        "Please select Local AI (Ollama), OpenAI, Anthropic, or Gemini in **AI Settings**."
                    ),
                }

        try:
            logger.debug(
                "Agent dispatch: provider=%s model=%s context_chars=%d",
                provider,
                self.model,
                len(context_str),
            )
            t0 = time.perf_counter()
            for ev in _dispatch():
                if ev.get("type") == "token":
                    chunks.append(str(ev.get("token", "")))
                yield ev
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            full_response = "".join(chunks)
            logger.debug(
                "Agent complete: provider=%s model=%s elapsed=%.1fms response_chars=%d tool_events=%d",
                provider,
                self.model,
                elapsed_ms,
                len(full_response),
                len(self.tool_context.tool_events),
            )
            self.history.append({"role": "user", "content": query})
            if full_response:
                self.history.append({"role": "assistant", "content": full_response})

    # ------------------------------------------------------------------
    # Text chunking helper
    # ------------------------------------------------------------------

    @staticmethod
    def _emit_text_tokens(text: str) -> Iterator[dict[str, Any]]:
        tokens = re.split(r"(\s+)", text)
        is_test = bool(os.getenv("PYTEST_CURRENT_TEST"))
        buf: list[str] = []
        buf_len = 0
        for tok in tokens:
            if not tok:
                continue
            buf.append(tok)
            buf_len += len(tok)
            if buf_len >= 18:
                yield {"type": "token", "token": "".join(buf)}
                buf.clear()
                buf_len = 0
                if not is_test:
                    time.sleep(0.005)
        if buf:
            yield {"type": "token", "token": "".join(buf)}

    def _build_chat_messages(self, query: str, context_str: str) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        for h in self.history[-10:]:
            role = h.get("role", "user")
            content = h.get("content", "")
            if role in ("user", "assistant") and content:
                msgs.append({"role": role, "content": content})
        msgs.append({"role": "user", "content": f"{context_str}\n\nUser Request: {query}"})
        return msgs

    # ------------------------------------------------------------------
    # 1. OpenAI & OpenAI-compatible Tool-Calling Agent Loop
    # ------------------------------------------------------------------

    def _agent_openai(self, query: str, context_str: str, api_key: str) -> Iterator[dict[str, Any]]:
        if not self.model or not self.model.strip():
            yield {
                "type": "token",
                "token": "⚠️ **No model specified for OpenAI.** Please select a model in **AI Settings**.",
            }
            return

        selected_model = self.model.strip()
        target_url = (self.base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        messages = self._build_chat_messages(query, context_str)
        tools_payload = [t.to_openai_tool() for t in SAES_AGENT_TOOLS]

        try:
            with httpx.Client(timeout=120.0) as client:
                for _ in range(MAX_TOOL_ITERATIONS):
                    payload: dict[str, Any] = {
                        "model": selected_model,
                        "messages": messages,
                        "tools": tools_payload,
                        "tool_choice": "auto",
                    }
                    res = _post_with_retry(client, target_url, headers=headers, json_payload=payload)
                    data = res.json()
                    msg = data["choices"][0]["message"]
                    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
                    if reasoning:
                        yield {"type": "thought", "thought": str(reasoning).strip()}
                    tool_calls = msg.get("tool_calls") or []

                    if not tool_calls:
                        raw_content = str(msg.get("content") or "")
                        think_text, clean_content = _extract_think_blocks(raw_content)
                        if think_text:
                            yield {"type": "thought", "thought": think_text}
                        yield from self._emit_text_tokens(clean_content)
                        return

                    if msg.get("content"):
                        think_text, pre_text = _extract_think_blocks(str(msg["content"]))
                        combined_thought = "\n\n".join(p for p in (think_text, pre_text) if p)
                        if combined_thought:
                            yield {"type": "thought", "thought": combined_thought}

                    messages.append(msg)
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        t_name = str(fn.get("name", ""))
                        try:
                            t_args = json.loads(fn.get("arguments") or "{}")
                        except Exception:
                            t_args = {}

                        tool_res_str = "{}"
                        for ev in self._run_tool_and_emit(t_name, t_args):
                            if ev["type"] == "tool_result":
                                tool_res_str = _truncate_tool_result(ev["result"])
                            yield ev

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.get("id", t_name),
                                "content": tool_res_str,
                            }
                        )

                yield from self._emit_text_tokens("Reached maximum tool execution steps. Please refine your request.")
        except Exception as exc:
            logger.exception("OpenAI API error: %s", _sanitize_error_message(exc))
            yield {
                "type": "token",
                "token": f"⚠️ **OpenAI API error:** {_sanitize_error_message(exc)}",
            }

    # ------------------------------------------------------------------
    # 2. Anthropic Claude Tool-Calling Agent Loop
    # ------------------------------------------------------------------

    def _agent_anthropic(self, query: str, context_str: str, api_key: str) -> Iterator[dict[str, Any]]:
        if not self.model or not self.model.strip():
            yield {
                "type": "token",
                "token": "⚠️ **No model specified for Anthropic.** Please select a model in **AI Settings**.",
            }
            return

        selected_model = self.model.strip()
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        messages: list[dict[str, Any]] = []
        for h in self.history[-10:]:
            if h.get("role") in ("user", "assistant") and h.get("content"):
                messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": f"{context_str}\n\nUser Request: {query}"})
        tools_payload = [t.to_anthropic_tool() for t in SAES_AGENT_TOOLS]

        try:
            with httpx.Client(timeout=120.0) as client:
                for _ in range(MAX_TOOL_ITERATIONS):
                    payload: dict[str, Any] = {
                        "model": selected_model,
                        "max_tokens": 4096,
                        "system": self.system_prompt,
                        "messages": messages,
                        "tools": tools_payload,
                    }
                    res = _post_with_retry(
                        client,
                        "https://api.anthropic.com/v1/messages",
                        headers=headers,
                        json_payload=payload,
                    )
                    data = res.json()
                    content_blocks = data.get("content", [])
                    for b in content_blocks:
                        if b.get("type") == "thinking" and b.get("thinking"):
                            yield {"type": "thought", "thought": str(b["thinking"]).strip()}
                    tool_uses = [b for b in content_blocks if b.get("type") == "tool_use"]

                    if not tool_uses:
                        text_parts = [b.get("text", "") for b in content_blocks if b.get("type") == "text"]
                        yield from self._emit_text_tokens("".join(text_parts))
                        return

                    pre_texts = [
                        str(b.get("text", "")).strip()
                        for b in content_blocks
                        if b.get("type") == "text" and b.get("text")
                    ]
                    if pre_texts:
                        yield {"type": "thought", "thought": "\n".join(pre_texts)}

                    messages.append({"role": "assistant", "content": content_blocks})
                    tool_results_blocks: list[dict[str, Any]] = []
                    for tu in tool_uses:
                        t_name = str(tu.get("name", ""))
                        t_args = tu.get("input") or {}
                        tool_res_str = "{}"
                        for ev in self._run_tool_and_emit(t_name, t_args):
                            if ev["type"] == "tool_result":
                                tool_res_str = _truncate_tool_result(ev["result"])
                            yield ev
                        tool_results_blocks.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tu.get("id", t_name),
                                "content": tool_res_str,
                            }
                        )
                    messages.append({"role": "user", "content": tool_results_blocks})

                yield from self._emit_text_tokens("Reached maximum tool execution steps. Please refine your request.")
        except Exception as exc:
            logger.exception("Claude API error: %s", _sanitize_error_message(exc))
            yield {
                "type": "token",
                "token": f"⚠️ **Claude API error:** {_sanitize_error_message(exc)}",
            }

    # ------------------------------------------------------------------
    # 3. Google Gemini Tool-Calling Agent Loop
    # ------------------------------------------------------------------

    def _agent_gemini(self, query: str, context_str: str, api_key: str) -> Iterator[dict[str, Any]]:
        if not self.model or not self.model.strip():
            yield {
                "type": "token",
                "token": "⚠️ **No model specified for Gemini.** Please select a model in **AI Settings**.",
            }
            return

        selected_model = self.model.strip().replace("models/", "")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{selected_model}:generateContent?key={api_key}"

        contents: list[dict[str, Any]] = []
        for h in self.history[-10:]:
            role = "model" if h.get("role") == "assistant" else "user"
            if h.get("content"):
                contents.append({"role": role, "parts": [{"text": h["content"]}]})
        contents.append({"role": "user", "parts": [{"text": f"{context_str}\n\nUser Request: {query}"}]})

        tools_payload = [{"functionDeclarations": [t.to_gemini_declaration() for t in SAES_AGENT_TOOLS]}]

        try:
            with httpx.Client(timeout=120.0) as client:
                for _ in range(MAX_TOOL_ITERATIONS):
                    payload: dict[str, Any] = {
                        "systemInstruction": {"parts": [{"text": self.system_prompt}]},
                        "contents": contents,
                        "tools": tools_payload,
                    }
                    res = _post_with_retry(client, url, json_payload=payload)
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if not candidates:
                        break
                    model_content = candidates[0].get("content", {})
                    parts = model_content.get("parts", [])
                    for p in parts:
                        if p.get("thought") and p.get("text"):
                            yield {"type": "thought", "thought": str(p["text"]).strip()}
                    fn_calls = [p["functionCall"] for p in parts if "functionCall" in p]

                    if not fn_calls:
                        text = "".join(p.get("text", "") for p in parts if "text" in p and not p.get("thought"))
                        yield from self._emit_text_tokens(text)
                        return

                    pre_texts = [
                        str(p.get("text", "")).strip()
                        for p in parts
                        if "text" in p and not p.get("thought") and p.get("text")
                    ]
                    if pre_texts:
                        yield {"type": "thought", "thought": "\n".join(pre_texts)}

                    contents.append(model_content)
                    response_parts: list[dict[str, Any]] = []
                    for fc in fn_calls:
                        t_name = str(fc.get("name", ""))
                        t_args = fc.get("args") or {}
                        tool_res: dict[str, Any] = {}
                        for ev in self._run_tool_and_emit(t_name, t_args):
                            if ev["type"] == "tool_result":
                                tool_res = ev["result"]
                            yield ev
                        response_parts.append(
                            {
                                "functionResponse": {
                                    "name": t_name,
                                    "response": json.loads(_truncate_tool_result(tool_res)),
                                }
                            }
                        )
                    contents.append({"role": "user", "parts": response_parts})

                yield from self._emit_text_tokens("Reached maximum tool execution steps. Please refine your request.")
        except Exception as exc:
            logger.exception("Gemini API error: %s", _sanitize_error_message(exc))
            yield {
                "type": "token",
                "token": f"⚠️ **Gemini API error:** {_sanitize_error_message(exc)}",
            }

    # ------------------------------------------------------------------
    # 4. Local Ollama / v1 Tool-Calling Agent Loop
    # ------------------------------------------------------------------

    def _agent_local(self, query: str, context_str: str) -> Iterator[dict[str, Any]]:
        if not self.model or not self.model.strip():
            yield {
                "type": "token",
                "token": (
                    "⚠️ **No model specified for Local AI.** "
                    "Please select or pull an Ollama model in **AI Settings**."
                ),
            }
            return

        selected_model = self.model.strip()
        clean_url = (self.base_url or "http://localhost:11434").rstrip("/")
        if clean_url.endswith("/v1"):
            # OpenAI-compatible local server
            old_base = self.base_url
            self.base_url = clean_url[:-3] if clean_url.endswith("/v1") else clean_url
            try:
                yield from self._agent_openai(query, context_str, api_key="local")
            finally:
                self.base_url = old_base
            return

        messages = self._build_chat_messages(query, context_str)
        tools_payload = [t.to_openai_tool() for t in SAES_AGENT_TOOLS]
        num_ctx = _get_ollama_num_ctx()
        timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)

        try:
            with httpx.Client(timeout=timeout) as client:
                use_tools = True
                nudged = False
                for step in range(MAX_TOOL_ITERATIONS):
                    payload: dict[str, Any] = {
                        "model": selected_model,
                        "stream": True,
                        "messages": messages,
                        "options": {"num_ctx": num_ctx},
                    }
                    if use_tools:
                        payload["tools"] = tools_payload

                    logger.debug(
                        "Ollama POST %s/api/chat step=%d model=%s num_ctx=%d use_tools=%s messages=%d",
                        clean_url,
                        step + 1,
                        selected_model,
                        num_ctx,
                        use_tools,
                        len(messages),
                    )
                    step_t0 = time.perf_counter()

                    thinking_parts: list[str] = []
                    content_parts: list[str] = []
                    tool_calls: list[dict[str, Any]] = []
                    done_reason: str | None = None
                    retry_without_tools = False

                    with client.stream("POST", f"{clean_url}/api/chat", json=payload) as res:
                        if res.status_code == 400 and use_tools:
                            # Model may not support Ollama native tool calling; retry without tools
                            logger.debug("Ollama returned 400 with tools enabled; retrying without tools")
                            use_tools = False
                            retry_without_tools = True
                        else:
                            res.raise_for_status()
                            for line in res.iter_lines():
                                if not line.strip():
                                    continue
                                chunk = json.loads(line)
                                chunk_msg = chunk.get("message") or {}
                                if chunk_msg.get("thinking"):
                                    thinking_parts.append(str(chunk_msg["thinking"]))
                                if chunk_msg.get("content"):
                                    content_parts.append(str(chunk_msg["content"]))
                                if chunk_msg.get("tool_calls"):
                                    tool_calls.extend(chunk_msg["tool_calls"])
                                if chunk.get("done_reason"):
                                    done_reason = str(chunk["done_reason"])

                    if retry_without_tools:
                        continue

                    thinking_text = "".join(thinking_parts).strip()
                    raw_content = "".join(content_parts)
                    logger.debug(
                        "Ollama response step=%d elapsed=%.1fms done_reason=%s "
                        "thinking_chars=%d content_chars=%d tool_calls=%d",
                        step + 1,
                        (time.perf_counter() - step_t0) * 1000.0,
                        done_reason,
                        len(thinking_text),
                        len(raw_content),
                        len(tool_calls),
                    )
                    if thinking_text:
                        yield {"type": "thought", "thought": thinking_text}

                    if not tool_calls:
                        think_text, clean_content = _extract_think_blocks(raw_content)
                        if think_text:
                            yield {"type": "thought", "thought": think_text}
                        if clean_content:
                            yield from self._emit_text_tokens(clean_content)
                            return

                        if done_reason == "length":
                            logger.warning(
                                "Ollama model=%s stopped mid-response (done_reason=length, num_ctx=%d)",
                                selected_model,
                                num_ctx,
                            )
                            yield from self._emit_text_tokens(
                                f"⚠️ **Local model `{selected_model}` stopped mid-response "
                                "(context/token limit reached while thinking).** "
                                "Try starting a new chat, shortening the prompt, or selecting a "
                                "larger-context model in **AI Settings**."
                            )
                            return

                        if not nudged:
                            nudged = True
                            logger.debug(
                                "Ollama model=%s returned empty content and no tool_calls; nudging",
                                selected_model,
                            )
                            if raw_content:
                                messages.append({"role": "assistant", "content": raw_content})
                            messages.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "Please continue and provide your final response or call "
                                        "the appropriate tool."
                                    ),
                                }
                            )
                            continue

                        logger.warning(
                            "Ollama model=%s finished without response or tool_calls (done_reason=%s)",
                            selected_model,
                            done_reason,
                        )
                        yield from self._emit_text_tokens(
                            f"⚠️ **Local model `{selected_model}` finished thinking without "
                            "returning a response or tool call.** Please try again or switch "
                            "models in **AI Settings**."
                        )
                        return

                    if raw_content:
                        think_text, pre_text = _extract_think_blocks(raw_content)
                        combined_thought = "\n\n".join(p for p in (think_text, pre_text) if p)
                        if combined_thought:
                            yield {"type": "thought", "thought": combined_thought}

                    messages.append({"role": "assistant", "content": raw_content, "tool_calls": tool_calls})
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        t_name = str(fn.get("name", ""))
                        raw_args = fn.get("arguments", {})
                        t_args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                        tool_res_str = "{}"
                        for ev in self._run_tool_and_emit(t_name, t_args):
                            if ev["type"] == "tool_result":
                                tool_res_str = _truncate_tool_result(ev["result"])
                            yield ev
                        messages.append({"role": "tool", "content": tool_res_str})

                yield from self._emit_text_tokens("Reached maximum tool execution steps. Please refine your request.")
        except Exception as exc:
            logger.exception("Local AI error on %s: %s", clean_url, exc)
            yield {
                "type": "token",
                "token": (
                    f"⚠️ **Local AI error on `{clean_url}`:** {exc!s}. "
                    "Please ensure Ollama is running or select another provider in **AI Settings**."
                ),
            }

    # ------------------------------------------------------------------
    # Provider Discovery & Ollama Management (unchanged public API)
    # ------------------------------------------------------------------

    @staticmethod
    def fetch_local_models(base_url: str = "http://localhost:11434") -> list[str]:
        """Fetch available models from a local Ollama server."""
        clean_url = (base_url or "http://localhost:11434").rstrip("/")
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{clean_url}/api/tags")
                if res.status_code == 200:
                    data = res.json()
                    return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        except Exception:
            pass
        return []

    @staticmethod
    def start_ollama_service() -> bool:
        """Attempt to start the local Ollama background daemon."""
        binary_path = shutil.which("ollama")
        if not binary_path:
            for p in ["/usr/local/bin/ollama", "/opt/homebrew/bin/ollama", "/usr/bin/ollama"]:
                if os.path.exists(p):
                    binary_path = p
                    break
        if binary_path:
            try:
                subprocess.Popen(
                    [binary_path, "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                time.sleep(1.5)
                return True
            except Exception:
                return False
        return False

    @staticmethod
    def probe_provider(
        provider: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        """Probe AI provider connectivity and retrieve available models."""
        prov = (provider or "local").lower().strip()

        if prov == "local":
            clean_url = (base_url or "http://localhost:11434").rstrip("/")
            models = StatisticalChatAgent.fetch_local_models(clean_url)
            if models:
                return {
                    "status": "ok",
                    "models": models,
                    "message": f"Connected to Ollama ({len(models)} model(s) available).",
                    "can_start_ollama": False,
                }
            try:
                with httpx.Client(timeout=2.0) as client:
                    resp = client.get(f"{clean_url}/api/tags")
                    if resp.status_code == 200:
                        return {
                            "status": "ok",
                            "models": [],
                            "message": "Connected to Ollama, but no models installed. Run: ollama run llama3.2",
                            "can_start_ollama": False,
                        }
            except Exception:
                pass

            binary = shutil.which("ollama")
            if not binary:
                for p in ["/usr/local/bin/ollama", "/opt/homebrew/bin/ollama", "/usr/bin/ollama"]:
                    if os.path.exists(p):
                        binary = p
                        break
            if binary:
                return {
                    "status": "binary_found_not_running",
                    "models": [],
                    "message": f"Ollama is installed at {binary}, but the service is currently stopped.",
                    "can_start_ollama": True,
                    "binary_path": binary,
                }
            return {
                "status": "not_installed",
                "models": [],
                "message": "Ollama is not installed or not in PATH.",
                "can_start_ollama": False,
            }

        if prov in ("gemini", "google"):
            key = (api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
            if not key:
                return {
                    "status": "error",
                    "models": [],
                    "message": "Google Gemini API key is missing. Please enter your API key.",
                }
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
                with httpx.Client(timeout=10.0) as client:
                    res = client.get(url)
                    if res.status_code in (400, 403):
                        return {
                            "status": "error",
                            "models": [],
                            "message": "Invalid Google Gemini API key or unauthorized project.",
                        }
                    res.raise_for_status()
                    data = res.json()
                    models = [
                        m["name"].replace("models/", "")
                        for m in data.get("models", [])
                        if "generateContent" in m.get("supportedGenerationMethods", [])
                    ]
                    preferred = [
                        "gemini-3.5-flash",
                        "gemini-3.5-flash-lite",
                        "gemini-3.8-flash",
                        "gemini-flash-latest",
                        "gemini-2.5-pro",
                    ]
                    sorted_models = [m for m in preferred if m in models] + [m for m in models if m not in preferred]
                    return {
                        "status": "ok",
                        "models": sorted_models,
                        "message": f"Successfully connected to Google Gemini ({len(models)} models available).",
                    }
            except Exception as e:
                return {
                    "status": "error",
                    "models": [],
                    "message": f"Failed to connect to Google Gemini: {e!s}",
                }

        if prov == "openai":
            key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
            if not key:
                return {"status": "error", "models": [], "message": "OpenAI API key is missing."}
            try:
                clean_url = (base_url or "https://api.openai.com/v1").rstrip("/")
                with httpx.Client(timeout=10.0) as client:
                    res = client.get(f"{clean_url}/models", headers={"Authorization": f"Bearer {key}"})
                    if res.status_code == 401:
                        return {"status": "error", "models": [], "message": "Invalid OpenAI API key."}
                    res.raise_for_status()
                    data = res.json()
                    chat_models = [
                        m["id"]
                        for m in data.get("data", [])
                        if any(k in m["id"] for k in ["gpt-4", "gpt-3.5", "o1", "o3", "chat"])
                    ]
                    preferred = ["gpt-4o-mini", "gpt-4o", "o3-mini", "gpt-4.5-preview"]
                    sorted_models = [m for m in preferred if m in chat_models] + [
                        m for m in sorted(chat_models) if m not in preferred
                    ]
                    return {
                        "status": "ok",
                        "models": sorted_models,
                        "message": f"Successfully connected to OpenAI ({len(sorted_models)} models available).",
                    }
            except Exception as e:
                return {"status": "error", "models": [], "message": f"Failed to connect to OpenAI: {e!s}"}

        if prov == "anthropic":
            key = (api_key or os.getenv("ANTHROPIC_API_KEY") or "").strip()
            if not key:
                return {"status": "error", "models": [], "message": "Anthropic API key is missing."}
            try:
                headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
                with httpx.Client(timeout=10.0) as client:
                    res = client.get("https://api.anthropic.com/v1/models", headers=headers)
                    if res.status_code == 401:
                        return {"status": "error", "models": [], "message": "Invalid Anthropic API key."}
                    if res.status_code == 200:
                        data = res.json()
                        models = [m["id"] for m in data.get("data", [])]
                        return {
                            "status": "ok",
                            "models": models,
                            "message": f"Successfully connected to Anthropic Claude ({len(models)} models available).",
                        }
                    return {
                        "status": "error",
                        "models": [],
                        "message": f"Failed to fetch models from Anthropic (HTTP {res.status_code}).",
                    }
            except Exception as e:
                return {"status": "error", "models": [], "message": f"Failed to connect to Anthropic: {e!s}"}

        return {"status": "error", "models": [], "message": f"Unknown provider: {provider}"}

    @staticmethod
    def pull_ollama_model(name: str, base_url: str = "http://localhost:11434") -> dict[str, str]:
        """Request the local Ollama daemon to download/pull a model."""
        clean_url = (base_url or "http://localhost:11434").rstrip("/")
        clean_name = name.strip()
        if not clean_name:
            return {"status": "error", "message": "Model name cannot be empty."}
        try:
            with httpx.Client(timeout=600.0) as client:
                res = client.post(f"{clean_url}/api/pull", json={"name": clean_name, "stream": False})
                if res.status_code == 200:
                    return {
                        "status": "ok",
                        "message": f"Model '{clean_name}' downloaded successfully into Ollama.",
                    }
                return {
                    "status": "error",
                    "message": f"Ollama pull returned HTTP {res.status_code}: {res.text}",
                }
        except Exception as e:
            return {"status": "error", "message": f"Failed to pull model '{clean_name}': {e!s}"}

    @staticmethod
    def stream_pull_ollama_model(
        name: str,
        base_url: str = "http://localhost:11434",
    ) -> Iterator[str]:
        """Stream progress chunks of Ollama model pull."""
        clean_url = (base_url or "http://localhost:11434").rstrip("/")
        clean_name = name.strip()
        if not clean_name:
            yield json.dumps({"status": "error", "error": "Model name cannot be empty."}) + "\n"
            return
        try:
            with httpx.Client(timeout=1800.0) as client:
                with client.stream(
                    "POST", f"{clean_url}/api/pull", json={"name": clean_name, "stream": True}
                ) as response:
                    if response.status_code != 200:
                        yield (
                            json.dumps({"status": "error", "error": f"Ollama returned HTTP {response.status_code}"})
                            + "\n"
                        )
                        return
                    for line in response.iter_lines():
                        if line:
                            yield f"{line}\n"
        except Exception as e:
            yield json.dumps({"status": "error", "error": str(e)}) + "\n"
