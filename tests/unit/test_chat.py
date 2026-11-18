"""Unit tests for SAES-AI statistical chat assistant."""

import io
import json
from typing import Any

import httpx
import saesV2

from saes_ai.chat.agent import DEFAULT_OLLAMA_NUM_CTX, StatisticalChatAgent, _extract_think_blocks
from saes_ai.chat.context import build_analysis_context

SAMPLE_CSV = """algorithm,instance,metric,run_id,value,maximize
NSGAII,ZDT1,HV,0,0.662145,True
NSGAII,ZDT1,HV,1,0.661890,True
NSGAII,ZDT1,HV,2,0.662012,True
MOEAD,ZDT1,HV,0,0.665421,True
MOEAD,ZDT1,HV,1,0.665100,True
MOEAD,ZDT1,HV,2,0.665900,True
"""


def test_build_analysis_context() -> None:
    dataset = saesV2.load(io.StringIO(SAMPLE_CSV))
    report = saesV2.analyze(dataset, generate_plots=False)

    ctx = build_analysis_context(report, dataset)
    assert "ZDT1" in ctx
    assert "HV" in ctx
    assert "NSGAII" in ctx
    assert "MOEAD" in ctx


def test_statistical_chat_agent_provider_errors() -> None:
    dataset = saesV2.load(io.StringIO(SAMPLE_CSV))
    report = saesV2.analyze(dataset, generate_plots=False)

    # Local AI provider (no model specified)
    local_no_model = StatisticalChatAgent(
        report=report,
        dataset=dataset,
        provider="local",
        base_url="http://127.0.0.1:99999",
    )
    resp_no_model = local_no_model.ask("Tell me about the test results")
    assert "No model specified for Local AI" in resp_no_model

    # OpenAI provider (missing key)
    openai_agent = StatisticalChatAgent(
        report=report,
        dataset=dataset,
        provider="openai",
        api_key=None,
    )
    resp_openai = openai_agent.ask("Tell me about the results")
    assert "OpenAI API key missing" in resp_openai

    # Anthropic provider (missing key)
    anthropic_agent = StatisticalChatAgent(
        report=report,
        dataset=dataset,
        provider="anthropic",
        api_key=None,
    )
    resp_anthropic = anthropic_agent.ask("Tell me about the results")
    assert "Anthropic API key missing" in resp_anthropic

    # Gemini provider (missing key)
    gemini_agent = StatisticalChatAgent(
        report=report,
        dataset=dataset,
        provider="gemini",
        api_key=None,
    )
    resp_gemini = gemini_agent.ask("Tell me about the results")
    assert "Google Gemini API key missing" in resp_gemini


def test_probe_provider() -> None:
    # 1. Local probe (unreachable port)
    res_local = StatisticalChatAgent.probe_provider("local", base_url="http://127.0.0.1:99999")
    assert res_local["status"] in ("offline", "error", "ok", "binary_found_not_running", "not_installed")

    # 2. OpenAI probe (invalid key)
    res_openai = StatisticalChatAgent.probe_provider("openai", api_key="sk-invalid")
    assert res_openai["status"] == "error"

    # 3. Anthropic probe (invalid key)
    res_anthropic = StatisticalChatAgent.probe_provider("anthropic", api_key="sk-ant-invalid")
    assert res_anthropic["status"] == "error"

    # 4. Gemini probe (invalid key)
    res_gemini = StatisticalChatAgent.probe_provider("gemini", api_key="invalid-key")
    assert res_gemini["status"] == "error"


def test_fetch_local_models_graceful_on_unreachable() -> None:
    models = StatisticalChatAgent.fetch_local_models(base_url="http://127.0.0.1:99999")
    assert isinstance(models, list)
    assert len(models) == 0


def test_pull_ollama_model_empty_and_unreachable() -> None:
    # Empty model name
    res_empty = StatisticalChatAgent.pull_ollama_model("", base_url="http://127.0.0.1:99999")
    assert res_empty["status"] == "error"

    # Unreachable daemon
    res_unreach = StatisticalChatAgent.pull_ollama_model("llama3.2", base_url="http://127.0.0.1:99999")
    assert res_unreach["status"] == "error"


def test_stream_pull_ollama_model_empty_and_unreachable() -> None:
    # Empty name
    gen_empty = list(StatisticalChatAgent.stream_pull_ollama_model("", base_url="http://127.0.0.1:99999"))
    assert len(gen_empty) == 1
    assert "error" in gen_empty[0]

    # Unreachable
    gen_unreach = list(StatisticalChatAgent.stream_pull_ollama_model("llama3.2", base_url="http://127.0.0.1:99999"))
    assert len(gen_unreach) == 1
    assert "error" in gen_unreach[0]


def test_statistical_chat_agent_stream_ask() -> None:
    dataset = saesV2.load(io.StringIO(SAMPLE_CSV))
    report = saesV2.analyze(dataset, generate_plots=False)

    agent = StatisticalChatAgent(report=report, dataset=dataset, provider="openai", api_key=None)
    tokens = list(agent.stream_ask("Why was this statistical test chosen?"))
    assert len(tokens) > 0
    full_text = "".join(tokens)
    assert "OpenAI API key missing" in full_text


def test_extract_think_blocks_closed_and_unclosed() -> None:
    thought, clean = _extract_think_blocks("<think>step 1</think>\nFinal answer")
    assert thought == "step 1"
    assert clean == "Final answer"

    trunc_thought, trunc_clean = _extract_think_blocks("<think>truncated mid-thought")
    assert trunc_thought == "truncated mid-thought"
    assert trunc_clean == ""


def test_local_ollama_truncation_and_empty_response_handling(monkeypatch: Any) -> None:
    captured_payloads: list[dict[str, Any]] = []

    class _FakeStreamResponse:
        def __init__(self, lines: list[str], status_code: int = 200) -> None:
            self._lines = lines
            self.status_code = status_code

        def __enter__(self) -> "_FakeStreamResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise httpx.HTTPStatusError(
                    "err",
                    request=httpx.Request("POST", "http://x"),
                    response=httpx.Response(self.status_code),
                )

        def iter_lines(self) -> list[str]:
            return self._lines

    def _make_client_cls(responses: list[list[str]]) -> type[httpx.Client]:
        idx = {"i": 0}

        class _FakeClient:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def __enter__(self) -> "_FakeClient":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def stream(self, method: str, url: str, json: dict[str, Any]) -> _FakeStreamResponse:
                captured_payloads.append(json)
                lines = responses[idx["i"]]
                idx["i"] += 1
                return _FakeStreamResponse(lines)

        return _FakeClient  # type: ignore[return-value]

    # Case 1: done_reason == "length" mid-thought -> emits thought + visible warning (never silent)
    length_chunks = [
        json.dumps({"message": {"role": "assistant", "thinking": "Thinking part 1. "}}),
        json.dumps({"message": {"role": "assistant", "thinking": "So I"}, "done": True, "done_reason": "length"}),
    ]
    monkeypatch.setattr("saes_ai.chat.agent.httpx.Client", _make_client_cls([length_chunks]))

    agent = StatisticalChatAgent(provider="local", model="qwen3:8b", base_url="http://localhost:11434")
    events = list(agent.stream_events("Generate a critical-distance diagram"))

    assert captured_payloads[0]["stream"] is True
    assert captured_payloads[0]["options"]["num_ctx"] == DEFAULT_OLLAMA_NUM_CTX
    thoughts = [e["thought"] for e in events if e["type"] == "thought"]
    tokens = "".join(e["token"] for e in events if e["type"] == "token")
    assert thoughts == ["Thinking part 1. So I"]
    assert "context/token limit reached while thinking" in tokens

    # Case 2: done_reason == "stop" with thinking only -> nudges once and returns answer
    stop_empty = [
        json.dumps(
            {"message": {"role": "assistant", "thinking": "Let me check."}, "done": True, "done_reason": "stop"}
        ),
    ]
    followup = [
        json.dumps(
            {"message": {"role": "assistant", "content": "Here is the answer."}, "done": True, "done_reason": "stop"}
        ),
    ]
    monkeypatch.setattr("saes_ai.chat.agent.httpx.Client", _make_client_cls([stop_empty, followup]))
    events2 = list(agent.stream_events("Hello"))
    tokens2 = "".join(e["token"] for e in events2 if e["type"] == "token")
    assert "Here is the answer." in tokens2
