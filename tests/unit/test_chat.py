"""Unit tests for SAES-AI statistical chat assistant."""

import io

import saes

from saes_ai.chat.agent import StatisticalChatAgent
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
    dataset = saes.load(io.StringIO(SAMPLE_CSV))
    report = saes.analyze(dataset, generate_plots=False)

    ctx = build_analysis_context(report, dataset)
    assert "ZDT1" in ctx
    assert "HV" in ctx
    assert "NSGAII" in ctx
    assert "MOEAD" in ctx


def test_statistical_chat_agent_offline_queries() -> None:
    dataset = saes.load(io.StringIO(SAMPLE_CSV))
    report = saes.analyze(dataset, generate_plots=False)

    agent = StatisticalChatAgent(report=report, dataset=dataset)

    # Test "why" test selection query
    resp_why = agent.ask("Why was this statistical test chosen?")
    assert "Test Selection Rationale" in resp_why
    assert "ZDT1" in resp_why

    # Test "effect size" query
    resp_eff = agent.ask("What is the effect size?")
    assert "Effect Size" in resp_eff

    # Test "assumptions" query
    resp_assumptions = agent.ask("Were the normality and variance assumptions satisfied?")
    assert "Assumptions" in resp_assumptions

    # Test general summary
    resp_summary = agent.ask("Please summarize the findings.")
    assert "Statistical Findings Summary" in resp_summary


def test_statistical_chat_agent_providers_fallback() -> None:
    dataset = saes.load(io.StringIO(SAMPLE_CSV))
    report = saes.analyze(dataset, generate_plots=False)

    # Local AI provider (offline fallback if Ollama not running)
    local_agent = StatisticalChatAgent(
        report=report,
        dataset=dataset,
        provider="local",
        base_url="http://127.0.0.1:99999",  # unreachable port
    )
    resp_local = local_agent.ask("Tell me about the test results")
    assert len(resp_local) > 0

    # OpenAI provider (offline fallback if key not provided)
    openai_agent = StatisticalChatAgent(
        report=report,
        dataset=dataset,
        provider="openai",
        api_key=None,
    )
    resp_openai = openai_agent.ask("Tell me about the results")
    assert "OpenAI" in resp_openai or "Test" in resp_openai

    # Anthropic provider (offline fallback if key not provided)
    anthropic_agent = StatisticalChatAgent(
        report=report,
        dataset=dataset,
        provider="anthropic",
        api_key=None,
    )
    resp_anthropic = anthropic_agent.ask("Tell me about the results")
    assert "Anthropic" in resp_anthropic or "Test" in resp_anthropic

    # Gemini provider (offline fallback if key not provided)
    gemini_agent = StatisticalChatAgent(
        report=report,
        dataset=dataset,
        provider="gemini",
        api_key=None,
    )
    resp_gemini = gemini_agent.ask("Tell me about the results")
    assert "Google Gemini" in resp_gemini or "Test" in resp_gemini


def test_probe_provider() -> None:
    # 1. Offline probe
    res_offline = StatisticalChatAgent.probe_provider("offline")
    assert res_offline["status"] == "ok"
    assert "deterministic-statistical-engine" in res_offline["models"]

    # 2. Local probe (unreachable port)
    res_local = StatisticalChatAgent.probe_provider("local", base_url="http://127.0.0.1:99999")
    assert res_local["status"] in ("offline", "error", "ok", "binary_found_not_running")

    # 3. OpenAI probe (invalid key)
    res_openai = StatisticalChatAgent.probe_provider("openai", api_key="sk-invalid")
    assert res_openai["status"] == "error"

    # 4. Anthropic probe (invalid key)
    res_anthropic = StatisticalChatAgent.probe_provider("anthropic", api_key="sk-ant-invalid")
    assert res_anthropic["status"] == "error"

    # 5. Gemini probe (invalid key)
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
    dataset = saes.load(io.StringIO(SAMPLE_CSV))
    report = saes.analyze(dataset, generate_plots=False)

    agent = StatisticalChatAgent(report=report, dataset=dataset, provider="offline")
    tokens = list(agent.stream_ask("Why was this statistical test chosen?"))
    assert len(tokens) > 0
    full_text = "".join(tokens)
    assert "Test Selection Rationale" in full_text
