"""Statistical chat assistant REST & Streaming endpoints for SAES-AI."""

import io
import json
import logging
from collections.abc import Generator
from pathlib import Path

import saes
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from saes.server.storage import get_datasets_dir

from saes_ai.chat.agent import StatisticalChatAgent
from saes_ai.chat.tools import ToolContext
from saes_ai.server.schemas import (
    ChatRequest,
    ChatResponse,
    LocalModelsResponse,
    OllamaPullRequest,
    OllamaPullResponse,
    ProviderProbeRequest,
    ProviderProbeResponse,
)

logger = logging.getLogger("saes_ai.server.routes")

router = APIRouter(prefix="/api/v1", tags=["Chat"])


@router.get("/chat/models/local", response_model=LocalModelsResponse)
def get_local_models(base_url: str = "http://localhost:11434") -> LocalModelsResponse:
    """Fetch installed models from a local Ollama or compatible server."""
    models = StatisticalChatAgent.fetch_local_models(base_url=base_url)
    return LocalModelsResponse(models=models, status="ok" if models else "empty")


@router.post("/chat/probe", response_model=ProviderProbeResponse)
def probe_provider_connection(req: ProviderProbeRequest) -> ProviderProbeResponse:
    """Test connectivity to any AI provider and retrieve available models."""
    res = StatisticalChatAgent.probe_provider(
        provider=req.provider,
        api_key=req.api_key,
        base_url=req.base_url,
    )
    return ProviderProbeResponse(**res)


@router.post("/chat/ollama/start")
def start_ollama_daemon() -> dict[str, bool]:
    """Attempt to start the local Ollama background server daemon."""
    started = StatisticalChatAgent.start_ollama_service()
    return {"started": started}


@router.post("/chat/ollama/pull", response_model=OllamaPullResponse)
def pull_ollama_model(req: OllamaPullRequest) -> OllamaPullResponse:
    """Download / pull a model into the local Ollama server."""
    res = StatisticalChatAgent.pull_ollama_model(
        name=req.name,
        base_url=req.base_url or "http://localhost:11434",
    )
    return OllamaPullResponse(**res)


@router.post("/chat/ollama/pull/stream")
def pull_ollama_model_stream(req: OllamaPullRequest) -> StreamingResponse:
    """Download / pull a model into the local Ollama server with progress streaming."""
    generator = StatisticalChatAgent.stream_pull_ollama_model(
        name=req.name,
        base_url=req.base_url or "http://localhost:11434",
    )
    return StreamingResponse(generator, media_type="application/x-ndjson")


def _build_agent_for_request(req: ChatRequest) -> tuple[StatisticalChatAgent, bool]:
    """Initialize StatisticalChatAgent and ToolContext for a chat request."""
    dataset = None
    report = None
    datasets_dir = get_datasets_dir()

    if req.dataset_name:
        try:
            safe_name = Path(req.dataset_name).name
            file_path = datasets_dir / safe_name
            if file_path.exists() and file_path.is_file():
                dataset = saes.load(file_path)
                val = saes.validate(dataset)
                if val.is_valid:
                    report = saes.analyze(dataset, alpha=req.alpha, generate_plots=False)
                else:
                    logger.debug("Dataset %s failed validation: %s", safe_name, val.errors)
        except Exception as exc:
            logger.debug("Failed loading dataset %s: %s", req.dataset_name, exc)

    if dataset is None and req.csv_data:
        try:
            dataset = saes.load(io.StringIO(req.csv_data))
            val = saes.validate(dataset)
            if val.is_valid:
                report = saes.analyze(dataset, alpha=req.alpha, generate_plots=False)
        except Exception as exc:
            logger.debug("Failed loading inline csv_data: %s", exc)

    logger.debug(
        "Chat request: provider=%s model=%s dataset=%s report=%s history=%d view=%s/%s query=%r",
        req.provider,
        req.model,
        req.dataset_name or ("<csv_data>" if req.csv_data else None),
        report is not None,
        len(req.history),
        req.current_view,
        req.current_tab,
        req.query[:120],
    )

    tool_ctx = ToolContext(
        datasets_dir=datasets_dir,
        active_dataset_name=Path(req.dataset_name).name if req.dataset_name else None,
        staged_csv=req.attached_csv,
        staged_filename=req.attached_filename,
        alpha=req.alpha,
        current_view=req.current_view,
        current_tab=req.current_tab,
    )
    history = [{"role": h.role, "content": h.content} for h in req.history]

    agent = StatisticalChatAgent(
        report=report,
        dataset=dataset,
        provider=req.provider,
        model=req.model,
        api_key=req.api_key,
        base_url=req.base_url,
        tool_context=tool_ctx,
        history=history,
    )
    return agent, report is not None


@router.post("/chat", response_model=ChatResponse)
def chat_with_agent(req: ChatRequest) -> ChatResponse:
    """Ask a question to the statistical AI agent."""
    agent, context_used = _build_agent_for_request(req)
    answer = agent.ask(req.query)
    return ChatResponse(
        response=answer,
        context_used=context_used or bool(agent.tool_context.tool_events),
        ui_actions=agent.tool_context.ui_actions,
        tool_calls=agent.tool_context.tool_events,
        artifacts=agent.tool_context.artifacts,
    )


@router.post("/chat/stream")
def chat_with_agent_stream(req: ChatRequest) -> StreamingResponse:
    """Ask a question to the statistical AI agent with live token and tool event streaming."""
    agent, base_context_used = _build_agent_for_request(req)

    def event_generator() -> Generator[str, None, None]:
        try:
            for event in agent.stream_events(req.query):
                ev_type = event.get("type")
                ctx_used = base_context_used or bool(agent.tool_context.tool_events)
                if ev_type == "token":
                    yield (
                        json.dumps(
                            {
                                "type": "token",
                                "token": event.get("token", ""),
                                "done": False,
                                "context_used": ctx_used,
                            }
                        )
                        + "\n"
                    )
                elif ev_type == "thought":
                    yield (
                        json.dumps(
                            {
                                "type": "thought",
                                "thought": event.get("thought", ""),
                                "done": False,
                                "context_used": ctx_used,
                            }
                        )
                        + "\n"
                    )
                elif ev_type == "tool_call":
                    yield (
                        json.dumps(
                            {
                                "type": "tool_call",
                                "tool": event.get("tool"),
                                "arguments": event.get("arguments", {}),
                                "done": False,
                                "context_used": ctx_used,
                            }
                        )
                        + "\n"
                    )
                elif ev_type == "tool_result":
                    yield (
                        json.dumps(
                            {
                                "type": "tool_result",
                                "tool": event.get("tool"),
                                "ok": event.get("ok", True),
                                "summary": event.get("summary", ""),
                                "done": False,
                                "context_used": ctx_used,
                            }
                        )
                        + "\n"
                    )
                elif ev_type == "ui_action":
                    yield (
                        json.dumps(
                            {
                                "type": "ui_action",
                                "action": event.get("action", {}),
                                "done": False,
                                "context_used": ctx_used,
                            }
                        )
                        + "\n"
                    )
                elif ev_type == "artifact":
                    yield (
                        json.dumps(
                            {
                                "type": "artifact",
                                "artifact": event.get("artifact", {}),
                                "done": False,
                                "context_used": ctx_used,
                            }
                        )
                        + "\n"
                    )

            final_ctx = base_context_used or bool(agent.tool_context.tool_events)
            logger.debug(
                "Chat stream done: tools=%d artifacts=%d ui_actions=%d context_used=%s",
                len(agent.tool_context.tool_events),
                len(agent.tool_context.artifacts),
                len(agent.tool_context.ui_actions),
                final_ctx,
            )
            yield (
                json.dumps(
                    {
                        "type": "done",
                        "token": "",
                        "done": True,
                        "context_used": final_ctx,
                        "ui_actions": agent.tool_context.ui_actions,
                        "artifacts": agent.tool_context.artifacts,
                    }
                )
                + "\n"
            )
        except Exception as e:
            logger.exception("Chat stream error: %s", e)
            yield (
                json.dumps(
                    {
                        "type": "error",
                        "error": str(e),
                        "token": "",
                        "done": True,
                        "context_used": base_context_used,
                    }
                )
                + "\n"
            )

    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
