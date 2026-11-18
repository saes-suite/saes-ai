"""Pydantic request and response schemas for SAES-AI REST & Streaming endpoints."""

from typing import Any

from pydantic import BaseModel, Field


class ChatHistoryItem(BaseModel):
    """Single historical message turn for multi-turn agent context."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """User query request for the statistical chat assistant."""

    query: str
    dataset_name: str | None = None
    csv_data: str | None = None
    attached_csv: str | None = None
    attached_filename: str | None = None
    current_view: str | None = None
    current_tab: str | None = None
    history: list[ChatHistoryItem] = Field(default_factory=list)
    alpha: float = 0.05
    provider: str = "local"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None


class ChatResponse(BaseModel):
    """Response from statistical chat assistant."""

    response: str
    context_used: bool
    ui_actions: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)


class LocalModelsResponse(BaseModel):
    """List of models available in local Ollama or compatible server."""

    models: list[str] = Field(default_factory=list)
    status: str = "ok"


class ProviderProbeRequest(BaseModel):
    """Request to probe AI provider connectivity and retrieve available models."""

    provider: str
    api_key: str | None = None
    base_url: str | None = None


class ProviderProbeResponse(BaseModel):
    """Response of provider connectivity probe with dynamic models list."""

    status: str = "ok"
    models: list[str] = Field(default_factory=list)
    message: str | None = None
    can_start_ollama: bool = False
    binary_path: str | None = None


class OllamaPullRequest(BaseModel):
    """Request to download a model into local Ollama daemon."""

    name: str
    base_url: str | None = None


class OllamaPullResponse(BaseModel):
    """Response of Ollama pull operation."""

    status: str = "ok"
    message: str
