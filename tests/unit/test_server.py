"""Unit tests for SAES-AI FastAPI REST endpoints."""

import json

from fastapi.testclient import TestClient

from saes_ai.server.app import app

client = TestClient(app)

SAMPLE_CSV = """algorithm,instance,metric,run_id,value,maximize
NSGAII,ZDT1,HV,0,0.662145,True
NSGAII,ZDT1,HV,1,0.661890,True
NSGAII,ZDT1,HV,2,0.662012,True
MOEAD,ZDT1,HV,0,0.665421,True
MOEAD,ZDT1,HV,1,0.665100,True
MOEAD,ZDT1,HV,2,0.665900,True
"""


def test_health_endpoint() -> None:
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    assert res.json()["service"] == "saes-ai-api"


def test_chat_endpoint() -> None:
    res = client.post(
        "/api/v1/chat",
        json={
            "query": "Why was this test chosen?",
            "csv_data": SAMPLE_CSV,
            "alpha": 0.05,
            "provider": "offline",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "response" in data
    assert data["context_used"] is True


def test_chat_stream_endpoint() -> None:
    res = client.post(
        "/api/v1/chat/stream",
        json={
            "query": "Why was this test chosen?",
            "csv_data": SAMPLE_CSV,
            "alpha": 0.05,
            "provider": "offline",
        },
    )
    assert res.status_code == 200
    assert "token" in res.text
    tokens = [json.loads(line)["token"] for line in res.text.strip().split("\n") if line.strip()]
    full_text = "".join(tokens)
    assert "Test Selection Rationale" in full_text


def test_get_local_models_endpoint() -> None:
    res = client.get("/api/v1/chat/models/local?base_url=http://127.0.0.1:99999")
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    assert data["status"] in ("ok", "empty")


def test_probe_provider_endpoint() -> None:
    # 1. Offline probe
    res_off = client.post("/api/v1/chat/probe", json={"provider": "offline"})
    assert res_off.status_code == 200
    data_off = res_off.json()
    assert data_off["status"] == "ok"
    assert "deterministic-statistical-engine" in data_off["models"]

    # 2. Local probe (unreachable port)
    res_local = client.post(
        "/api/v1/chat/probe",
        json={"provider": "local", "base_url": "http://127.0.0.1:99999"},
    )
    assert res_local.status_code == 200


def test_start_ollama_daemon_endpoint() -> None:
    res = client.post("/api/v1/chat/ollama/start")
    assert res.status_code == 200
    assert "started" in res.json()


def test_pull_ollama_endpoint() -> None:
    res = client.post(
        "/api/v1/chat/ollama/pull",
        json={"name": "llama3.2", "base_url": "http://127.0.0.1:99999"},
    )
    assert res.status_code == 200
    assert res.json()["status"] == "error"


def test_pull_ollama_stream_endpoint() -> None:
    res = client.post(
        "/api/v1/chat/ollama/pull/stream",
        json={"name": "llama3.2", "base_url": "http://127.0.0.1:99999"},
    )
    assert res.status_code == 200
    lines = [json.loads(line) for line in res.text.strip().split("\n") if line.strip()]
    assert len(lines) >= 1
    assert "error" in lines[0]
