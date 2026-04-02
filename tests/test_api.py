"""Integration tests for IntelliConfig FastAPI server."""

import pytest
from fastapi.testclient import TestClient
from intelliconfig.api.server import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_systems():
    r = client.get("/api/systems")
    assert r.status_code == 200
    data = r.json()
    assert len(data) >= 3
    assert any(s["id"] == "h200_sxm" for s in data)


def test_support():
    r = client.post("/api/support", json={"model": "Qwen/Qwen3-32B-FP8", "system": "h200_sxm"})
    assert r.status_code == 200
    data = r.json()
    assert data["agg_supported"] is True
    assert data["disagg_supported"] is True


def test_generate():
    r = client.post("/api/generate", json={"model": "Qwen/Qwen3-32B-FP8", "gpus": 8, "system": "h200_sxm"})
    assert r.status_code == 200
    data = r.json()
    assert data["backend"] == "vllm"
    assert data["tp"] >= 1


def test_recommend():
    r = client.post("/api/recommend", json={
        "model": "Qwen/Qwen3-32B-FP8",
        "gpus": 8,
        "system": "h200_sxm",
        "isl": 4000,
        "osl": 1000,
        "ttft": 2000.0,
        "tpot": 30.0,
        "prefix": 0,
        "top_n": 3,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["chosen_mode"] in ("agg", "disagg")
    assert data["best_throughput_per_gpu"] > 0
    for cfg in data.get("agg_configs", []) + data.get("disagg_configs", []):
        backend = cfg.get("backend", cfg.get("(p)backend", ""))
        if backend:
            assert backend == "vllm"
