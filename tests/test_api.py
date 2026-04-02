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


def test_manifest_agg():
    fake_result = {
        "chosen_mode": "agg",
        "model_path": "meta-llama/Llama-3.1-8B",
        "total_gpus": 8,
        "agg_configs": [{"tp": 2, "replicas": 4}],
        "disagg_configs": [],
    }
    r = client.post("/api/manifest", json={
        "recommend_result": fake_result,
        "namespace": "test-ns",
    })
    assert r.status_code == 200
    data = r.json()
    assert data["mode"] == "agg"
    assert "InferencePool" in data["yaml"]
    assert "HTTPRoute" in data["yaml"]
    assert "test-ns" in data["yaml"]


def test_manifest_disagg():
    fake_result = {
        "chosen_mode": "disagg",
        "model_path": "openai/gpt-oss-120b",
        "total_gpus": 8,
        "agg_configs": [],
        "disagg_configs": [{"(p)tp": 1, "(p)workers": 4, "(d)tp": 4, "(d)workers": 1}],
    }
    r = client.post("/api/manifest", json={
        "recommend_result": fake_result,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["mode"] == "disagg"
    assert "NixlConnector" in data["yaml"]
    assert "prefill" in data["yaml"]
    assert "decode" in data["yaml"]


def test_manifest_mode_override():
    fake_result = {
        "chosen_mode": "disagg",
        "model_path": "meta-llama/Llama-3.1-8B",
        "total_gpus": 4,
        "agg_configs": [{"tp": 2, "replicas": 2}],
        "disagg_configs": [{"(p)tp": 1, "(p)workers": 2, "(d)tp": 2, "(d)workers": 1}],
    }
    r = client.post("/api/manifest", json={
        "recommend_result": fake_result,
        "mode_override": "agg",
    })
    assert r.status_code == 200
    data = r.json()
    assert data["mode"] == "agg"
    assert "NixlConnector" not in data["yaml"]
