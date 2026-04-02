"""Integration tests for IntelliConfig core engine."""

import pytest
from intelliconfig.core.engine import recommend, generate, support, systems


def test_systems_returns_supported_gpus():
    result = systems()
    assert len(result) >= 3
    ids = [s["id"] for s in result]
    assert "h100_sxm" in ids
    assert "h200_sxm" in ids
    assert "a100_sxm" in ids


def test_support_qwen3_h200():
    result = support(model="Qwen/Qwen3-32B-FP8", system="h200_sxm")
    assert result.model_path == "Qwen/Qwen3-32B-FP8"
    assert result.system == "h200_sxm"
    assert result.agg_supported is True
    assert result.disagg_supported is True


def test_generate_qwen3_h200():
    result = generate(model="Qwen/Qwen3-32B-FP8", gpus=8, system="h200_sxm")
    assert result.model_path == "Qwen/Qwen3-32B-FP8"
    assert result.backend == "vllm"
    assert result.total_gpus == 8
    assert result.tp >= 1
    assert result.replicas >= 1


def test_recommend_qwen3_h200():
    result = recommend(
        model="Qwen/Qwen3-32B-FP8",
        gpus=8,
        system="h200_sxm",
        isl=4000,
        osl=1000,
        ttft=2000.0,
        tpot=30.0,
    )
    assert result.model_path == "Qwen/Qwen3-32B-FP8"
    assert result.system == "h200_sxm"
    assert result.chosen_mode in ("agg", "disagg")
    assert result.best_throughput_per_gpu > 0
    assert result.ttft_ms > 0
    assert result.tpot_ms > 0
    assert len(result.agg_configs) > 0 or len(result.disagg_configs) > 0


def test_recommend_all_outputs_are_vllm():
    """Verify no TensorRT-LLM or SGLang references leak through."""
    result = recommend(
        model="Qwen/Qwen3-32B-FP8",
        gpus=8,
        system="h200_sxm",
    )
    for configs in [result.agg_configs, result.disagg_configs]:
        for cfg in configs:
            backend = cfg.get("backend", cfg.get("(p)backend", ""))
            if backend:
                assert backend == "vllm", f"Non-vLLM backend found: {backend}"
