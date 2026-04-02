"""
IntelliConfig core engine: wraps aiconfigurator's Python API with vLLM-only defaults.

Provides recommend(), generate(), support(), and systems() as the canonical
interface for the CLI, API server, and any future consumers.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

BACKEND = "vllm"

SUPPORTED_SYSTEMS = ["h100_sxm", "h200_sxm", "a100_sxm"]

SYSTEM_LABELS = {
    "h100_sxm": "NVIDIA H100 SXM",
    "h200_sxm": "NVIDIA H200 SXM",
    "a100_sxm": "NVIDIA A100 SXM",
}


@dataclass
class RecommendResult:
    """Structured output from a recommend() call."""

    chosen_mode: str
    best_throughput_tok_s: float
    best_throughput_per_gpu: float
    best_throughput_per_user: float
    ttft_ms: float
    tpot_ms: float
    request_latency_ms: float
    agg_configs: list[dict[str, Any]]
    disagg_configs: list[dict[str, Any]]
    model_path: str
    system: str
    total_gpus: int
    isl: int
    osl: int
    speedup: float | None = None


@dataclass
class GenerateResult:
    """Structured output from a generate() call."""

    model_path: str
    system: str
    backend: str
    backend_version: str
    total_gpus: int
    tp: int
    pp: int
    replicas: int
    max_batch_size: int
    output_dir: str | None = None


@dataclass
class SupportResult:
    """Structured output from a support() call."""

    model_path: str
    system: str
    agg_supported: bool
    disagg_supported: bool


def _df_to_records(df) -> list[dict[str, Any]]:
    """Convert a pandas DataFrame to a list of dicts, handling NaN values."""
    if df is None or df.empty:
        return []
    return df.fillna("").to_dict(orient="records")


def recommend(
    model: str,
    gpus: int,
    system: str,
    *,
    isl: int = 4000,
    osl: int = 1000,
    ttft: float = 2000.0,
    tpot: float = 30.0,
    prefix: int = 0,
    top_n: int = 5,
) -> RecommendResult:
    """
    Find the best vLLM config by sweeping aggregated vs disaggregated serving.

    Wraps aiconfigurator.cli.cli_default with backend locked to vLLM.
    """
    from aiconfigurator.cli import cli_default

    result = cli_default(
        model_path=model,
        total_gpus=gpus,
        system=system,
        backend=BACKEND,
        isl=isl,
        osl=osl,
        ttft=ttft,
        tpot=tpot,
        prefix=prefix,
        top_n=top_n,
    )

    chosen = result.chosen_exp
    best_tp = result.best_throughputs.get(chosen, 0.0)
    latencies = result.best_latencies.get(chosen, {})

    agg_key = [k for k in result.best_configs if "agg" in k and "disagg" not in k]
    disagg_key = [k for k in result.best_configs if "disagg" in k]

    agg_records = _df_to_records(result.best_configs.get(agg_key[0])) if agg_key else []
    disagg_records = _df_to_records(result.best_configs.get(disagg_key[0])) if disagg_key else []

    agg_best = result.best_throughputs.get(agg_key[0], 0.0) if agg_key else 0.0
    disagg_best = result.best_throughputs.get(disagg_key[0], 0.0) if disagg_key else 0.0
    speedup = disagg_best / agg_best if agg_best > 0 else None

    best_row = {}
    if chosen in result.best_configs and not result.best_configs[chosen].empty:
        best_row = result.best_configs[chosen].iloc[0].to_dict()

    return RecommendResult(
        chosen_mode=chosen,
        best_throughput_tok_s=best_row.get("tokens/s/gpu_cluster", best_tp * gpus),
        best_throughput_per_gpu=best_row.get("tokens/s/gpu", best_tp),
        best_throughput_per_user=best_row.get("tokens/s/user", 0.0),
        ttft_ms=best_row.get("ttft", latencies.get("ttft", 0.0)),
        tpot_ms=best_row.get("tpot", latencies.get("tpot", 0.0)),
        request_latency_ms=best_row.get("request_latency", latencies.get("request_latency", 0.0)),
        agg_configs=agg_records,
        disagg_configs=disagg_records,
        model_path=model,
        system=system,
        total_gpus=gpus,
        isl=isl,
        osl=osl,
        speedup=speedup,
    )


def generate(
    model: str,
    gpus: int,
    system: str,
) -> GenerateResult:
    """
    Generate a naive (no-sweep) vLLM configuration.

    Wraps aiconfigurator.cli.cli_generate with backend locked to vLLM.
    """
    from aiconfigurator.cli import cli_generate

    raw = cli_generate(
        model_path=model,
        total_gpus=gpus,
        system=system,
        backend=BACKEND,
    )

    par = raw.get("parallelism", {})
    return GenerateResult(
        model_path=model,
        system=system,
        backend=BACKEND,
        backend_version=raw.get("backend_version", ""),
        total_gpus=gpus,
        tp=par.get("tp", 1),
        pp=par.get("pp", 1),
        replicas=par.get("replicas", 1),
        max_batch_size=par.get("max_batch_size", 512),
        output_dir=raw.get("output_dir"),
    )


def support(
    model: str,
    system: str,
) -> SupportResult:
    """
    Check whether a model/system combo is supported for agg and disagg with vLLM.
    """
    from aiconfigurator.cli import cli_support

    agg, disagg = cli_support(
        model_path=model,
        system=system,
        backend=BACKEND,
    )

    return SupportResult(
        model_path=model,
        system=system,
        agg_supported=agg,
        disagg_supported=disagg,
    )


def systems() -> list[dict[str, str]]:
    """Return the list of supported GPU systems for vLLM."""
    return [{"id": s, "label": SYSTEM_LABELS.get(s, s)} for s in SUPPORTED_SYSTEMS]
