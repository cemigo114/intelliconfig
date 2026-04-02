"""Generate vLLM serve commands from IntelliConfig recommendation results."""

from __future__ import annotations

from typing import Any


def build_vllm_serve_cmd(
    model: str,
    tp: int = 1,
    pp: int = 1,
    *,
    max_num_batched_tokens: int | None = None,
    max_num_seqs: int | None = None,
    gpu_memory_utilization: float = 0.92,
    kv_cache_dtype: str = "auto",
    enable_chunked_prefill: bool = False,
    enable_prefix_caching: bool = False,
    port: int = 8000,
    extra_args: dict[str, Any] | None = None,
) -> str:
    """Build a vllm serve command string."""
    parts = [
        "vllm serve",
        model,
        f"--tensor-parallel-size {tp}",
    ]
    if pp > 1:
        parts.append(f"--pipeline-parallel-size {pp}")
    if max_num_batched_tokens:
        parts.append(f"--max-num-batched-tokens {max_num_batched_tokens}")
    if max_num_seqs:
        parts.append(f"--max-num-seqs {max_num_seqs}")
    parts.append(f"--gpu-memory-utilization {gpu_memory_utilization}")
    if kv_cache_dtype != "auto":
        parts.append(f"--kv-cache-dtype {kv_cache_dtype}")
    if enable_chunked_prefill:
        parts.append("--enable-chunked-prefill")
    if enable_prefix_caching:
        parts.append("--enable-prefix-caching")
    parts.append(f"--port {port}")

    if extra_args:
        for k, v in extra_args.items():
            if isinstance(v, bool):
                if v:
                    parts.append(f"--{k}")
            else:
                parts.append(f"--{k} {v}")

    return " \\\n  ".join(parts)


def vllm_serve_from_recommend(result: dict[str, Any]) -> str:
    """
    Extract the top config from a recommend result and produce a vllm serve command.
    """
    chosen = result.get("chosen_mode", "agg")
    configs = result.get("disagg_configs" if "disagg" in chosen else "agg_configs", [])
    if not configs:
        return build_vllm_serve_cmd(result["model_path"])

    top = configs[0]
    tp = int(top.get("tp", top.get("(p)tp", 1)))
    pp = int(top.get("pp", top.get("(p)pp", 1)))

    return build_vllm_serve_cmd(
        model=result["model_path"],
        tp=tp,
        pp=pp,
    )
