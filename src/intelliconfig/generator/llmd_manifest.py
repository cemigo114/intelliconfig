"""
Generate llm-d Kubernetes deployment manifests from IntelliConfig results.

Based on the actual llm-d deployment model (v0.5+):
  - llm-d-modelservice Helm chart: creates prefill/decode Deployments for vLLM
  - inferencepool Helm chart (GAIE): creates InferencePool CRD + EPP Deployment
  - llm-d-infra Helm chart: creates Gateway + infrastructure
  - HTTPRoute: routes Gateway traffic to InferencePool

References:
  https://github.com/llm-d/llm-d/tree/main/guides/inference-scheduling
  https://github.com/llm-d/llm-d/tree/main/guides/pd-disaggregation
  https://gateway-api-inference-extension.sigs.k8s.io/api-types/inferencepool/
"""

from __future__ import annotations

import yaml
from typing import Any

# Pinned to llm-d v0.5.x component versions
VLLM_IMAGE = "ghcr.io/llm-d/llm-d-cuda:v0.5.1"
EPP_IMAGE = "ghcr.io/llm-d/llm-d-inference-scheduler:v0.6.0"
ROUTING_SIDECAR_IMAGE = "ghcr.io/llm-d/llm-d-routing-sidecar:v0.6.0"

INFERENCEPOOL_API_VERSION = "inference.networking.k8s.io/v1"
GATEWAY_API_VERSION = "gateway.networking.k8s.io/v1"


def _model_short_name(model: str) -> str:
    return model.split("/")[-1].lower().replace(".", "-")


def _common_labels(model: str, guide: str) -> dict[str, str]:
    return {
        "llm-d.ai/inference-serving": "true",
        "llm-d.ai/model": _model_short_name(model),
        "intelliconfig.io/guide": guide,
    }


def _vllm_container(
    model: str,
    tp: int,
    *,
    port: int = 8000,
    image: str = VLLM_IMAGE,
    gpu_memory_utilization: float = 0.95,
    max_model_len: int | None = None,
    disagg: bool = False,
) -> dict[str, Any]:
    """Build a vLLM container spec matching llm-d-modelservice patterns."""
    args = [
        "--disable-uvicorn-access-log",
        f"--gpu-memory-utilization={gpu_memory_utilization}",
    ]
    if max_model_len:
        args.extend(["--max-model-len", str(max_model_len)])
    if disagg:
        args.extend([
            "--block-size", "128",
            "--kv-transfer-config",
            '{"kv_connector":"NixlConnector", "kv_role":"kv_both"}',
        ])

    container: dict[str, Any] = {
        "name": "vllm",
        "image": image,
        "command": ["python", "-m", "vllm.entrypoints.openai.api_server"],
        "args": ["--model", model, f"--tensor-parallel-size={tp}"] + args,
        "ports": [
            {"containerPort": port, "name": "vllm", "protocol": "TCP"},
        ],
        "resources": {
            "limits": {"nvidia.com/gpu": str(tp), "cpu": "16", "memory": "64Gi"},
            "requests": {"nvidia.com/gpu": str(tp), "cpu": "16", "memory": "64Gi"},
        },
        "startupProbe": {
            "httpGet": {"path": "/v1/models", "port": "vllm"},
            "initialDelaySeconds": 15,
            "periodSeconds": 30,
            "timeoutSeconds": 5,
            "failureThreshold": 60,
        },
        "livenessProbe": {
            "httpGet": {"path": "/health", "port": "vllm"},
            "periodSeconds": 10,
            "timeoutSeconds": 5,
            "failureThreshold": 3,
        },
        "readinessProbe": {
            "httpGet": {"path": "/v1/models", "port": "vllm"},
            "periodSeconds": 5,
            "timeoutSeconds": 2,
            "failureThreshold": 3,
        },
        "volumeMounts": [
            {"name": "shm", "mountPath": "/dev/shm"},
        ],
    }

    if disagg:
        container["ports"].append(
            {"containerPort": 5600, "name": "nixl", "protocol": "TCP"}
        )
        container["env"] = [
            {
                "name": "VLLM_NIXL_SIDE_CHANNEL_HOST",
                "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}},
            },
        ]

    return container


def _worker_deployment(
    name: str,
    model: str,
    tp: int,
    replicas: int,
    *,
    namespace: str,
    labels: dict[str, str],
    role: str | None = None,
    image: str = VLLM_IMAGE,
    disagg: bool = False,
    port: int = 8000,
    max_model_len: int | None = None,
) -> dict[str, Any]:
    """Build a K8s Deployment for vLLM workers (prefill or decode)."""
    pod_labels = {**labels}
    if role:
        pod_labels["llm-d.ai/role"] = role

    container = _vllm_container(
        model, tp, port=port, image=image, disagg=disagg, max_model_len=max_model_len,
    )

    volumes: list[dict[str, Any]] = [
        {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}},
    ]

    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": labels,
        },
        "spec": {
            "replicas": replicas,
            "selector": {"matchLabels": pod_labels},
            "template": {
                "metadata": {"labels": pod_labels},
                "spec": {
                    "containers": [container],
                    "volumes": volumes,
                },
            },
        },
    }


def _inference_pool(
    name: str,
    namespace: str,
    selector_labels: dict[str, str],
    epp_name: str,
    target_port: int = 8000,
) -> dict[str, Any]:
    """Build an InferencePool CRD (Gateway API Inference Extension)."""
    return {
        "apiVersion": INFERENCEPOOL_API_VERSION,
        "kind": "InferencePool",
        "metadata": {
            "name": name,
            "namespace": namespace,
        },
        "spec": {
            "targetPorts": [{"number": target_port}],
            "selector": selector_labels,
            "extensionRef": {"name": epp_name},
        },
    }


def _epp_deployment(
    name: str,
    namespace: str,
    *,
    image: str = EPP_IMAGE,
    pd_aware: bool = False,
) -> dict[str, Any]:
    """Build an EPP (Endpoint Picker / Inference Scheduler) Deployment."""
    args = ["--ext-proc-port=9002"]
    if pd_aware:
        args.append("--plugins-config-file=pd-config.yaml")

    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {"app": name},
        },
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"app": name}},
            "template": {
                "metadata": {"labels": {"app": name}},
                "spec": {
                    "containers": [{
                        "name": "epp",
                        "image": image,
                        "args": args,
                        "ports": [
                            {"containerPort": 9002, "name": "grpc", "protocol": "TCP"},
                            {"containerPort": 9090, "name": "metrics", "protocol": "TCP"},
                        ],
                    }],
                },
            },
        },
    }


def _http_route(
    name: str,
    namespace: str,
    gateway_name: str,
    pool_name: str,
    pool_port: int = 8000,
) -> dict[str, Any]:
    """Build an HTTPRoute that sends traffic to the InferencePool."""
    return {
        "apiVersion": GATEWAY_API_VERSION,
        "kind": "HTTPRoute",
        "metadata": {
            "name": name,
            "namespace": namespace,
        },
        "spec": {
            "parentRefs": [{
                "group": "gateway.networking.k8s.io",
                "kind": "Gateway",
                "name": gateway_name,
            }],
            "rules": [{
                "matches": [{"path": {"type": "PathPrefix", "value": "/"}}],
                "backendRefs": [{
                    "group": "inference.networking.x-k8s.io",
                    "kind": "InferencePool",
                    "name": pool_name,
                    "port": pool_port,
                    "weight": 1,
                }],
                "timeouts": {
                    "backendRequest": "0s",
                    "request": "0s",
                },
            }],
        },
    }


def generate_agg_manifest(
    model: str,
    tp: int,
    replicas: int,
    *,
    namespace: str = "llm-d",
    image: str = VLLM_IMAGE,
    gateway_name: str = "inference-gateway",
    max_model_len: int | None = None,
) -> str:
    """
    Generate llm-d manifests for aggregated (non-disaggregated) vLLM serving.

    Produces:
      1. vLLM worker Deployment (decode-only, standard continuous batching)
      2. InferencePool CRD (routes to vLLM pods via label selector)
      3. EPP Deployment (inference scheduler for prefix-cache-aware routing)
      4. HTTPRoute (connects Gateway to InferencePool)
    """
    short = _model_short_name(model)
    labels = _common_labels(model, "inference-scheduling")

    deploy = _worker_deployment(
        name=f"vllm-{short}",
        model=model,
        tp=tp,
        replicas=replicas,
        namespace=namespace,
        labels=labels,
        image=image,
        max_model_len=max_model_len,
    )

    pool_name = f"pool-{short}"
    epp_name = f"epp-{short}"

    pool = _inference_pool(
        name=pool_name,
        namespace=namespace,
        selector_labels={
            "llm-d.ai/inference-serving": "true",
            "llm-d.ai/model": short,
        },
        epp_name=epp_name,
    )

    epp = _epp_deployment(name=epp_name, namespace=namespace)

    route = _http_route(
        name=f"route-{short}",
        namespace=namespace,
        gateway_name=gateway_name,
        pool_name=pool_name,
    )

    docs = [deploy, pool, epp, route]
    return "---\n".join(yaml.dump(d, default_flow_style=False, sort_keys=False) for d in docs)


def generate_disagg_manifest(
    model: str,
    prefill_tp: int,
    prefill_workers: int,
    decode_tp: int,
    decode_workers: int,
    *,
    namespace: str = "llm-d",
    image: str = VLLM_IMAGE,
    gateway_name: str = "inference-gateway",
    max_model_len: int | None = None,
) -> str:
    """
    Generate llm-d manifests for disaggregated (prefill/decode split) serving.

    Produces:
      1. Prefill Deployment (TP-optimized for prompt processing)
      2. Decode Deployment (TP-optimized for autoregressive generation, with NIXL)
      3. InferencePool CRD (routes to all inference-serving pods)
      4. EPP Deployment (P/D-aware inference scheduler)
      5. HTTPRoute (connects Gateway to InferencePool)

    Both prefill and decode workers are configured with NIXL KV-cache transfer
    connectors for zero-copy KV handoff over RDMA/RoCE.
    """
    short = _model_short_name(model)
    base_labels = _common_labels(model, "pd-disaggregation")

    prefill_labels = {**base_labels, "llm-d.ai/role": "prefill"}
    decode_labels = {**base_labels, "llm-d.ai/role": "decode"}

    prefill_deploy = _worker_deployment(
        name=f"vllm-prefill-{short}",
        model=model,
        tp=prefill_tp,
        replicas=prefill_workers,
        namespace=namespace,
        labels=prefill_labels,
        role="prefill",
        image=image,
        disagg=True,
        max_model_len=max_model_len,
    )

    decode_deploy = _worker_deployment(
        name=f"vllm-decode-{short}",
        model=model,
        tp=decode_tp,
        replicas=decode_workers,
        namespace=namespace,
        labels=decode_labels,
        role="decode",
        image=image,
        disagg=True,
        port=8200,
        max_model_len=max_model_len,
    )

    pool_name = f"pool-{short}"
    epp_name = f"epp-{short}"

    pool = _inference_pool(
        name=pool_name,
        namespace=namespace,
        selector_labels={
            "llm-d.ai/inference-serving": "true",
            "llm-d.ai/model": short,
        },
        epp_name=epp_name,
    )

    epp = _epp_deployment(name=epp_name, namespace=namespace, pd_aware=True)

    route = _http_route(
        name=f"route-{short}",
        namespace=namespace,
        gateway_name=gateway_name,
        pool_name=pool_name,
    )

    docs = [prefill_deploy, decode_deploy, pool, epp, route]
    return "---\n".join(yaml.dump(d, default_flow_style=False, sort_keys=False) for d in docs)


def manifest_from_recommend(result: dict[str, Any], **kwargs: Any) -> str:
    """Generate the appropriate llm-d manifests from a recommend result."""
    chosen = result.get("chosen_mode", "agg")
    model = result["model_path"]

    if "disagg" in chosen and result.get("disagg_configs"):
        top = result["disagg_configs"][0]
        return generate_disagg_manifest(
            model=model,
            prefill_tp=int(top.get("(p)tp", top.get("tp", 1))),
            prefill_workers=int(top.get("(p)workers", 1)),
            decode_tp=int(top.get("(d)tp", top.get("tp", 1))),
            decode_workers=int(top.get("(d)workers", 1)),
            **kwargs,
        )

    configs = result.get("agg_configs", [])
    if configs:
        top = configs[0]
        tp = int(top.get("tp", 1))
        replicas = int(top.get("replicas", 1))
    else:
        tp = 1
        replicas = result.get("total_gpus", 1)

    return generate_agg_manifest(model=model, tp=tp, replicas=replicas, **kwargs)
