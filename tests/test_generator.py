"""Tests for the llm-d manifest generator."""

import yaml
import pytest
from intelliconfig.generator.llmd_manifest import (
    generate_agg_manifest,
    generate_disagg_manifest,
    manifest_from_recommend,
)


def _parse_all(text: str) -> list[dict]:
    return list(yaml.safe_load_all(text))


class TestAggregatedManifest:
    def test_produces_four_documents(self):
        text = generate_agg_manifest("meta-llama/Llama-3.1-8B", tp=2, replicas=4)
        docs = _parse_all(text)
        assert len(docs) == 4

    def test_contains_deployment(self):
        docs = _parse_all(generate_agg_manifest("meta-llama/Llama-3.1-8B", tp=2, replicas=4))
        deploy = docs[0]
        assert deploy["kind"] == "Deployment"
        assert deploy["apiVersion"] == "apps/v1"
        assert deploy["spec"]["replicas"] == 4

    def test_contains_inference_pool(self):
        docs = _parse_all(generate_agg_manifest("meta-llama/Llama-3.1-8B", tp=2, replicas=4))
        pool = docs[1]
        assert pool["kind"] == "InferencePool"
        assert pool["spec"]["targetPorts"][0]["number"] == 8000
        assert "selector" in pool["spec"]
        assert "extensionRef" in pool["spec"]

    def test_contains_epp_deployment(self):
        docs = _parse_all(generate_agg_manifest("meta-llama/Llama-3.1-8B", tp=2, replicas=4))
        epp = docs[2]
        assert epp["kind"] == "Deployment"
        assert "epp" in epp["metadata"]["name"]
        assert epp["spec"]["template"]["spec"]["containers"][0]["image"].startswith(
            "ghcr.io/llm-d/llm-d-inference-scheduler"
        )

    def test_contains_http_route(self):
        docs = _parse_all(generate_agg_manifest("meta-llama/Llama-3.1-8B", tp=2, replicas=4))
        route = docs[3]
        assert route["kind"] == "HTTPRoute"
        assert route["apiVersion"] == "gateway.networking.k8s.io/v1"

    def test_gpu_resources_match_tp(self):
        docs = _parse_all(generate_agg_manifest("meta-llama/Llama-3.1-8B", tp=4, replicas=2))
        container = docs[0]["spec"]["template"]["spec"]["containers"][0]
        assert container["resources"]["limits"]["nvidia.com/gpu"] == "4"

    def test_custom_namespace(self):
        docs = _parse_all(generate_agg_manifest("meta-llama/Llama-3.1-8B", tp=2, replicas=4, namespace="prod"))
        for doc in docs:
            assert doc["metadata"]["namespace"] == "prod"

    def test_labels_include_model(self):
        docs = _parse_all(generate_agg_manifest("meta-llama/Llama-3.1-8B", tp=2, replicas=4))
        deploy = docs[0]
        assert deploy["metadata"]["labels"]["llm-d.ai/model"] == "llama-3-1-8b"
        assert deploy["metadata"]["labels"]["llm-d.ai/inference-serving"] == "true"

    def test_vllm_probes_present(self):
        docs = _parse_all(generate_agg_manifest("meta-llama/Llama-3.1-8B", tp=2, replicas=4))
        container = docs[0]["spec"]["template"]["spec"]["containers"][0]
        assert "startupProbe" in container
        assert "livenessProbe" in container
        assert "readinessProbe" in container


class TestDisaggregatedManifest:
    def test_produces_five_documents(self):
        text = generate_disagg_manifest(
            "openai/gpt-oss-120b",
            prefill_tp=1, prefill_workers=4,
            decode_tp=4, decode_workers=1,
        )
        docs = _parse_all(text)
        assert len(docs) == 5

    def test_prefill_and_decode_deployments(self):
        docs = _parse_all(generate_disagg_manifest(
            "openai/gpt-oss-120b",
            prefill_tp=1, prefill_workers=4,
            decode_tp=4, decode_workers=1,
        ))
        prefill = docs[0]
        decode = docs[1]
        assert prefill["kind"] == "Deployment"
        assert "prefill" in prefill["metadata"]["name"]
        assert prefill["spec"]["replicas"] == 4

        assert decode["kind"] == "Deployment"
        assert "decode" in decode["metadata"]["name"]
        assert decode["spec"]["replicas"] == 1

    def test_nixl_kv_transfer_config(self):
        docs = _parse_all(generate_disagg_manifest(
            "openai/gpt-oss-120b",
            prefill_tp=1, prefill_workers=4,
            decode_tp=4, decode_workers=1,
        ))
        for deploy in docs[:2]:
            container = deploy["spec"]["template"]["spec"]["containers"][0]
            args = container["args"]
            kv_arg = [a for a in args if "NixlConnector" in str(a)]
            assert len(kv_arg) > 0, f"Missing NIXL config in {deploy['metadata']['name']}"

    def test_epp_is_pd_aware(self):
        docs = _parse_all(generate_disagg_manifest(
            "openai/gpt-oss-120b",
            prefill_tp=1, prefill_workers=4,
            decode_tp=4, decode_workers=1,
        ))
        epp = docs[3]
        container = epp["spec"]["template"]["spec"]["containers"][0]
        assert any("pd-config" in str(a) for a in container["args"])

    def test_role_labels(self):
        docs = _parse_all(generate_disagg_manifest(
            "openai/gpt-oss-120b",
            prefill_tp=1, prefill_workers=4,
            decode_tp=4, decode_workers=1,
        ))
        prefill_labels = docs[0]["metadata"]["labels"]
        decode_labels = docs[1]["metadata"]["labels"]
        assert prefill_labels["llm-d.ai/role"] == "prefill"
        assert decode_labels["llm-d.ai/role"] == "decode"

    def test_nixl_port_present(self):
        docs = _parse_all(generate_disagg_manifest(
            "openai/gpt-oss-120b",
            prefill_tp=1, prefill_workers=4,
            decode_tp=4, decode_workers=1,
        ))
        for deploy in docs[:2]:
            container = deploy["spec"]["template"]["spec"]["containers"][0]
            port_names = [p["name"] for p in container["ports"]]
            assert "nixl" in port_names


class TestManifestFromRecommend:
    def test_agg_result(self):
        result = {
            "chosen_mode": "agg",
            "model_path": "meta-llama/Llama-3.1-8B",
            "total_gpus": 8,
            "agg_configs": [{"tp": 2, "replicas": 4}],
            "disagg_configs": [],
        }
        text = manifest_from_recommend(result)
        docs = _parse_all(text)
        assert len(docs) == 4
        assert docs[0]["kind"] == "Deployment"
        assert docs[0]["spec"]["replicas"] == 4

    def test_disagg_result(self):
        result = {
            "chosen_mode": "disagg",
            "model_path": "openai/gpt-oss-120b",
            "total_gpus": 8,
            "agg_configs": [],
            "disagg_configs": [{
                "(p)tp": 1, "(p)workers": 4,
                "(d)tp": 4, "(d)workers": 1,
            }],
        }
        text = manifest_from_recommend(result)
        docs = _parse_all(text)
        assert len(docs) == 5
        assert "prefill" in docs[0]["metadata"]["name"]
        assert "decode" in docs[1]["metadata"]["name"]
