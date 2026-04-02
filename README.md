# IntelliConfig

Lightweight LLM serving configuration optimizer for **vLLM + llm-d**.

Powered by [aiconfigurator](https://github.com/ai-dynamo/aiconfigurator), stripped to vLLM-only with a simplified CLI, FastAPI backend, and React UI.

## Quick Start

### Prerequisites

```bash
# Clone aiconfigurator (for perf data + engine)
git clone https://github.com/ai-dynamo/aiconfigurator.git
cd aiconfigurator && git lfs pull
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Install IntelliConfig
cd ../configmax
pip install -e .
```

### CLI

```bash
# Check model/hardware support
intelliconfig support --model Qwen/Qwen3-32B-FP8 --system h200_sxm

# Generate naive config (no sweep)
intelliconfig generate --model Qwen/Qwen3-32B-FP8 --gpus 8 --system h200_sxm

# Full recommendation (agg vs disagg sweep)
intelliconfig recommend --model Qwen/Qwen3-32B-FP8 --gpus 8 --system h200_sxm

# JSON output for scripting
intelliconfig recommend --model Qwen/Qwen3-32B-FP8 --gpus 8 --system h200_sxm --json
```

### Web UI + API Server

```bash
# Start the server (API + static UI)
intelliconfig serve --port 8000

# Dev mode (with hot reload)
intelliconfig serve --port 8000 --dev
```

Visit `http://localhost:8000` for the web UI, `http://localhost:8000/docs` for API docs.

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/systems` | List supported GPU systems |
| GET | `/api/health` | Health check |
| POST | `/api/recommend` | Full agg vs disagg sweep |
| POST | `/api/generate` | Naive config generation |
| POST | `/api/support` | Check model/system support |

## Supported Systems

| System | GPU | vLLM Version |
|--------|-----|-------------|
| `h100_sxm` | NVIDIA H100 SXM | 0.12.0 |
| `h200_sxm` | NVIDIA H200 SXM | 0.12.0 |
| `a100_sxm` | NVIDIA A100 SXM | 0.12.0 |

## Architecture

```
configmax/
├── src/intelliconfig/
│   ├── cli/          # Typer CLI (recommend, generate, support, serve)
│   ├── core/         # Engine wrapper (aiconfigurator with vLLM defaults)
│   ├── api/          # FastAPI server
│   └── generator/    # vLLM serve commands + llm-d K8s manifests
├── ui/               # React + Vite frontend
└── tests/
```

IntelliConfig wraps aiconfigurator's Python API with `backend="vllm"` hardcoded, providing:
- Simplified CLI (no need to specify `--backend`)
- REST API for the React UI
- llm-d deployment manifest generation (replaces Dynamo-specific output)

### llm-d Manifest Generator

The generator produces **real llm-d deployment manifests** based on v0.5+ patterns:

**Aggregated mode** (inference-scheduling):
- vLLM worker `Deployment` (continuous batching, single worker type)
- `InferencePool` CRD (Gateway API Inference Extension — routes to vLLM pods)
- EPP `Deployment` (inference scheduler with prefix-cache-aware routing)
- `HTTPRoute` (connects Gateway to InferencePool)

**Disaggregated mode** (prefill/decode split):
- Prefill worker `Deployment` (TP-optimized for prompt processing)
- Decode worker `Deployment` (TP-optimized for generation, NIXL KV transfer)
- `InferencePool` CRD (routes to all inference-serving pods)
- P/D-aware EPP `Deployment` (inference scheduler with PD scoring plugins)
- `HTTPRoute` (connects Gateway to InferencePool)

Component images are pinned to llm-d v0.5.x:
- vLLM: `ghcr.io/llm-d/llm-d-cuda:v0.5.1`
- EPP: `ghcr.io/llm-d/llm-d-inference-scheduler:v0.6.0`
- Routing sidecar: `ghcr.io/llm-d/llm-d-routing-sidecar:v0.6.0`

## Development

```bash
# UI dev server (hot reload, proxies API to localhost:8000)
cd ui && npm run dev

# Run API server in dev mode
intelliconfig serve --dev

# Run tests
pytest tests/
```

## License

Apache-2.0 (same as aiconfigurator)
