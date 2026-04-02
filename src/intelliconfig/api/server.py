"""IntelliConfig FastAPI server: bridges the React UI to the core engine."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="IntelliConfig API",
    description="Lightweight LLM serving configuration optimizer for vLLM + llm-d",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──


class RecommendRequest(BaseModel):
    model: str = Field(..., description="HuggingFace model ID or local path")
    gpus: int = Field(..., ge=1, description="Total GPU count")
    system: str = Field(..., description="GPU system id")
    isl: int = Field(4000, ge=1)
    osl: int = Field(1000, ge=1)
    ttft: float = Field(2000.0, ge=0)
    tpot: float = Field(30.0, ge=0)
    prefix: int = Field(0, ge=0)
    top_n: int = Field(5, ge=1, le=20)


class GenerateRequest(BaseModel):
    model: str
    gpus: int = Field(..., ge=1)
    system: str


class SupportRequest(BaseModel):
    model: str
    system: str


class ManifestRequest(BaseModel):
    recommend_result: dict[str, Any] = Field(..., description="Full output from /api/recommend")
    mode_override: str | None = Field(None, description="Force 'agg' or 'disagg' instead of engine's chosen_mode")
    namespace: str = Field("llm-d", description="K8s namespace for the deployment")
    gateway_name: str = Field("inference-gateway", description="Name of the K8s Gateway resource")
    image: str | None = Field(None, description="Override the vLLM container image")
    max_model_len: int | None = Field(None, description="Override vLLM --max-model-len")


# ── Endpoints ──


@app.get("/api/systems")
def get_systems() -> list[dict[str, str]]:
    """Return list of supported GPU systems."""
    from intelliconfig.core.engine import systems
    return systems()


@app.post("/api/recommend")
def post_recommend(req: RecommendRequest) -> dict[str, Any]:
    """Run a full agg vs disagg sweep and return the best config."""
    from intelliconfig.core.engine import recommend

    try:
        result = recommend(
            model=req.model,
            gpus=req.gpus,
            system=req.system,
            isl=req.isl,
            osl=req.osl,
            ttft=req.ttft,
            tpot=req.tpot,
            prefix=req.prefix,
            top_n=req.top_n,
        )
        return asdict(result)
    except Exception as e:
        logger.exception("recommend failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate")
def post_generate(req: GenerateRequest) -> dict[str, Any]:
    """Generate a naive vLLM configuration."""
    from intelliconfig.core.engine import generate

    try:
        result = generate(model=req.model, gpus=req.gpus, system=req.system)
        return asdict(result)
    except Exception as e:
        logger.exception("generate failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/support")
def post_support(req: SupportRequest) -> dict[str, Any]:
    """Check model/system vLLM support."""
    from intelliconfig.core.engine import support

    try:
        result = support(model=req.model, system=req.system)
        return asdict(result)
    except Exception as e:
        logger.exception("support failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/manifest")
def post_manifest(req: ManifestRequest) -> dict[str, str]:
    """Generate llm-d K8s deployment manifests from a recommend result."""
    from intelliconfig.generator.llmd_manifest import (
        manifest_from_recommend,
        generate_agg_manifest,
        generate_disagg_manifest,
        VLLM_IMAGE,
    )

    try:
        result = req.recommend_result
        image = req.image or VLLM_IMAGE
        kwargs: dict[str, Any] = {
            "namespace": req.namespace,
            "gateway_name": req.gateway_name,
            "image": image,
            "max_model_len": req.max_model_len,
        }

        if req.mode_override:
            result = {**result, "chosen_mode": req.mode_override}

        yaml_text = manifest_from_recommend(result, **kwargs)
        mode = result.get("chosen_mode", "agg")
        is_disagg = "disagg" in mode

        return {
            "yaml": yaml_text,
            "mode": "disagg" if is_disagg else "agg",
            "resource_count": str(yaml_text.count("---\n") + 1),
        }
    except Exception as e:
        logger.exception("manifest generation failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}


# Mount static UI build if it exists
_ui_dist = Path(__file__).resolve().parent.parent.parent.parent / "ui" / "dist"
if _ui_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_ui_dist), html=True), name="ui")
