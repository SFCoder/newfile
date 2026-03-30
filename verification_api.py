"""
FastAPI verification server for Qwen/Qwen2.5-7B.

This module is purely HTTP infrastructure: model loading, lifecycle management,
and endpoint wiring.  All verification logic lives in verifier.py.
All model loading goes through ModelRegistry, which verifies weight hashes
before returning a model instance.

Endpoints
---------
GET  /health   — liveness check, returns device and model metadata
POST /verify   — accepts a VerificationBundle, returns a VerificationResult
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from verifier import VerificationBundle, VerificationResult, verify
from model_registry import ModelRegistry, DEFAULT_REGISTRY_PATH

logger = logging.getLogger(__name__)

SUPPORTED_MODEL = "Qwen/Qwen2.5-7B"

# ---------------------------------------------------------------------------
# Model singleton — loaded and hash-verified via ModelRegistry
# ---------------------------------------------------------------------------

_MODEL = None
_TOKENIZER = None
_LOADED_MODEL_NAME: str = ""


def load_model(model_name: str = SUPPORTED_MODEL, registry: ModelRegistry = None):
    """
    Load model and tokenizer via ModelRegistry (verifies weight hash first).
    Idempotent — returns the cached singleton on repeated calls.
    """
    global _MODEL, _TOKENIZER, _LOADED_MODEL_NAME
    if _MODEL is not None:
        return _MODEL, _TOKENIZER

    if registry is None:
        registry = ModelRegistry(DEFAULT_REGISTRY_PATH)

    _MODEL, _TOKENIZER = registry.load_verified_model(model_name)
    _LOADED_MODEL_NAME = model_name
    return _MODEL, _TOKENIZER


def get_model():
    """Return (model, tokenizer).  Raises RuntimeError if not yet loaded."""
    if _MODEL is None:
        raise RuntimeError("Model not loaded — call load_model() first.")
    return _MODEL, _TOKENIZER


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_model(SUPPORTED_MODEL)
    yield


app = FastAPI(
    title="Sparse Replay Verifier",
    description=(
        "Accepts a VerificationBundle (prompt + claimed tokens + neuron masks), "
        "replays the sparse computation against real model weights, and returns "
        "whether the output matches."
    ),
    lifespan=lifespan,
)


@app.get("/health")
def health():
    model, _ = get_model()
    return {
        "status": "ok",
        "model": _LOADED_MODEL_NAME,
        "device": str(next(model.parameters()).device),
        "num_layers": len(model.model.layers),
        "intermediate_size": model.config.intermediate_size,
    }


@app.post("/verify", response_model=VerificationResult)
def verify_endpoint(bundle: VerificationBundle):
    if bundle.model_name != SUPPORTED_MODEL:
        raise HTTPException(
            status_code=400,
            detail=(
                f"This server only supports {SUPPORTED_MODEL!r}. "
                f"Received {bundle.model_name!r}."
            ),
        )
    model, tokenizer = get_model()
    return verify(bundle, model, tokenizer)
