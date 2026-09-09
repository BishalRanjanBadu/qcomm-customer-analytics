"""FastAPI serving layer.

Design points that are not defaults:

  * NO field defaults. A caller who omits an input gets 422, not a confident 200 computed
    from schema placeholders.
  * /live and /health are split. Liveness touches no I/O; readiness validates the artifact.
  * The artifact loads at STARTUP, not lazily. A lazy global races across workers and
    readiness passes before the model exists, so traffic reaches a pod that cannot serve.
  * A failed reload sets the cached artifact to None. Leaving the old object in place means
    a pod serves the old model while reporting the new version — a silent provenance lie.
"""
from __future__ import annotations

import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from .predict import PredictionError, score
from .preprocess import ContractViolation
from .registry import Artifact, ManifestMismatch, load_current

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}')
log = logging.getLogger("qcomm")

_ART: Artifact | None = None
_LOAD_ERROR: str | None = None

# The 45-day feature window bounds recency and tenure. Values outside it did not come
# from this pipeline.
FEATURE_WINDOW_DAYS = 45
ALLOWED_ORIGINS = [o for o in os.environ.get("CORS_ORIGINS", "").split(",") if o]


def _load(strict: bool = True) -> None:
    global _ART, _LOAD_ERROR
    try:
        _ART = load_current(strict_libs=strict)
        _LOAD_ERROR = None
        log.info(f"loaded artifact {_ART.version} from {_ART.prefix}")
    except Exception as e:
        _ART = None                      # clear it — never serve a stale model as a new one
        _LOAD_ERROR = f"{type(e).__name__}: {e}"
        log.error(f"artifact load FAILED: {_LOAD_ERROR}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load(strict=os.environ.get("STRICT_LIBS", "true").lower() == "true")
    yield


app = FastAPI(title="QCOMM RFM Segmentation", version="1.0.0", lifespan=lifespan)
if ALLOWED_ORIGINS:
    app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS,
                       allow_methods=["GET", "POST"], allow_headers=["*"])


class Customer(BaseModel):
    """Every field required. No defaults — see the module docstring."""
    customer_id: str | None = Field(None, description="Echoed back. Not a model input.")
    recency_days: float = Field(..., description="Days since last order at the snapshot")
    frequency: int = Field(..., description="Orders in the 45-day window")
    monetary_mean: float = Field(..., description="Mean order value, INR")

    @field_validator("recency_days")
    @classmethod
    def _recency(cls, v):
        if not (0 <= v <= FEATURE_WINDOW_DAYS):
            raise ValueError(f"recency_days must be within [0, {FEATURE_WINDOW_DAYS}]; got {v}")
        return v

    @field_validator("frequency")
    @classmethod
    def _freq(cls, v):
        if v < 1:
            raise ValueError(f"frequency must be >= 1 (a customer with no orders has no RFM row); got {v}")
        return v

    @field_validator("monetary_mean")
    @classmethod
    def _mon(cls, v):
        if v <= 0:
            raise ValueError(f"monetary_mean must be > 0; got {v}")
        return v


class SegmentRequest(BaseModel):
    customers: list[Customer] = Field(..., min_length=1, max_length=1000)


class SegmentResult(BaseModel):
    customer_id: str | None
    segment_id: int
    segment_name: str
    distance_to_centroid: float


class SegmentResponse(BaseModel):
    request_id: str
    results: list[SegmentResult]
    model_version: str
    contract_version: str
    source: str


@app.exception_handler(ContractViolation)
async def _contract(request: Request, exc: ContractViolation):
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        content={"error": "contract_violation", "detail": str(exc)})


@app.exception_handler(PredictionError)
async def _predfail(request: Request, exc: PredictionError):
    log.error(f"prediction failure: {exc}")
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        content={"error": "prediction_failed", "detail": str(exc)})


@app.get("/live")
def live():
    """Liveness. Process only, no I/O. Must not depend on S3 or the artifact."""
    return {"status": "alive"}


@app.get("/health")
def health():
    """Readiness. Green only when a validated artifact is loaded."""
    if _ART is None:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            content={"status": "not_ready", "reason": _LOAD_ERROR})
    return {"status": "ready", "model_version": _ART.version,
            "contract_version": _ART.contract_version,
            "features": list(_ART.features), "k": len(_ART.segment_names)}


@app.post("/v1/segment", response_model=SegmentResponse)
def segment(req: SegmentRequest):
    if _ART is None:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            content={"error": "not_ready", "detail": _LOAD_ERROR})
    rid = str(uuid.uuid4())
    df = pd.DataFrame([c.model_dump() for c in req.customers])
    ids = df.pop("customer_id")
    out = score(df, _ART)
    log.info(f'{{"request_id":"{rid}","n":{len(df)},"model_version":"{_ART.version}"}}')
    return SegmentResponse(
        request_id=rid,
        results=[SegmentResult(customer_id=ids.iloc[i], **out.iloc[i].to_dict())
                 for i in range(len(out))],
        model_version=_ART.version, contract_version=_ART.contract_version,
        source="model",
    )


# NO /admin/reload ENDPOINT.
#
# An earlier draft exposed one. The Service publishes port 80 -> 8000 through an
# internet-facing LoadBalancer, so an unauthenticated admin route would let anyone on the
# internet force a model reload. It was removed rather than wrapped in a token: the
# artifact loads at STARTUP, so `kubectl rollout restart` is already the correct and
# audited way to pick up a new alias, and it exercises the same code path the deploy does.
#
# The rollback drill (runbook step 19) uses rollout restart, which is a STRONGER
# demonstration: with replicas: 1 and maxUnavailable: 0 a new pod that fails readiness
# never receives traffic, and the old pod keeps serving throughout.
