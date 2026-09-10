# syntax=docker/dockerfile:1
# Multi-stage. Builder carries the toolchain; the final image does not.
#
# Base tag resolves to the latest 3.13.x at build time, not exactly the 3.13.15 that
# produced the pickle. Patch-level CPython drift does not affect pickle compatibility —
# that is determined by library versions and the pickle protocol, both of which the
# manifest check enforces at startup. The image DIGEST is recorded at deploy so the
# running image is identified exactly.

FROM python:3.13-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv && /opt/venv/bin/pip install -r requirements.txt

# ---------------------------------------------------------------- final
FROM python:3.13-slim AS final

# apt-get upgrade patches base-image CVEs but makes the OS layer non-reproducible: two
# builds a week apart get different package sets. Accepted deliberately; the deploy records
# the image digest so a running image is still identified exactly.
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root. HOME is declared explicitly and cross-checked against every mountPath in k8s/ —
# with readOnlyRootFilesystem an unset HOME resolving somewhere unwritable crash-loops on
# the cluster only, and is invisible locally.
RUN useradd --create-home --uid 10001 appuser
ENV HOME=/home/appuser \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AWS_REGION=ap-south-2 \
    S3_BUCKET=qcomm-rfm \
    ENV_PREFIX=dev \
    ROLE=api \
    PORT=8000

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser entrypoint.sh ./entrypoint.sh

# Verification runs in the FINAL stage — the one that ships, and the only one with
# libgomp1. An import check in the builder fails on the check itself, not on the thing
# being checked. This imports the APPLICATION, not just its dependencies, so a syntax
# error in src/ fails the build instead of surfacing at readiness.
RUN python -c "\
import numpy, pandas, sklearn, scipy, hashlib; \
from src.preprocess import FEATURES; \
from src.registry import EXPECTED_FEATURE_ORDER_HASH; \
import src.api; \
h = hashlib.sha256('|'.join(FEATURES).encode()).hexdigest()[:16]; \
print('libs', numpy.__version__, pandas.__version__, sklearn.__version__, scipy.__version__); \
print('feature_contract_hash', h); \
assert h == EXPECTED_FEATURE_ORDER_HASH, (h, EXPECTED_FEATURE_ORDER_HASH); \
print('image self-check OK')"

USER appuser
EXPOSE 8000
# No curl, no wget: probes are httpGet, executed by the kubelet. Every package in this
# image has a named consumer inside it.
ENTRYPOINT ["./entrypoint.sh"]
