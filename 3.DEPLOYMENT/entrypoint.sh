#!/bin/sh
# Fail fast with exit 78 (EX_CONFIG) on an unrecognised role, naming the expected values.
# Crashing on "command not found" tells the operator nothing.
set -eu

ROLE="${ROLE:-api}"
PORT="${PORT:-8000}"

case "$ROLE" in
  api)
    echo "{\"msg\":\"starting api on 0.0.0.0:${PORT}\"}"
    exec uvicorn src.api:app --host 0.0.0.0 --port "${PORT}" --workers 1
    ;;
  *)
    echo "FATAL: ROLE='${ROLE}' is not recognised. Expected: api" >&2
    exit 78
    ;;
esac
