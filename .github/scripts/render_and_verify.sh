#!/usr/bin/env bash
# Render manifests to files, VERIFY, then apply.
#
# `envsubst | kubectl apply` hides an unset variable: `image: registry/repo:` is valid
# YAML and fails minutes later as ImagePullBackOff, which looks nothing like a config bug.
# Rendering to disk lets us grep and parse before anything reaches the cluster.
set -euo pipefail

: "${IMAGE:?IMAGE must be set (full ECR reference including an immutable SHA tag)}"
SRC_DIR="${1:-k8s}"
OUT_DIR="${2:-rendered}"

PY="$(command -v python3 || command -v python)"   # Git Bash has no python3
mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/*.yml

echo "rendering with IMAGE=${IMAGE}"
for f in "$SRC_DIR"/*.yml; do
  out="$OUT_DIR/$(basename "$f")"
  IMAGE="$IMAGE" envsubst '${IMAGE}' < "$f" > "$out"
done

fail=0

# 1. Unresolved variables — on NON-COMMENT lines only. A literal ${...} inside a comment
#    is documentation, and matching it produces a false UNRESOLVED.
if grep -nE '^[^#]*\$\{[A-Z_]+\}' "$OUT_DIR"/*.yml; then
  echo "FAIL: unresolved variable above" >&2; fail=1
fi

# 2. Every image: line must be non-empty, fully qualified, and NOT :latest.
#    Inline comments are stripped FIRST. Without that, a line documented as
#    "# ... never :latest" matches its own warning text and fails the success path.
found_image=0
while IFS= read -r line; do
  ref="$(echo "$line" | sed 's/#.*//' | sed 's/.*image:[[:space:]]*//' | tr -d '[:space:]')"
  [ -z "$ref" ] && { echo "FAIL: empty image reference in a rendered manifest" >&2; fail=1; continue; }
  found_image=1
  case "$ref" in
    *:) echo "FAIL: image has an EMPTY TAG: '$ref' — this is valid YAML and becomes ImagePullBackOff minutes later" >&2; fail=1 ;;
    *:latest) echo "FAIL: refusing to deploy :latest — deploy by immutable SHA" >&2; fail=1 ;;
    *.dkr.ecr.*.amazonaws.com/*:*) echo "  image OK: $ref" ;;
    *) echo "FAIL: not a fully-qualified ECR reference: $ref" >&2; fail=1 ;;
  esac
done < <(grep -h '^[^#]*[[:space:]]image:' "$OUT_DIR"/*.yml || true)
if [ "$found_image" -eq 0 ]; then
  echo "FAIL: no image: line found in any rendered manifest" >&2; fail=1
fi

# 3. YAML must parse.
# 3. YAML must parse. `[ $? -ne 0 ] && fail=1` would abort the script under `set -e`
#    whenever the test is false, so the result is captured explicitly instead.
if ! "$PY" - "$OUT_DIR" <<'PYEOF'
import glob, sys, yaml
bad = 0
for f in sorted(glob.glob(f"{sys.argv[1]}/*.yml")):
    try:
        docs = [d for d in yaml.safe_load_all(open(f)) if d]
        print(f"  parsed {f}: {[d['kind'] for d in docs]}")
    except Exception as e:
        print(f"FAIL: {f} does not parse: {e}", file=sys.stderr); bad = 1
sys.exit(bad)
PYEOF
then
  fail=1
fi

if [ "$fail" -ne 0 ]; then
  echo "RENDER VERIFICATION FAILED — nothing applied" >&2
  exit 1
fi
echo "render verification PASSED"
