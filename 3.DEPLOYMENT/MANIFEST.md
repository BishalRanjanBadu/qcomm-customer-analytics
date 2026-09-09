# qcomm-phase2-v3 — manifest

**Build marker:** `QCOMM_PHASE2_V3_2026-09-09`

Browsers cache archives by filename and sometimes serve a stale one. Check the marker.

## Where this goes

Extract into `QCOMM_CUSTOMER_ANALYTICS/3.DEPLOYMENT/`, then **move `.github/` and
`.gitignore` up to the repo root** (runbook step 2). GitHub reads workflows only from the
root; left inside the subfolder, CI never runs — no error, no red X, just nothing.
`.dockerignore` stays put, because Docker reads it from the build context.

## Tree

```
qcomm-phase2-v3/
├── MANIFEST.md              this file
├── PHASE2_RUNBOOK.md        START HERE — 21 steps, real paths baked in
├── README.md                architecture + design decisions
├── Dockerfile               multi-stage; self-check runs in the FINAL stage
├── entrypoint.sh            exit 78 on an unrecognised ROLE
├── requirements.txt         pinned to the PROMOTED ARTIFACT, not to latest
├── requirements-dev.txt     what CI installs; -r includes requirements.txt
├── .gitignore  .dockerignore  .env.example
├── src/         api.py  predict.py  preprocess.py  registry.py  s3_io.py
├── tests/       test_api.py  test_contract.py  test_parity.py  test_registry.py
│                conftest.py  _make_fixture.py
│                fixtures/   sample_customers.csv  fixture_model.pkl  fixture_manifest.json
│                golden/     payloads.json  record.py  check_live.py  README.md
├── k8s/         config.yml  deployment.yml  service.yml  pdb.yml
│                canary/     deployment-canary.yml  README.md   (DEFERRED)
└── .github/     workflows/mlops_pipeline.yml   scripts/render_and_verify.sh
```

Verify after extraction — dotfiles are the ones that silently go missing:

```bash
ls -a                          # must show .github .gitignore .dockerignore
grep -c QCOMM_PHASE2_V3 MANIFEST.md    # -> 2
```

## Verified by execution before shipping

| check | result |
|---|---|
| Test suite, credentials unset | **29 passed** |
| `render_and_verify.sh` success path | PASSED |
| `render_and_verify.sh` empty tag | correctly rejected |
| `render_and_verify.sh` `:latest` | correctly rejected |
| `render_and_verify.sh` unset `IMAGE` | correctly rejected |
| Live API + golden record → check round-trip | PASSED, 6 accept + 5 reject |
| `POST /admin/reload` after removal | **HTTP 404** |
| CI push tags (must be SHA only, no `:latest`) | `${{ steps.meta.outputs.image }}` |
| `.gitignore` with real decoys (35 MB CSV, `.env`, `.parquet`) | all ignored; fixture staged |
| All k8s YAML parses | 5 manifests |
| Workflow YAML parses | 4 jobs |

## Upstream verification, one pass, 2026-09-09

PyPI JSON API — every pin exists with `cp313` manylinux wheels for x86_64 **and** aarch64:
`numpy 2.1.3`, `pandas 2.2.3`, `scikit-learn 1.6.1`, `scipy 1.16.3` (from the promoted
manifest), plus `fastapi 0.141.1`, `uvicorn 0.52.4`, `pydantic 2.13.5`, `boto3 1.43.90`,
`pytest 9.1.1`.

`git ls-remote --tags` — action pins: `checkout@v7`, `setup-python@v7`,
`configure-aws-credentials@v6`, `amazon-ecr-login@v2`, `build-push-action@v7`,
`setup-buildx-action@v4`, `trivy-action@v0.36.0`.

**`aquasecurity/trivy-action` publishes NO floating major.** `@v0` and `@0.36.0` both fail
at "Set up job", before any step runs. Only the full v-prefixed tag works.

## Bugs found and fixed

### Caught by the tests during the build (v1)

| bug | effect |
|---|---|
| `JSONResponse(status.HTTP_503, {...})` — positional args are `(content, status_code)` | **every error path returned HTTP 200 with the status code as the body.** 5 sites. |
| Fixture manifest carried the build machine's library versions | could never satisfy the production manifest check |
| `render_and_verify.sh` matched its own inline comment "never `:latest`" | the success path failed |

### Caught by the line-by-line audit (v1 -> v2)

| # | defect | why it mattered |
|---|---|---|
| 1 | **`/admin/reload` reachable through the internet-facing LoadBalancer with no auth** | port 80 -> 8000 is public. Anyone could force a model reload. **Endpoint removed**; the rollback drill now uses `kubectl rollout restart`, which is the correct path anyway because the artifact loads at startup. |
| 2 | **ECR `--image-tag-mutability IMMUTABLE` + CI pushing `:latest` every run** | the **second** CI run and every run after would fail with `ImageTagAlreadyExistsException`. Immutability and a floating tag are mutually exclusive. `:latest` removed from CI. |
| 3 | **`joblib` and `threadpoolctl` unpinned** | scikit-learn pulls them with `>=`, so two builds a week apart are not identical — which defeats the four pins that exist to create the parity contract. Now `joblib==1.6.0`, `threadpoolctl==3.6.0`, both verified pure-python on PyPI. |
| 4 | **6 stale runbook step cross-references** | `k8s/canary/README.md` said step 17, the workflow said steps 14/15/16, `check_live.py` said step 16, `tests/golden/README.md` said steps 12/14/16. All pointed at the wrong steps after renumbering. All corrected. |
| 5 | **`$SHA` lost across terminal sessions** | set in step 12, used in step 17, with a ~20-minute cluster create between them. Step 17 now re-derives it and verifies the tag exists in ECR. |
| 6 | **`kill %1` job control** | fragile and can target the wrong job. Replaced with an explicit `$API_PID`. |

The rollback drill was rewritten around `rollout restart`, which is a **stronger**
demonstration than a reload endpoint: with `replicas: 1` and `maxUnavailable: 0`, a new pod
that fails readiness never receives traffic and the old pod keeps serving throughout. The
drill now proves that property directly by running the golden check against the live
endpoint *while* the bad pod is failing.

## Known gaps — deliberate

- **`tests/golden/expected.json` is absent.** It must be recorded against the real promoted
  artifact (runbook step 8), never a fixture-trained stand-in. The CI `golden` job skips
  with a message while it is absent and enforces once committed.
- **The container was never built here** — no Docker in this environment. The Dockerfile's
  self-check runs at your first `docker build` (runbook step 9).
- **Nothing was applied to a cluster.** Manifests are rendered and parsed, not applied.
- **No Phase 3 anything** — no drift job, no retraining, no alarms, no prediction logging.

Build marker: `QCOMM_PHASE2_V3_2026-09-09`
