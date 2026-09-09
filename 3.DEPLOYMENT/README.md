# QCOMM Customer Analytics — RFM segmentation

Production ML system: KMeans segmentation over log-scaled Recency / Frequency / mean
Monetary, served from a container on EKS, with the artifact resolved through an S3 alias.

**One model. No supervised classifier.** `dormant_45d` is the *validation variable* — it
is measured on the 45 days after the snapshot, a window the clustering never sees, and it
is in `FORBIDDEN` so it can never become an input.

## Automation model

- **Trigger 1 — code:** `git push` → CI/CD (test → build → ECR → EKS → post-deploy golden check)
- **Trigger 2 — data:** EventBridge → Lambda → drift → retrain → promote alias → pods reload *(Phase 3, not built)*

**CI cannot bootstrap infrastructure.** The EKS cluster, the OIDC provider and the IRSA
ServiceAccount are one-time human steps that the pipeline *verifies* but never creates.
Omitting this sends the first failed deploy's debugging in entirely the wrong direction.

## Repository layout

Phase 2 lives in the `3.DEPLOYMENT/` subfolder. **`.github/` and `.gitignore` sit at the
repository root** — GitHub reads workflows only from the root, and a `.gitignore` governs
only its own directory and below, so it has to be above `1.RAW/`.

```
QCOMM_CUSTOMER_ANALYTICS/        <- git repo root
├── .github/workflows/           CI, parameterised with PROJECT_DIR: 3.DEPLOYMENT
├── .gitignore                   covers 1.RAW/ and exempts the committed fixture
├── 1.RAW/                       raw CSVs — ignored; S3 is the source of truth
├── <notebooks>/                 Phase-1 notebooks — committed documentation
└── 3.DEPLOYMENT/
    ├── Dockerfile  entrypoint.sh  requirements*.txt  .dockerignore
```

```
3.DEPLOYMENT/
src/         preprocess.py  the ONE transform; align() restores trained dtypes
             registry.py    alias resolution + manifest validation at startup
             predict.py     scoring; re-raises with the cause named
             api.py         FastAPI; /live and /health split; no field defaults
             s3_io.py       default credential chain ONLY
tests/       parity, contract, registry, api  — 29 tests, zero cloud credentials
             fixtures/      committed; built BY THE PRODUCTION CODE PATH
             golden/        payloads committed; expected.json recorded in runbook step 8
k8s/         deployment, service, config, pdb;  canary/ is correct and DEFERRED
.github/     mlops_pipeline.yml (4 jobs), render_and_verify.sh
```

## Phase 1 result

| | |
|---|---|
| Artifact | `v_20260909T1228Z_nogit` |
| Gate | outcome spread **7.5838×** vs a permutation null p95 of **1.0214** |
| Negative control | random partition **1.0211×** → correctly fails |
| ARI vs vendor reference | **0.9903** |
| Threshold | 3.0×, **PROVISIONAL** — no cost model supplied |

Excluded as gates, with reasons recorded in `metrics.json`: bootstrap stability ARI
(passes at 0.960 on shuffled data), ANOVA F (circular), silhouette (0.261 real vs 0.256 on
a marginal-shuffled null).

## Design decisions that are not defaults

**The scaler is reconstructed from manifest constants, not unpickled.** StandardScaler is
`(x - mean) / scale`, so reconstructing removes an entire cross-version-unpickle failure
mode. `test_parity.py` pins the reconstruction against sklearn's own transform to 1e-12.

**The manifest is validated at startup and fails readiness on mismatch.** A bad promotion
then fails readiness, the previous pods keep serving, and nothing rolls forward. It pins
feature order, feature **dtypes**, contract version and library versions — order alone is
necessary and not sufficient.

**A failed reload clears the cached artifact.** Leaving the previous object in place means
a pod keeps serving the old model while reporting the new version — a silent provenance lie.

**No Pydantic field has a default.** A caller who omits an input gets 422, not a confident
200 computed from schema placeholders.

**The test job has no cloud credentials at all.** Data resolves to a committed fixture.
This removes the whole class of "CI failed on NoSuchKey / missing credentials".

**The fixture is built by the production code path** (`tests/_make_fixture.py`), with the
production dtypes and the production scaler constants. A fixture fitted differently from
production certifies the wrong thing.

**Deploy by SHA, never `:latest`.** `render_and_verify.sh` refuses `:latest` and refuses an
empty tag — `image: registry/repo:` is valid YAML and becomes `ImagePullBackOff` minutes
later, which looks nothing like a config error.

## Known deviations

| deviation | reason |
|---|---|
| Canary deferred | needs ≥2 pods; one `t3.medium` cannot host them. Manifests shipped in `k8s/canary/`. |
| Bake monitor deferred | needs prediction logging, which Phase 3 enables with the IRSA `s3:PutObject` scope. A gate reading a signal nothing produces fails closed on every deploy. |
| `git_sha = "nogit"` | the repo was not cloned when notebook 07 ran, so the artifact has no code provenance. Re-run notebook 07 after the clone to mint a version with a real SHA. |
| Threshold provisional | statistical derivation only; review trigger is the retention action and its per-customer cost. |
| Data is synthetic | a causal simulation, not observed behaviour. |

## Cost

EKS control plane **$0.10/hour ≈ ₹6,400/month**, per cluster, billed whether or not a pod
runs — never free tier. Plus one `t3.medium` and one LoadBalancer. **≈ ₹10,800/month
total.** Runbook step 21 tears it down; delete the LoadBalancer Service *before* the
cluster or the ELB is orphaned and keeps billing.
