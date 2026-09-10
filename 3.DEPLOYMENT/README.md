# 🛒 Q-Commerce Customer Analytics — Production MLOps

**A leakage-audited RFM customer segmentation system for quick-commerce, using KMeans on log-scaled Recency / Frequency / Monetary behaviour, validated against a held-out 45-day dormancy outcome and served from EKS with an S3-backed artifact registry, contract validation, CI/CD, and train/serve parity.**

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Model](https://img.shields.io/badge/Model-KMeans-orange)
![Features](https://img.shields.io/badge/features-3-informational)
![Segments](https://img.shields.io/badge/segments-4-success)
![Gate](https://img.shields.io/badge/Gate-7.58×%20spread-success)
![ARI](https://img.shields.io/badge/ARI-0.9903-success)
![Tests](https://img.shields.io/badge/tests-29%20passing-success)
![Serving](https://img.shields.io/badge/Serving-FastAPI-009688)
![Infra](https://img.shields.io/badge/Infra-AWS%20EKS-orange?logo=amazonaws)
![CI](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=githubactions)

---

## What this repository actually is

Most customer-segmentation projects stop after calculating RFM scores and plotting four
clusters. This repository carries the segmentation through to a production-style ML system:
from snapshot reconstruction and leakage controls, through statistical gating and artifact
versioning, to a containerised FastAPI service deployed on EKS.

The important engineering is in the boundary between analytics and production — making sure
the segmentation is reproducible, the validation outcome cannot leak into the model, and the
artifact served in production is the artifact that was actually promoted.

| Phase | Scope | Status |
|---|---|---|
| **1 — Customer Analytics** | Data reconstruction, RFM engineering, EDA, hypothesis testing, leakage controls, KMeans segmentation, validation gates | **complete — `QCOMM_PHASE1_V2_2026-09-09`** |
| **2 — Deployment** | Production preprocessing, artifact registry, FastAPI, tests, Docker, EKS manifests, CI/CD | **complete — `QCOMM_PHASE2_V3_2026-09-09`** |
| **3 — Operate & Monitor** | Prediction logging, drift detection, automated retraining, automated promotion | **not started** |

Each phase contains a manifest/runbook documenting the assumptions, verification results and
known deviations rather than treating the README as the only source of truth.

---

## ✅ What was independently verified, not asserted

The strongest claims in this repository are backed by executable checks or recorded build
verification.

**The clustering has an external referee.** `dormant_45d` is measured during the 45 days after
the snapshot. The clustering never sees that period, and the variable is explicitly placed in
`FORBIDDEN`, so it cannot become a feature. This prevents a segmentation from being validated
against the same outcome used to construct it.

**The feature window is explicitly bounded.** The production feature window is **45 days**,
not the 90 days suggested by the vendor dictionary. The rebuilt pipeline verifies
`max(tenure)=45.00` and `max(recency)=45.00`.

**The split is fixed once.** The Phase-1 pipeline stamps a 70/15/15 train/validation/test
split and carries `__split` through the stages. Model fitting is performed on train rows;
`k` selection uses validation; the final gate is evaluated on the held-out test set.

**The gate was tested in both directions.** The real test segmentation achieved an outcome
spread of **7.5838×**, while a random-partition negative control produced **1.0211×** and
correctly failed the 3.0× floor. A 10-run permutation null produced a p95 of **1.0214×**.

**The segmentation reproduces the reference solution.** The rebuilt four-segment solution
achieved **ARI = 0.9903** against the vendor reference segmentation.

**Production code has a 29-test suite with no cloud credentials.** Contract, API, registry and
parity tests use committed fixtures. The production fixture itself is generated through the
production code path, reducing the risk of certifying a test-only preprocessing implementation.

**Deployment uses immutable image identity.** The CI/CD path refuses an empty image tag and
refuses `:latest`; deployment is intended to use a concrete SHA-tagged image.

---

## Overview

Segments quick-commerce customers into **four behaviour-driven groups** using KMeans over
log-scaled RFM features, then profiles and serves those segments through a production API.

| | |
|---|---|
| Problem type | Unsupervised customer segmentation |
| Algorithm | **KMeans** |
| Input space | Log-scaled Recency / Frequency / mean Monetary |
| Number of segments | **4** |
| Snapshot | **2026-02-15 UTC** |
| Feature window | **45 days** |
| Split | 70/15/15 stratified train / validation / test |
| Seed | **42** |
| External validation | `dormant_45d` over the following 45 days |
| Serving | FastAPI + Docker + AWS EKS |
| Artifact store | Amazon S3 |

### Why RFM and not a supervised classifier?

The objective is to discover actionable customer groups rather than predict a labelled class.
The model therefore uses only three behavioural axes:

- **Recency** — how recently the customer ordered
- **Frequency** — how often the customer ordered
- **Monetary** — average order value

`dormant_45d` is deliberately **not** a modelling target. It is an outcome observed after the
snapshot and acts as an independent business referee for whether the discovered segments are
meaningfully separated.

---

## Business problem

Quick-commerce teams need a compact way to distinguish high-value loyal customers from
customers at risk of becoming inactive and customers who are already dormant. A useful
segmentation should support different retention, engagement and lifecycle actions without
building the segments from the outcome itself.

The engineering challenge is equally important: customer features must represent only the
information available at the snapshot, while the validation window must remain completely
outside the modelling period.

This project therefore treats **temporal boundaries, leakage prevention, reproducibility and
serving parity as first-class requirements**, rather than as documentation after the model is
built.

---

## Dataset & snapshot design

The Phase-1 pipeline reconstructs customer-level behaviour from order data and creates a
snapshot table for segmentation.

| Property | Value |
|---|---|
| Snapshot date | **15-Feb-2026** |
| Source period | **01-Jan-2026 → 31-Mar-2026** |
| Customer feature window | **45 days before snapshot** |
| Validation window | **45 days after snapshot** |
| Customers in rebuilt dataset | **439,492** |
| Train / validation / test | **70% / 15% / 15%** |
| Random seed | **42** |

The feature chain also records important data-quality findings, including **497 enum-drift
rows**, **661 clock-skew rows**, and an MNAR pattern in `order_value` where blank rates in the
top/bottom deciles differed by roughly **7.2×**.

### Geography is intentionally excluded

`city_id`, `delivery_city` and `billing_city` are excluded from the rebuilt customer feature
table. The design treats geography as a potential proxy for protected characteristics rather
than allowing it to drive the customer segmentation.

---

## Methodology

| # | Stage | Does |
|---|---|---|
| 01 | Data Loading & First Look | Reconstruct customer/order views, validate snapshot boundaries, inspect schema and stamp the split |
| 02 | Data Cleaning | Standardise fields, handle invalid values and preserve the modelling boundary |
| 03 | Missing Values & Outliers | Impute and cap where appropriate without deleting the high-value customers the project is intended to identify |
| 04 | Statistics & EDA | Analyse RFM distributions, customer behaviour and segmentability |
| 05 | Hypothesis Testing | Test behavioural differences and document statistical evidence |
| 06 | Scaling & Feature Pipeline | Build the deterministic RFM transformation and leakage checks |
| 07 | Segmentation, Gates & Artifacts | Select `k`, fit KMeans, evaluate the external outcome, name segments and write versioned artifacts |

The Phase-1 pipeline is available in both notebook and `.py` form. The `.py` files are
jupytext sources so the logic remains diff-friendly in Git.

---

## Feature engineering

The production segmentation uses **three and only three modelling features**:

| Feature | Meaning | Transformation |
|---|---|---|
| `recency_days` | Days since most recent order | log-scaled before clustering |
| `frequency` | Number of orders in the feature window | log-scaled before clustering |
| `monetary_mean` | Mean order value | log-scaled before clustering |

### Why `monetary_mean`, not `monetary_sum`

`monetary_sum = monetary_mean × frequency`. Including both would count frequency twice in the
KMeans distance calculation. `monetary_sum` is therefore a **profile-only** field used to
describe segments after fitting; it does not define them.

### What is explicitly forbidden

The following columns are never allowed into the clustering feature matrix:

```text
segment
segment_label
orders_next_45d
customer_id
in_orders_sample
dormant_45d
__split
```

This makes the leakage boundary explicit and testable rather than dependent on notebook
inspection.

---

## Choosing the number of segments

`k` is evaluated on the **validation set**, never selected using the held-out test outcome.
The sweep considers `k=2…8` and evaluates outcome separation, segment-size constraints and
silhouette as supporting information.

**k=4 is a business decision, not a claim that four is the mathematical optimum.** Outcome
separation continues to improve for larger partitions, but four groups are considered a
practical number for distinct operational campaigns and also reproduce the reference
segmentation with very high agreement.

This distinction matters: a clustering project should not silently optimise for a metric that
makes the eventual business action impossible to operate.

---

## Segment definitions

Cluster IDs are arbitrary KMeans labels, so segment names are **derived from measured
behaviour**, not hardcoded to a cluster number.

The four behavioural segments are:

| Segment | Business interpretation |
|---|---|
| 🏆 **Champions** | Highest-engagement customers with the strongest retention value |
| 🔁 **Steady Regulars** | Consistent customers with stable repeat behaviour |
| ⚠️ **High-Value At-Risk** | Valuable customers showing elevated dormancy risk |
| 💤 **Lost / Dormant** | Customers with the weakest recent engagement and highest dormancy |

The ordering is generated from the held-out behavioural outcome, while the outcome itself is
never fed into KMeans.

---

## Validation & quality gate

The primary production gate is **outcome separation** — the ratio between the highest and
lowest segment dormancy rates on the held-out test set.

| Gate | Threshold | Actual | Status |
|---|---:|---:|---|
| Outcome spread | ≥ **3.0×** | **7.5838×** | ✅ PASS |
| Outcome separation vs permutation null | > **1.0214× p95** | **7.5838×** | ✅ PASS |
| Segment size | 2%–60% | within bounds | ✅ PASS |
| Vendor/reference agreement | ARI > 0.90 | **0.9903** | ✅ PASS |
| Negative control | random partition must fail | **1.0211×** | ✅ PASS |

The **3.0× threshold is provisional**. Its statistical derivation clears the permutation null
by a wide margin, but it is not a cost-optimised business threshold. A future retention-cost
model should revisit it before automated retraining and promotion.

### Why silhouette is not the gate

Silhouette measures geometric compactness, not whether the resulting segments are useful for
retention decisions. It is reported as supporting evidence, but it is deliberately not the
promotion gate.

Similarly, bootstrap ARI and ANOVA-style feature separation were rejected as gates because
they can remain strong on shuffled/no-signal data. The repository explicitly demonstrates
this failure mode rather than assuming conventional clustering metrics prove business value.

---

## Key findings

- **The segmentation separates future dormancy strongly:** the held-out outcome spread is
  **7.58×** from the least to most dormant segment.
- **The gate is not merely tuned to pass:** a random partition produces only **1.021×** spread
  and correctly fails the 3.0× threshold.
- **The rebuilt pipeline is highly consistent with the reference segmentation:** ARI is
  **0.9903**.
- **Customer identity is not used as a feature**, and geography is deliberately excluded from
  the modelling table.
- **The feature window is 45 days**, despite a vendor dictionary claiming 90 days; this was
  verified directly from the rebuilt customer table.
- **Outliers are capped rather than dropped**, because high-spend customers are one of the
  populations the segmentation is specifically intended to identify.
- **Permutation testing exposes a common clustering trap:** monotonic dormancy ordering can
  appear even when R/F/M structure is destroyed. The magnitude of separation, not monotonicity
  alone, carries the evidence.

---

## Production architecture

```text
                 ┌──────────────────────────────┐
                 │        Order Data / S3       │
                 └──────────────┬───────────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │ Phase 1: Snapshot + RFM      │
                 │ 45-day feature window        │
                 └──────────────┬───────────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │ KMeans on log-scaled R/F/M   │
                 │ k = 4                        │
                 └──────────────┬───────────────┘
                                │
                    quality gate + artifact
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │       S3 Artifact Registry   │
                 │ versioned model + manifest   │
                 └──────────────┬───────────────┘
                                │
                         startup validation
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │ FastAPI container             │
                 │ Docker → ECR → EKS           │
                 └──────────────┬───────────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │ Customer segment API          │
                 │ Champions / Regulars /       │
                 │ At-Risk / Dormant            │
                 └──────────────────────────────┘
```

### Two automation paths

**Code path**

```text
git push
  → GitHub Actions
  → tests
  → Docker build
  → ECR
  → EKS deployment
  → post-deploy golden check
```

**Data path — Phase 3, not yet built**

```text
new data
  → EventBridge
  → Lambda
  → drift detection
  → retraining
  → validation gate
  → artifact promotion
  → pod reload
```

CI deliberately does **not** bootstrap the EKS cluster, OIDC provider or IRSA infrastructure.
Those are one-time infrastructure steps that the deployment process verifies.

---

## Serving API

The Phase-2 FastAPI service separates liveness from readiness and validates the promoted
artifact before reporting the service ready.

| Route | Purpose |
|---|---|
| `GET /live` | Process liveness; no external I/O |
| `GET /health` | Readiness; artifact + manifest validation |
| `GET /v1/model` | Promoted model/artifact metadata |
| `GET /v1/schema` | Request contract |
| `POST /v1/predict` | Generate customer segment |

The API intentionally does not provide silent defaults for required Pydantic fields. Missing
inputs return **422** rather than generating a plausible-looking segment from fabricated
values.

A failed artifact reload clears the cached artifact rather than continuing to serve an old
model while reporting the new version.

---

## Artifact registry & reproducibility

The promoted artifact is:

```text
v_20260909T1228Z_nogit
```

The manifest records the feature contract, dtypes and library versions needed to reconstruct
the serving environment.

The scaler is reconstructed from manifest constants rather than unpickled. This avoids an
entire class of cross-version serialisation failures: the production transformation is
mathematically equivalent to `(x - mean) / scale`, and parity tests pin that reconstruction
to scikit-learn's transform at **1e-12** tolerance.

The current artifact carries `git_sha = "nogit"` because the Phase-1 artifact was created
before the repository clone was available. This is a documented provenance gap; re-running
Phase-1 after cloning the repository should mint an artifact with a real commit SHA.

---

## Test strategy

The deployment suite contains **29 tests** across four main areas:

| Test group | What it protects |
|---|---|
| Contract | Request schema, feature order and expected dtypes |
| API | HTTP behaviour, validation and endpoint contracts |
| Registry | Artifact resolution, manifest compatibility and failure behaviour |
| Parity | Production preprocessing/scaler reconstruction and expected predictions |

The test environment uses committed fixtures and requires **zero cloud credentials**. This
makes CI deterministic and prevents a test run from failing simply because an S3 object or
credential is unavailable.

Golden payloads are also committed for the live endpoint round-trip checks.

---

## Repository structure

```text
qcomm-customer-analytics/
│
├── .github/
│   ├── workflows/
│   │   └── mlops_pipeline.yml
│   └── scripts/
│       └── render_and_verify.sh
│
├── 2.RUNNNG_CODE_FILES/
│   ├── qcomm-phase1-v1-LATER/
│   └── qcomm-phase1-v2/
│       ├── MANIFEST.md
│       ├── PHASE1_RUNBOOK.md
│       └── notebooks/
│           ├── 01_Data_Loading_and_First_Look
│           ├── 02_Data_Cleaning
│           ├── 03_Missing_Values_and_Outliers
│           ├── 04_Statistics_and_EDA
│           ├── 05_Hypothesis_Testing
│           ├── 06_Scaling_and_Feature_Pipeline
│           └── 07_Segmentation_Gates_and_Artifacts
│
├── 3.DEPLOYMENT/
│   ├── Dockerfile
│   ├── entrypoint.sh
│   ├── requirements.txt
│   ├── MANIFEST.md
│   ├── PHASE2_RUNBOOK.md
│   ├── src/
│   │   ├── api.py
│   │   ├── predict.py
│   │   ├── preprocess.py
│   │   ├── registry.py
│   │   └── s3_io.py
│   ├── tests/
│   │   ├── test_api.py
│   │   ├── test_contract.py
│   │   ├── test_parity.py
│   │   ├── test_registry.py
│   │   └── fixtures/
│   └── k8s/
│       ├── config.yml
│       ├── deployment.yml
│       ├── service.yml
│       ├── pdb.yml
│       └── canary/
│
├── CURRENT.backup.json
├── gha-trust.json
└── irsa-policy.json
```

---

## Technology stack

**Data & ML**

- Python 3.13
- Pandas
- NumPy
- SciPy
- scikit-learn 1.6.1
- KMeans

**Production serving**

- FastAPI 0.141.1
- Uvicorn 0.52.4
- Pydantic 2.13.5
- Boto3 1.43.90

**Infrastructure & MLOps**

- Docker
- Amazon ECR
- Amazon EKS
- Amazon S3
- IAM / IRSA
- Kubernetes
- GitHub Actions

**Testing**

- Pytest 9.1.1
- Contract tests
- Golden-record tests
- Train/serve parity tests
- Negative controls

---

## Known gaps & deliberate deviations

| Gap | Current status |
|---|---|
| Phase-3 drift monitoring | Not implemented |
| Automated retraining | Not implemented |
| Prediction logging | Not implemented |
| Automated model promotion | Not implemented |
| Canary deployment | Deferred; requires additional pod capacity |
| Bake monitoring | Deferred until prediction logging exists |
| Real Git SHA in promoted artifact | Pending Phase-1 re-run after repository clone |
| Golden expected record from real promoted artifact | Pending runbook step 8 |
| Docker build verification | Dockerfile self-check is prepared; container build was not executed in the build environment |
| EKS live deployment verification | Kubernetes manifests were rendered/parsed; live cluster application is an environment step |
| Quality threshold | **Provisional** until a business retention-cost model exists |

These are documented as gaps rather than presented as completed capabilities.

---

## Cost awareness

The deployment design uses AWS EKS with an EKS control plane plus compute and a LoadBalancer.
The repository runbook estimates approximately **₹10,800/month** for the documented small
cluster configuration. This is infrastructure cost, not a free-tier assumption.

The teardown runbook is intentionally part of the project because an idle EKS cluster and
LoadBalancer can continue generating charges.

---

## Run locally

### Phase 1 — analytics

The Phase-1 notebooks are designed to run top-to-bottom in a clean Colab/local environment.
AWS credentials should be supplied through the environment's normal credential mechanism —
never committed into a notebook.

### Phase 2 — API

From `3.DEPLOYMENT/`:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

pytest tests/ -v

uvicorn src.api:app --host 0.0.0.0 --port 8000
```

For Docker/Kubernetes deployment, follow `3.DEPLOYMENT/PHASE2_RUNBOOK.md`. The deployment
manifests are designed around an immutable SHA-tagged image rather than `:latest`.

---

## Project status

**Phase 1 — Customer Analytics:** ✅ Complete  
**Phase 2 — Production Deployment:** ✅ Complete  
**Phase 3 — Monitoring & Automated Retraining:** ⏳ Planned

### Next engineering milestones

1. Re-run Phase 1 after cloning the repository to attach a real Git SHA to the promoted artifact.
2. Record the golden expected response against the real promoted artifact.
3. Complete the live Docker build and EKS deployment verification.
4. Add prediction logging and feature-distribution monitoring.
5. Define a business-costed retention objective before turning the provisional gate into an automated promotion policy.
6. Implement EventBridge → Lambda → drift → retrain → gate → promotion.

---

## Author

**Bishal Ranjan Badu**

Data Science | Machine Learning | MLOps | Customer Analytics

---
