# Phase 1 runbook — QCOMM Customer Analytics

Real values baked in. No `<PLACEHOLDER>` anywhere except the two credential values you
type yourself. Every step has a verify command — you should never be left wondering
whether something worked.

```
Bucket    qcomm-rfm
Region    ap-south-2  (Asia Pacific, Hyderabad)
Env       dev
Account   117211782845
Runtime   Google Colab
Repo      github.com/BishalRanjanBadu/qcomm-customer-analytics   (not cloned yet — deferred)
Project   C:\Users\HP\Desktop\Bishal_\END TO END PROJECTS\ML PROJECTS\QCOMM_CUSTOMER_ANALYTICS
```

**That path contains spaces** in `END TO END PROJECTS` and `ML PROJECTS`. Every reference
to it in a shell must be quoted. Unquoted, `cd` fails at `END`.

Phase 1 runs entirely in Colab against S3. Your local machine is not used except to hold
the downloaded notebooks.

---

## Step 0 — Extract and verify this archive

Extract to a folder you can find. Then check you got everything:

```
QCOMM_PHASE1_V1_2026-09-09
```

That marker string is inside `MANIFEST.md`. Browsers cache archives by filename and
sometimes serve a stale one — check the marker, not the filename.

Expected: 7 `.ipynb` files, 7 matching `.py` files, `MANIFEST.md`, this runbook.

---

## Step 1 — Confirm the raw objects are actually in S3

Do this before anything else. A key named `raw/` is not evidence of what is in it.

**Option A — CloudShell** (browser, no local AWS CLI needed). AWS console → the terminal
icon in the top bar → make sure the region reads **Hyderabad**.

**Option B — local**, if you have the AWS CLI.

```bash
export BUCKET="qcomm-rfm"
export REGION="ap-south-2"

aws s3 ls "s3://$BUCKET/dev/raw/" --region "$REGION"
```

Expected — two objects, these exact sizes:

```
34972534 qcomm_customers_rfm.csv
92390636 qcomm_orders.csv
```

Now verify the headers are intact, not just that files exist:

```bash
aws s3 cp "s3://$BUCKET/dev/raw/qcomm_customers_rfm.csv" - --region "$REGION" | head -1
```

Must print exactly:

```
customer_id,recency_days,frequency,monetary_mean,monetary_sum,tenure_days,promo_rate,mobile_rate,segment,segment_label,orders_next_45d,dormant_45d,in_orders_sample
```

```bash
aws s3 cp "s3://$BUCKET/dev/raw/qcomm_orders.csv" - --region "$REGION" | head -1 | tr ',' '\n' | wc -l
```

Must print `26`.

**If any of these disagree, stop.** Re-upload before running a single notebook. Notebook
01 will fail its assertions anyway, but failing here is cheaper to diagnose.

---

## Step 2 — Bucket settings

```bash
aws s3api get-bucket-versioning     --bucket "$BUCKET" --region "$REGION"
aws s3api get-public-access-block   --bucket "$BUCKET" --region "$REGION"
aws s3api get-bucket-encryption     --bucket "$BUCKET" --region "$REGION"
```

You want `Status: Enabled`, all four public-access flags `true`, and an encryption rule.

If versioning is off:

```bash
aws s3api put-bucket-versioning --bucket "$BUCKET" --region "$REGION" \
  --versioning-configuration Status=Enabled
```

Versioning is your only recovery path if a notebook overwrites an artifact. Turn it on
before Phase 1 writes anything, not after.

---

## Step 3 — Billing alarm

You are using root credentials in Colab. Root keys cannot be scoped by any policy, so a
budget alarm is the only early warning that exists.

AWS console → **Billing and Cost Management** → **Budgets** → **Create budget**

```
Type    Cost budget — Monthly
Amount  ₹500
Alerts  50%, 80%, 100%  →  your email
```

Set this in **us-east-1** — billing metrics are global but only published there. Confirm
the subscription email that arrives.

Phase 1's own cost is negligible: ~130 MB of S3 storage, roughly ₹5/month. The alarm is
about the key, not the workload.

---

## Step 4 — Colab Secrets

Open any Colab notebook → left sidebar → **key icon (🔑)** → **Add new secret**.

Two secrets, both with **Notebook access** toggled ON:

| Name | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | your key id |
| `AWS_SECRET_ACCESS_KEY` | your secret |

**Type them into the Secrets panel. Never into a code cell.** A key in a cell persists in
the `.ipynb` file, in Colab's autosave revision history, and in every commit afterwards.
Deleting the cell does not remove it from the revision history — at that point the key
has to be rotated, not deleted.

The notebooks read these via `google.colab.userdata` and set them as environment
variables, so boto3's default credential chain picks them up. Nothing is hardcoded.

---

## Step 5 — Upload the notebooks

Colab → **File → Upload notebook** → pick `01_Data_Loading_and_First_Look.ipynb`.

Repeat for all seven, or drop them into Google Drive and open from there. Drive is easier
if you will re-run over several sessions.

Verify: run only the S3 setup cell of notebook 01. It should print:

```
boto3 -> s3://qcomm-rfm/dev/  region=ap-south-2  colab=True
```

If `colab=False`, you are not in Colab. If it raises a credential error, the secret names
are wrong or notebook access is off — recheck Step 4.

---

## Step 6 — Run the notebooks in order

**Runtime → Restart and run all**, one notebook at a time, in order. Each reads its input
stage from S3 and writes its output stage. Do not skip one: `02` cannot run before `01`
has written `data/customers/01_loaded.parquet`.

| # | Notebook | ~time | Writes |
|---|---|---|---|
| 01 | Data Loading and First Look | 3–5 min | `contracts/schema_v1.json`, `data/{customers,orders}/01_loaded.parquet` |
| 02 | Data Cleaning | 2–3 min | `data/customers/02_cleaned.parquet`, `data/orders/02_cleaned.parquet` |
| 03 | Missing Values and Outliers | 2–3 min | `data/orders/03_imputed.parquet`, `data/customers/03_imputed.parquet`, `contracts/imputation_params_v1.json` |
| 04 | Statistics and EDA | 1–2 min | `data/customers/04_eda.parquet` |
| 05 | Hypothesis Testing | 1–2 min | `data/customers/05_hypothesis.parquet` |
| 06 | Scaling and Feature Pipeline | 4–6 min | `data/customers/06_scaled.parquet`, snapshots, `reference/drift_reference.parquet` |
| 07 | Segmentation, Gates and Artifacts | 4–6 min | `models/v_*/`, `models/CURRENT.json` |

Notebook 01 is the slowest reader — 92 MB from Hyderabad to a Colab VM in the US. Two to
three minutes for that single cell is normal, not a hang.

### What each notebook must print before you move on

**01** — five checkpoints:
```
D1 OK  max tenure=45.00  max recency=45.00  -> window is 45d, NOT 90d
D2 OK  orders customers=43,111  flagged=35,189  orphans=7,922
D3 OK  dormant_45d == (orders_next_45d == 0) on all rows
structural-null check OK
dormancy by split (must be near-identical): test 0.2875 / train 0.2875 / val 0.2875
```

**02** — `violations normalised: 497 rows`, `first violation on day 54`, `clock-skew rows: 661`

**03** — `top/bottom ratio: 7.2x -> MNAR confirmed`, then `chain B: NaN-free`

**06** — the parity block. This is the important one:
```
recency_days     exact match = 1.0000
frequency        exact match = 0.9937
tenure_days      exact match = 1.0000
dormant_45d      exact match = 1.0000
```
and the reference-window comparison showing `orders_30d` PSI dropping from ~0.2585
(days 0–44) to ~0.0013 (days 30–44).

**07** — the gate, and the negative control that proves it can fail:
```
ARI vs the vendor reference segmentation = 0.9903
null p95=1.0214   null was MONOTONE in 10/10 runs
GATE A: PASS      spread=7.58x   monotone=True   shares 22.5%-28.9%
random partition: spread=1.021x  ->  correctly FAILS
PROMOTED -> models/v_<timestamp>_<sha>
```

Small deviations in the last decimal are expected — `n_jobs=-1` with threaded estimators
is not bit-reproducible, and this is recorded in the manifest rather than claimed away.
A gate flipping from PASS to FAIL is **not** a rounding difference. Stop and tell me.

**Scope note.** This project builds **one model** — the segmentation. There is no
supervised classifier. `dormant_45d` appears throughout as the *validation variable*: it
is measured on the 45 days after the snapshot, which the clustering never sees, and it is
in `FORBIDDEN` so it can never become an input. It is the referee, not a target.

### If a cell fails

Read the actual error before acting. Do not blind-retry.

| Symptom | First thing to check |
|---|---|
| `NoSuchKey` | `aws s3 ls s3://qcomm-rfm/dev/data/` — did the previous notebook write? |
| `AccessDenied` | Step 4 secrets, and that the key is active in IAM |
| `InvalidAccessKeyId` | the secret value has a stray space or newline — re-paste it |
| `AssertionError` in 01 | the raw object does not match. Go back to Step 1 |
| Colab disconnects mid-run | expected on long idle. Re-run that notebook from the top — every stage is idempotent |
| `MemoryError` in 01 or 06 | Runtime → Change runtime type → High-RAM |

Every stage is idempotent. Re-running a notebook overwrites its own outputs and nothing
else, so a restart is always safe.

---

## Step 7 — Verify the S3 tree

```bash
aws s3 ls "s3://$BUCKET/dev/" --recursive --region "$REGION" \
  | grep -v ' 0 ' | awk '{print $3, $4}'
```

The `grep -v ' 0 '` drops the zero-byte folder markers the console created.

Expected, roughly:

```
contracts/schema_v1.json
contracts/imputation_params_v1.json
contracts/feature_spec_v1.json
data/customers/01_loaded.parquet ... 07_segmented.parquet
data/orders/01_loaded.parquet ... 06_snapshot_d30.parquet
models/CURRENT.json
models/v_<utc>_<sha>/kmeans.pkl, feature_names.pkl, manifest.json, metrics.json
raw/qcomm_customers_rfm.csv, raw/qcomm_orders.csv
reference/drift_reference.parquet
```

Confirm the promotion pointer:

```bash
aws s3 cp "s3://$BUCKET/dev/models/CURRENT.json" - --region "$REGION"
```

---

## Step 8 — Record the handoff

The last cell of notebook 07 prints a handoff block. **Copy it verbatim** — those are the
as-run numbers from the environment that produced the artifact.

The library versions in that block are not decoration. They become the train/serve parity
contract: the Phase-2 Docker image is pinned to **those** versions, not to whatever is
latest on PyPI. A cross-version unpickle can appear to succeed while silently corrupting
fitted state and returning plausible-but-wrong predictions.

Paste it back to me and Phase 1 is signed off.

---

## What is NOT in Phase 1

Nothing here creates a chargeable resource beyond S3 storage. Specifically not built:

- No Docker image, no ECR repository
- No EKS cluster — the control plane is **~$0.10/hour, ~$73/month, never free tier**,
  regardless of node size. That is a Phase-2 decision with a cost conversation attached.
- No serving API, no `src/`, no CI/CD
- No drift job, no retraining pipeline, no SageMaker
- No git commit — deferred at your request. The notebooks are the Phase-1 documentation
  and belong in the repo eventually; `.gitignore` must exclude `1.RAW/` and `*.csv`
  **before** the first `git add`, or 127 MB enters the history permanently.

## Teardown

If you abandon the project, this removes everything and stops all billing:

```bash
aws s3 rm "s3://$BUCKET" --recursive --region "$REGION"
aws s3api delete-bucket --bucket "$BUCKET" --region "$REGION"
```

Then rotate the root access key in IAM regardless of whether you continue.
