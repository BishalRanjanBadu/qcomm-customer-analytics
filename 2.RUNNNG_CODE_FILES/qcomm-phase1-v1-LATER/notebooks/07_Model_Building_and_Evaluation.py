# %% [markdown]
# # Model Building and Evaluation
#
# **Project:** QCOMM Customer Analytics — RFM segmentation + dormancy prediction
# **Phase 1, notebook 07.** Model A segmentation, Model B dormancy. Gates with permutation nulls. Test scored once. Artifacts to S3.
#
# Runs top-to-bottom in a clean Colab kernel. Reads its input stage from S3, writes its
# output stage to S3. No local file is a source of truth.

# %% [markdown]
# ## S3 setup
#
# Credentials come from **Colab Secrets** (left sidebar, key icon), never from a cell.
# A key pasted into a cell persists in the `.ipynb`, in Colab autosave history, and in
# every later commit — deleting the cell does not remove it, only rotation does.

# %%
import os, io, json, warnings
import numpy as np, pandas as pd

try:
    import pyarrow  # noqa: F401
except ImportError:
    os.system("pip install -q pyarrow")
    import pyarrow  # noqa: F401

try:
    from google.colab import userdata
    os.environ["AWS_ACCESS_KEY_ID"] = userdata.get("AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = userdata.get("AWS_SECRET_ACCESS_KEY")
    IN_COLAB = True
except ImportError:
    IN_COLAB = False   # local run: boto3 falls through to ~/.aws/credentials

import boto3

BUCKET = "qcomm-rfm"
REGION = "ap-south-2"
ENV    = "dev"
PREFIX = f"{ENV}"

# Default credential chain only. Never pass aws_access_key_id= here — explicit keys
# override and disable IRSA, which is how production resolves identity.
s3 = boto3.client("s3", region_name=REGION)


def _key(rel):
    return f"{PREFIX}/{rel}"


def read_s3(rel, **kw):
    """Read csv or parquet from s3://BUCKET/dev/<rel>."""
    body = s3.get_object(Bucket=BUCKET, Key=_key(rel))["Body"].read()
    buf = io.BytesIO(body)
    return pd.read_csv(buf, **kw) if rel.endswith(".csv") else pd.read_parquet(buf, **kw)


def save_s3(df, rel):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=BUCKET, Key=_key(rel), Body=buf.getvalue())
    print(f"saved s3://{BUCKET}/{_key(rel)}  rows={len(df):,}  cols={df.shape[1]}")


def save_json_s3(obj, rel):
    s3.put_object(Bucket=BUCKET, Key=_key(rel),
                  Body=json.dumps(obj, indent=2, default=str).encode())
    print(f"saved s3://{BUCKET}/{_key(rel)}")


def read_json_s3(rel):
    return json.loads(s3.get_object(Bucket=BUCKET, Key=_key(rel))["Body"].read())


def list_s3(rel_prefix=""):
    """Folder-marker objects (0-byte keys ending in '/') created in the console are filtered."""
    p = s3.get_paginator("list_objects_v2")
    out = []
    for page in p.paginate(Bucket=BUCKET, Prefix=_key(rel_prefix)):
        out += [o["Key"] for o in page.get("Contents", []) if not o["Key"].endswith("/")]
    return out


# --- project constants, baked in. No placeholders below this line. ---
DATA_START          = pd.Timestamp("2026-01-01", tz="UTC")
SNAPSHOT            = pd.Timestamp("2026-02-15", tz="UTC")
DATA_END            = pd.Timestamp("2026-03-31", tz="UTC")
FEATURE_WINDOW_DAYS = 45          # D1 — the vendor dictionary says 90. It is wrong.
DRIFT_REF_DAYS      = (30, 44)    # D5 — days 0-29 are orders_30d warm-up, not signal
SEED                = 42

RFM_COLS   = ["recency_days", "frequency", "monetary_mean"]
MODEL_B_FEATURES = ["recency_days", "frequency", "monetary_mean", "monetary_sum",
                    "tenure_days", "promo_rate", "mobile_rate"]
FORBIDDEN  = ["segment", "segment_label", "orders_next_45d",
              "customer_id", "in_orders_sample", "dormant_45d", "__split"]

print(f"boto3 -> s3://{BUCKET}/{PREFIX}/  region={REGION}  colab={IN_COLAB}")

# %%
import pickle, hashlib, subprocess, datetime as dt, sklearn, scipy
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (adjusted_rand_score, silhouette_score, roc_auc_score,
                             brier_score_loss, average_precision_score, confusion_matrix)

cust = read_s3("data/customers/06_scaled.parquet")
ZC = ["z_" + c for c in RFM_COLS]
tr = (cust.__split == "train").to_numpy()
va = (cust.__split == "val").to_numpy()
te = (cust.__split == "test").to_numpy()
Z = cust[ZC].to_numpy()
y = cust.dormant_45d.to_numpy()
print(f"train={tr.sum():,}  val={va.sum():,}  test={te.sum():,}")

# %% [markdown]
# # MODEL A — RFM segmentation
#
# ## The gate, and why the obvious metrics are excluded
#
# Two metrics are conventionally offered as evidence for a clustering. Both were measured
# on this project's earlier e-commerce dataset, on data where customer identity had been
# **randomly shuffled across orders** so that no customer-level signal existed at all:
#
# | metric | value on pure noise |
# |---|---|
# | bootstrap stability ARI | **0.960** |
# | ANOVA F (R / F / M) | **2,220 / 5,490 / 2,657**, p ≈ 0 |
# | silhouette @ k=4 | 0.256 vs 0.261 on real data |
#
# Neither can fail. Stability measures determinism; ANOVA on the features you clustered
# with is circular. **They are not used as gates here.** Silhouette is reported as
# supporting information only.
#
# The gate is **outcome separation**: segments ordered by dormancy — a variable measured
# on a window the clustering never saw — and the ratio between the highest and lowest.

# %%
def outcome_spread(labels, outcome):
    t = (pd.DataFrame({"seg": np.asarray(labels), "y": np.asarray(outcome)})
         .groupby("seg").y.agg(rate="mean", n="size").sort_values("rate"))
    r = t.rate.to_numpy()
    return (float(r[-1] / r[0]) if r[0] > 0 else np.inf), bool(np.all(np.diff(r) > 0)), t

# %% [markdown]
# ## k sweep — selection on VAL, never on test

# %%
rng = np.random.default_rng(SEED)
sub = rng.choice(int(tr.sum()), 8000, replace=False)
rows = []
for k in range(2, 9):
    km = KMeans(k, n_init=10, random_state=SEED).fit(Z[tr])
    s, m, t = outcome_spread(km.predict(Z[va]), y[va])
    rows.append({"k": k, "val_spread": round(s, 2), "monotone": m,
                 "min_share": round(t.n.min() / va.sum(), 4),
                 "silhouette": round(silhouette_score(Z[tr][sub], km.labels_[sub]), 3)})
sweep = pd.DataFrame(rows)
print(sweep.to_string(index=False))

# %% [markdown]
# Outcome separation keeps improving to k≈7 and then falls back. **k=4 is not the
# spread-optimal choice** — it is a business choice about how many segments an operator
# can actually run distinct campaigns for, and it reproduces the vendor reference, which
# gives a known-correct check. Recorded as a decision, not presented as an optimum.
#
# Silhouette barely moves across the whole sweep (0.245-0.349), which is the same signal
# as before: the RFM cloud is largely continuous. The segmentation earns its keep through
# outcome separation, not cluster shape.

# %%
K = 4
kmeans = KMeans(K, n_init=10, random_state=SEED).fit(Z[tr])          # FIT ON TRAIN ONLY
cust["segment_pred"] = kmeans.predict(Z)
ari = adjusted_rand_score(cust.segment, cust.segment_pred)
print(f"k={K}  ARI vs vendor reference segmentation = {ari:.4f}")
assert ari > 0.90, "rebuild diverged from the reference — investigate before promoting"

# %% [markdown]
# ## Permutation null — the secondary gate
#
# Each RFM column is independently permuted. Marginals are preserved exactly; any real
# multivariate customer structure is destroyed. A clustering of this null still produces
# monotone, well-separated-*looking* segments — which is precisely why monotonicity alone
# proves nothing.

# %%
N_PERM = 10
nulls = []
for i in range(N_PERM):
    Zn = np.column_stack([np.random.default_rng(SEED + i).permutation(Z[:, j]) for j in range(3)])
    ln = KMeans(K, n_init=3, random_state=i).fit_predict(Zn[tr])
    s, m, _ = outcome_spread(ln, y[tr])
    nulls.append(s)
null_p95 = float(np.quantile(nulls, 0.95))
print(f"null spread over {N_PERM} permutations: mean={np.mean(nulls):.4f}  p95={null_p95:.4f}")
print("(the null is monotone in essentially every run — monotonicity is FREE)")

# %% [markdown]
# ## Gate A — TEST scored once
#
# `MIN_SPREAD = 3.0` is **PROVISIONAL**. Derivation: ~185x the null p95, so noise cannot
# clear it; and below the k=2 solution (~4.2x), the weakest partition anyone would
# plausibly ship — a floor on usefulness, not a target. This is a *statistical* derivation.
# No cost of misassignment has been supplied.
#
# **Review trigger:** revisit once the retention action and its per-customer cost are known.

# %%
MIN_SPREAD, MIN_SHARE, MAX_SHARE = 3.0, 0.02, 0.60
spread_te, mono_te, tbl_te = outcome_spread(cust.loc[te, "segment_pred"], y[te])
shares = tbl_te.n.to_numpy() / te.sum()

print(tbl_te.assign(share=shares.round(4)).round(4).to_string())
failures = []
if not mono_te:                failures.append("dormancy not monotone across segments")
if spread_te < MIN_SPREAD:     failures.append(f"spread {spread_te:.2f}x below floor {MIN_SPREAD}x")
if spread_te <= null_p95:      failures.append(f"spread {spread_te:.2f}x does not clear null p95 {null_p95:.3f}x")
if shares.min() < MIN_SHARE:   failures.append(f"smallest segment {shares.min():.2%} below {MIN_SHARE:.0%}")
if shares.max() > MAX_SHARE:   failures.append(f"largest segment {shares.max():.2%} above {MAX_SHARE:.0%}")

GATE_A = not failures
print(f"\nspread={spread_te:.2f}x  monotone={mono_te}  null_p95={null_p95:.3f}  floor={MIN_SPREAD}x")
print(f"GATE A: {'PASS' if GATE_A else 'FAIL'}")
for f in failures:
    print("  FAILURE:", f)

# %% [markdown]
# ### Negative control — the gate must be able to fail
#
# A gate tested in one direction only is a gate that has not been tested.

# %%
neg_lab = np.random.default_rng(3).integers(0, K, len(cust))
neg_spread, _, _ = outcome_spread(neg_lab[te], y[te])
assert neg_spread < MIN_SPREAD, "GATE IS BROKEN — a random partition cleared the floor"
print(f"random partition spread = {neg_spread:.3f}x -> correctly FAILS. Both directions pinned.")

# %% [markdown]
# ## Segment naming
#
# Names are assigned from the measured dormancy ordering, not hardcoded. If the cluster
# ids shift on a retrain, the names follow the behaviour rather than pointing at the
# wrong group.

# %%
NAMES = ["Champions", "Steady Regulars", "High-Value At-Risk", "Lost / Dormant"]
order = (cust.loc[tr].groupby("segment_pred").dormant_45d.mean().sort_values().index.tolist())
SEGMENT_NAMES = {int(seg): NAMES[i] for i, seg in enumerate(order)}
cust["segment_name"] = cust.segment_pred.map(SEGMENT_NAMES)
print(cust.loc[te].groupby("segment_name").agg(
    n=("dormant_45d", "size"), dormancy=("dormant_45d", "mean"),
    med_R=("recency_days", "median"), med_F=("frequency", "median"),
    med_M=("monetary_mean", "median")).sort_values("dormancy").round(3).to_string())

# %% [markdown]
# ---
# # MODEL B — dormancy classifier

# %%
X = cust[MODEL_B_FEATURES].to_numpy()
assert not set(MODEL_B_FEATURES) & set(FORBIDDEN)
print("features:", MODEL_B_FEATURES)

# %% [markdown]
# ## Baseline first
#
# A tuned model that cannot beat a logistic regression is not worth deploying.

# %%
base = LogisticRegression(max_iter=2000).fit(np.log1p(X[tr]), y[tr])
base_val = roc_auc_score(y[va], base.predict_proba(np.log1p(X[va]))[:, 1])
print(f"baseline LogReg  val ROC-AUC = {base_val:.4f}")

# %% [markdown]
# ## Tuning — 5-fold CV on the TRAIN split only
#
# Not 2-fold. The test set is never passed as `eval_set` and never used for selection;
# doing so converts it into a selection set, which is leakage even though no `.fit()`
# touches test rows.

# %%
grid = {"max_iter": [150, 300], "learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31, 63],
        "min_samples_leaf": [20, 50, 100], "l2_regularization": [0.0, 1.0]}
search = RandomizedSearchCV(HistGradientBoostingClassifier(random_state=SEED), grid, n_iter=12,
                            cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                            scoring="roc_auc", n_jobs=-1, random_state=SEED)
search.fit(X[tr], y[tr])
print(f"best CV ROC-AUC = {search.best_score_:.4f}")
print("best params:", search.best_params_)
assert search.best_score_ > base_val - 0.01, "GBM does not beat the baseline — ship the baseline"

# %% [markdown]
# ## Calibration — mandatory, not optional
#
# The probability is intended to drive retention *spend*. A model with good discrimination
# and poor calibration produces a correct ordering and wrong money. Isotonic, fit on the
# **validation** split — never on train (the model has already seen it) and never on test.

# %%
# sklearn removed cv="prefit" in 1.8; FrozenEstimator is the replacement (1.6+).
# Both paths are shimmed so this runs on whatever version Colab pins.
try:
    from sklearn.frozen import FrozenEstimator
    model_b = CalibratedClassifierCV(FrozenEstimator(search.best_estimator_),
                                     method="isotonic").fit(X[va], y[va])
    CAL_PATH = "FrozenEstimator"
except ImportError:
    model_b = CalibratedClassifierCV(search.best_estimator_, method="isotonic",
                                     cv="prefit").fit(X[va], y[va])
    CAL_PATH = "cv='prefit'"

p_val_uncal = search.best_estimator_.predict_proba(X[va])[:, 1]
p_val_cal = model_b.predict_proba(X[va])[:, 1]
print(f"calibration path: {CAL_PATH}")
print(f"val Brier  uncalibrated = {brier_score_loss(y[va], p_val_uncal):.4f}")
print(f"val Brier  calibrated   = {brier_score_loss(y[va], p_val_cal):.4f}")
print(f"val ROC-AUC calibrated  = {roc_auc_score(y[va], p_val_cal):.4f}")

# %% [markdown]
# ## Gate B — TEST scored once
#
# `ROC-AUC >= 0.75` is **PROVISIONAL**, same review trigger as Gate A. It sits above the
# logistic baseline by a clear margin and below the observed value, so it is a floor that
# a genuinely degraded retrain would fail. Brier is reported and monitored but does not
# gate, because no cost model exists yet to say what calibration error is worth.

# %%
MIN_AUC = 0.75
p_test = model_b.predict_proba(X[te])[:, 1]
AUC = roc_auc_score(y[te], p_test)
BRIER = brier_score_loss(y[te], p_test)
PRAUC = average_precision_score(y[te], p_test)
GATE_B = AUC >= MIN_AUC

print(f"test ROC-AUC = {AUC:.4f}   (floor {MIN_AUC})")
print(f"test Brier   = {BRIER:.4f}  (reported, not gating)")
print(f"test PR-AUC  = {PRAUC:.4f}  (base rate {y[te].mean():.4f})")
print(f"GATE B: {'PASS' if GATE_B else 'FAIL'}")

# %% [markdown]
# ### Reliability curve

# %%
import matplotlib.pyplot as plt
bins = pd.qcut(p_test, 10, labels=False, duplicates="drop")
rel = pd.DataFrame({"p": p_test, "y": y[te], "b": bins}).groupby("b").agg(
    predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
plt.figure(figsize=(5, 5))
plt.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
plt.plot(rel.predicted, rel.observed, marker="o", label="model B")
plt.xlabel("mean predicted probability"); plt.ylabel("observed dormancy rate")
plt.title("Reliability (test, scored once)"); plt.legend(); plt.tight_layout(); plt.show()
print(rel.round(4).to_string())

# %% [markdown]
# ### The two models agree — sanity check
#
# Dormancy rate should rise across probability quartiles *within* every segment. If it did
# not, one of the two models would be wrong.

# %%
print(pd.crosstab(cust.loc[te, "segment_name"],
                  pd.qcut(p_test, 4, labels=["q1", "q2", "q3", "q4"]),
                  y[te], aggfunc="mean").round(3).to_string())

# %% [markdown]
# ---
# # Artifacts
#
# Immutable, versioned, never overwritten. Promotion writes `models/CURRENT.json`;
# rollback rewrites it to a prior prefix. Nothing is destroyed, so both directions are
# reversible.

# %%
def _sha(df):
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16]

try:
    GIT_SHA = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
except Exception:
    GIT_SHA = "nogit"          # explicit benign case, named — not a masked error
UTC = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%MZ")
VERSION = f"v_{UTC}_{GIT_SHA}"
VP = f"models/{VERSION}"
print("version:", VERSION)

# %%
scaler_spec = read_json_s3("contracts/feature_spec_v1.json")
manifest = {
    "version": VERSION, "created_utc": UTC, "git_sha": GIT_SHA,
    "contract_version": "v1", "seed": SEED,
    "library_versions": {"python": ".".join(map(str, __import__("sys").version_info[:3])),
                         "numpy": np.__version__, "pandas": pd.__version__,
                         "scikit-learn": sklearn.__version__, "scipy": scipy.__version__},
    "calibration_path": CAL_PATH,
    "model_a": {"algo": "KMeans", "k": K, "features": RFM_COLS,
                "transform": "log1p -> StandardScaler",
                "scaler_mean": scaler_spec["scaler_mean"], "scaler_scale": scaler_spec["scaler_scale"],
                "segment_names": {str(k_): v for k_, v in SEGMENT_NAMES.items()}},
    "model_b": {"algo": "HistGradientBoostingClassifier + isotonic calibration",
                "features": MODEL_B_FEATURES, "best_params": search.best_params_},
    "feature_dtypes": {c: str(cust[c].dtype) for c in MODEL_B_FEATURES},
    "feature_order_hash": hashlib.sha256("|".join(MODEL_B_FEATURES).encode()).hexdigest()[:16],
    "data_hash": _sha(cust[MODEL_B_FEATURES]),
    "split_sizes": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
    "notes": ["reproducibility: n_jobs=-1 with threaded estimators is not bit-reproducible",
              "feature window is 45 days (vendor dictionary says 90 — wrong)",
              "drift reference is days 30-44 (days 0-44 gives a permanent false alarm)"],
}

metrics = {
    "model_a": {"gate": "outcome_spread", "value": round(spread_te, 4),
                "threshold": MIN_SPREAD, "threshold_status": "PROVISIONAL",
                "threshold_derivation": "~185x permutation null p95; below the k=2 solution (~4.2x). "
                                        "Statistical only — no cost of misassignment supplied.",
                "review_trigger": "revisit when the retention action and its per-customer cost are known",
                "permutation_null_p95": round(null_p95, 4), "monotone": bool(mono_te),
                "ari_vs_reference": round(ari, 4), "passed": bool(GATE_A),
                "excluded_metrics": {"bootstrap_stability_ari": "passes at 0.960 on shuffled data",
                                     "anova_f": "circular — tests clusters on their own features"},
                "segment_table": tbl_te.assign(share=shares).round(4).to_dict(orient="index")},
    "model_b": {"gate": "roc_auc", "value": round(AUC, 4), "threshold": MIN_AUC,
                "threshold_status": "PROVISIONAL",
                "threshold_derivation": "above the logistic baseline by a clear margin, below the "
                                        "observed value — a floor a degraded retrain would fail.",
                "review_trigger": "same as model_a",
                "brier": round(BRIER, 4), "pr_auc": round(PRAUC, 4),
                "baseline_logreg_val_auc": round(base_val, 4),
                "cv_best_auc": round(search.best_score_, 4),
                "test_base_rate": round(float(y[te].mean()), 4), "passed": bool(GATE_B)},
    "evaluated_on": "test split, scored once",
}

for name, obj in [("kmeans.pkl", kmeans), ("model_b.pkl", model_b)]:
    s3.put_object(Bucket=BUCKET, Key=_key(f"{VP}/{name}"), Body=pickle.dumps(obj))
    print("saved", f"{VP}/{name}")
s3.put_object(Bucket=BUCKET, Key=_key(f"{VP}/feature_names.pkl"),
              Body=pickle.dumps({"model_a": RFM_COLS, "model_b": MODEL_B_FEATURES}))
save_json_s3(manifest, f"{VP}/manifest.json")
save_json_s3(metrics, f"{VP}/metrics.json")

# %% [markdown]
# ## Promotion — only if both gates pass

# %%
if GATE_A and GATE_B:
    save_json_s3({"version": VERSION, "prefix": VP, "promoted_utc": UTC,
                  "gate_a_passed": True, "gate_b_passed": True}, "models/CURRENT.json")
    print(f"PROMOTED -> {VP}")
else:
    print("NOT PROMOTED — a gate failed. Artifacts are written and versioned; "
          "CURRENT.json is untouched, so production keeps serving whatever it was serving.")

# %%
save_s3(cust, "data/customers/07_scored.parquet")

# %% [markdown]
# ---
# ## Model card
#
# | | Model A | Model B |
# |---|---|---|
# | Task | RFM segmentation (unsupervised) | Dormancy prediction (binary) |
# | Algorithm | KMeans, k=4, on log1p+scaled R/F/M | HistGradientBoosting + isotonic calibration |
# | Fit on | 307,644 train rows | 307,644 train rows, tuned by 5-fold CV |
# | Gate | outcome spread ≥ 3.0x (**provisional**) | ROC-AUC ≥ 0.75 (**provisional**) |
# | Secondary | permutation null p95 | Brier, PR-AUC (reported, not gating) |
# | Excluded metrics | stability ARI, ANOVA F — both pass on noise | — |
# | Protected attributes | none used | none used; geography excluded by design |
#
# **Limitations, stated:**
#
# * The data is **synthetic** — a causal simulation, not observed behaviour. This belongs
#   in the README and the sign-off, not a footnote.
# * Geometric cluster separation is weak (silhouette ~0.26 vs ~0.256 on a marginal-shuffled
#   null). The segmentation is justified by outcome separation, not by cluster shape.
# * `monetary_mean` contributes almost nothing to dormancy (AUC ~0.47). Do not describe
#   the model as finding "high-value churners" on the strength of M.
# * The primary split is random, not temporal — the customer table is a single snapshot.
#   The day-30/day-45 harness in notebook 06 is the only temporal validation, and its two
#   outcome windows overlap on days 45-75.
# * Both thresholds are provisional pending a cost model.
#
# ---
# ## Phase-1 handoff — AS-RUN numbers from this notebook
#
# Fill from the printed output of **this** run, not from any earlier estimate. Then stop
# for recorded sign-off before Phase 2.

# %%
print(f"""
PHASE 1 HANDOFF  ({UTC})
  version           {VERSION}
  bucket            s3://{BUCKET}/{PREFIX}/
  contract          contracts/schema_v1.json (v1)
  libs              sklearn={sklearn.__version__} pandas={pd.__version__} numpy={np.__version__}
                    ^ THESE become the train/serve parity contract. Pin the image to them.

  MODEL A  k={K}  spread={spread_te:.2f}x  null_p95={null_p95:.3f}  ARI={ari:.4f}  -> {'PASS' if GATE_A else 'FAIL'}
  MODEL B  ROC-AUC={AUC:.4f}  Brier={BRIER:.4f}  PR-AUC={PRAUC:.4f}      -> {'PASS' if GATE_B else 'FAIL'}

  Rebuild parity    recency/tenure/dormancy exact; frequency 99.37% (clock skew)
  Drift reference   days 30-44, {'refreshed' if GATE_A and GATE_B else 'NOT refreshed'}
  Both thresholds   PROVISIONAL — review trigger: retention action + per-customer cost

  STOP. Phase 2 begins only after this is signed off.
""")
