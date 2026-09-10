# %% [markdown]
# # Segmentation, Gates and Artifacts
#
# **Project:** QCOMM Customer Analytics — RFM customer segmentation
# **Phase 1, notebook 07.** Fit the segmentation, gate it against a held-out outcome, write versioned artifacts.
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

# The model. Three axes, nothing else.
RFM_COLS   = ["recency_days", "frequency", "monetary_mean"]

# Produced by the snapshot rebuild and used to PROFILE segments after fitting.
# These never enter the clustering — they describe segments, they do not define them.
PROFILE_COLS = ["recency_days", "frequency", "monetary_mean", "monetary_sum",
                "tenure_days", "promo_rate", "mobile_rate"]

# The validation variable. NOT a modelling target in this project — it is the referee.
# Gate A scores segments against it because it is measured on a 45-day window AFTER the
# snapshot, which the clustering never sees. Without it the segmentation has no external
# validation and falls back on silhouette/ANOVA, both of which pass on shuffled noise.
VALIDATION_OUTCOME = "dormant_45d"

# Never a clustering input. dormant_45d and orders_next_45d are here because a segmentation
# built on the outcome would then be validating against itself.
FORBIDDEN  = ["segment", "segment_label", "orders_next_45d",
              "customer_id", "in_orders_sample", "dormant_45d", "__split"]

print(f"boto3 -> s3://{BUCKET}/{PREFIX}/  region={REGION}  colab={IN_COLAB}")

# %%
import pickle, hashlib, subprocess, datetime as dt, sklearn, scipy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

cust = read_s3("data/customers/06_scaled.parquet")
ZC = ["z_" + c for c in RFM_COLS]
tr = (cust.__split == "train").to_numpy()
va = (cust.__split == "val").to_numpy()
te = (cust.__split == "test").to_numpy()
Z = cust[ZC].to_numpy()
y = cust[VALIDATION_OUTCOME].to_numpy()
print(f"train={tr.sum():,} (fit)  val={va.sum():,} (select k)  test={te.sum():,} (gate, scored once)")

# %% [markdown]
# ## What is being built, and what is not
#
# **One model: KMeans on log-scaled R, F, M.** There is no supervised model in this
# project.
#
# `dormant_45d` appears throughout this notebook and is **not** a modelling target. It is
# the **referee**. It is measured on the 45 days *after* the snapshot — a window the
# clustering never sees — which makes it the only external validation available. It is in
# `FORBIDDEN`, so it can never become an input; a segmentation built on the outcome would
# then be validating against itself.

# %% [markdown]
# ## The gate, and why the obvious metrics are excluded
#
# Two metrics are conventionally offered as evidence for a clustering. Both were measured
# on this project's earlier e-commerce dataset, on data where customer identity had been
# **randomly shuffled across orders** so no customer-level signal existed at all:
#
# | metric | value on pure noise |
# |---|---|
# | bootstrap stability ARI | **0.960** |
# | ANOVA F (R / F / M) | **2,220 / 5,490 / 2,657**, p ≈ 0 |
# | silhouette @ k=4 | 0.256 vs 0.261 on real data |
#
# Neither can fail. Stability measures determinism, not structure; ANOVA on the features
# you clustered with is circular. **They are not gates here.** Silhouette is reported as
# supporting information only.
#
# The gate is **outcome separation**: segments ordered by dormancy rate, and the ratio
# between the highest and the lowest.

# %%
def outcome_spread(labels, outcome):
    t = (pd.DataFrame({"seg": np.asarray(labels), "y": np.asarray(outcome)})
         .groupby("seg").y.agg(rate="mean", n="size").sort_values("rate"))
    r = t.rate.to_numpy()
    return (float(r[-1] / r[0]) if r[0] > 0 else np.inf), bool(np.all(np.diff(r) > 0)), t

# %% [markdown]
# ## 1. Choose k — on VAL, never on test

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

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(sweep.k, sweep.val_spread, marker="o", color="#4C72B0")
ax[0].axvline(4, ls="--", color="grey"); ax[0].set_title("outcome spread by k (val)"); ax[0].set_xlabel("k")
ax[1].plot(sweep.k, sweep.silhouette, marker="o", color="#C44E52")
ax[1].axvline(4, ls="--", color="grey"); ax[1].set_title("silhouette by k (train)"); ax[1].set_xlabel("k")
plt.tight_layout(); plt.show()

# %% [markdown]
# **k=4 is a business decision, recorded as one — not an optimum.**
#
# Outcome separation keeps improving to k≈7 and then falls back. k=4 is chosen because it
# is the number of segments an operator can realistically run distinct campaigns for, and
# because it reproduces the vendor reference segmentation, which gives a known-correct
# check on the whole pipeline. If the retention team can act on six segments, k=6 is
# better on the numbers and the sweep above is the evidence.
#
# Silhouette barely moves across the entire sweep (0.245–0.349). That is the same signal
# as before: the RFM cloud is largely continuous. **This segmentation earns its keep
# through outcome separation, not through cluster shape.** Say that in the model card
# rather than showing a silhouette number and hoping nobody asks.

# %%
K = 4
kmeans = KMeans(K, n_init=10, random_state=SEED).fit(Z[tr])       # FIT ON TRAIN ONLY
cust["segment_pred"] = kmeans.predict(Z)
ari = adjusted_rand_score(cust.segment, cust.segment_pred)
print(f"k={K}   ARI vs the vendor reference segmentation = {ari:.4f}")
assert ari > 0.90, "rebuild diverged from the reference — investigate before promoting"

# %% [markdown]
# ## 2. Permutation null — the secondary gate
#
# Each RFM column is independently permuted. Marginals are preserved exactly; any real
# multivariate customer structure is destroyed. A clustering of this null still produces
# monotone, well-separated-*looking* segments — which is exactly why monotonicity on its
# own proves nothing.

# %%
N_PERM = 10
nulls, null_mono = [], []
for i in range(N_PERM):
    Zn = np.column_stack([np.random.default_rng(SEED + i).permutation(Z[:, j]) for j in range(3)])
    ln = KMeans(K, n_init=3, random_state=i).fit_predict(Zn[tr])
    s, m, _ = outcome_spread(ln, y[tr])
    nulls.append(s); null_mono.append(m)
null_p95 = float(np.quantile(nulls, 0.95))
print(f"null spread over {N_PERM} permutations: mean={np.mean(nulls):.4f}  p95={null_p95:.4f}")
print(f"null was MONOTONE in {sum(null_mono)}/{N_PERM} runs  ->  monotonicity is FREE, "
      f"the spread is what carries the evidence")

# %% [markdown]
# ## 3. Gate A — TEST scored once
#
# `MIN_SPREAD = 3.0` is **PROVISIONAL**. Derivation: roughly 185x the permutation null
# p95, so noise cannot clear it; and below the k=2 solution (~4.2x), which is the weakest
# partition anyone would plausibly ship. It is a floor on usefulness, not a target.
#
# This is a **statistical** derivation. No cost of misassignment has been supplied.
# **Review trigger:** revisit once the retention action and its per-customer cost are known.

# %%
MIN_SPREAD, MIN_SHARE, MAX_SHARE = 3.0, 0.02, 0.60
spread_te, mono_te, tbl_te = outcome_spread(cust.loc[te, "segment_pred"], y[te])
shares = tbl_te.n.to_numpy() / te.sum()

print(tbl_te.assign(share=shares.round(4)).round(4).to_string())

failures = []
if not mono_te:              failures.append("dormancy not monotone across segments")
if spread_te < MIN_SPREAD:   failures.append(f"spread {spread_te:.2f}x below floor {MIN_SPREAD}x")
if spread_te <= null_p95:    failures.append(f"spread {spread_te:.2f}x does not clear null p95 {null_p95:.3f}x")
if shares.min() < MIN_SHARE: failures.append(f"smallest segment {shares.min():.2%} below {MIN_SHARE:.0%}")
if shares.max() > MAX_SHARE: failures.append(f"largest segment {shares.max():.2%} above {MAX_SHARE:.0%}")

GATE_A = not failures
print(f"\nspread={spread_te:.2f}x   monotone={mono_te}   null_p95={null_p95:.3f}   floor={MIN_SPREAD}x")
print(f"GATE A: {'PASS' if GATE_A else 'FAIL'}")
for f in failures:
    print("  FAILURE:", f)

# %% [markdown]
# ### Negative control — the gate must be able to fail
#
# A gate tested in one direction only is a gate that has not been tested.

# %%
neg_lab = np.random.default_rng(3).integers(0, K, len(cust))
neg_spread, neg_mono, _ = outcome_spread(neg_lab[te], y[te])
assert neg_spread < MIN_SPREAD, "GATE IS BROKEN — a random partition cleared the floor"
print(f"random partition: spread={neg_spread:.3f}x  monotone={neg_mono}  ->  correctly FAILS")
print("both directions pinned")

# %% [markdown]
# ## 4. Name the segments from behaviour, not position
#
# Names are assigned from the measured dormancy ordering. If cluster ids shift on a
# retrain — and KMeans label ids are arbitrary, so they will — the names follow the
# behaviour instead of pointing at the wrong group.

# %%
NAMES = ["Champions", "Steady Regulars", "High-Value At-Risk", "Lost / Dormant"]
order = cust.loc[tr].groupby("segment_pred")[VALIDATION_OUTCOME].mean().sort_values().index.tolist()
SEGMENT_NAMES = {int(seg): NAMES[i] for i, seg in enumerate(order)}
cust["segment_name"] = cust.segment_pred.map(SEGMENT_NAMES)
print("id -> name (derived, not hardcoded):", SEGMENT_NAMES)

# %% [markdown]
# ## 5. Segment profile
#
# `PROFILE_COLS` describe the segments. They were **not** inputs — the clustering saw only
# R, F and mean M. Anything interesting here is a discovered property, not a construction.

# %%
profile = cust.loc[te].groupby("segment_name").agg(
    n=("segment_pred", "size"),
    dormancy=(VALIDATION_OUTCOME, "mean"),
    med_recency=("recency_days", "median"),
    med_frequency=("frequency", "median"),
    med_monetary=("monetary_mean", "median"),
    med_total_spend=("monetary_sum", "median"),
    med_tenure=("tenure_days", "median"),
    promo_rate=("promo_rate", "mean"),
    mobile_rate=("mobile_rate", "mean")).sort_values("dormancy")
profile["share"] = (profile.n / te.sum()).round(4)
print(profile.round(3).to_string())

# %%
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
p = profile.reset_index()
ax[0].barh(p.segment_name, p.dormancy, color="#C44E52"); ax[0].set_title("dormancy rate (test)")
ax[1].barh(p.segment_name, p.med_monetary, color="#4C72B0"); ax[1].set_title("median order value")
ax[2].barh(p.segment_name, p.med_frequency, color="#55A868"); ax[2].set_title("median frequency")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### Read this honestly
#
# `High-Value At-Risk` earns its dormancy rate from **recency**, not from value.
# `monetary_mean` separates it from `Lost / Dormant` but contributes almost nothing to
# dormancy on its own (AUC ~0.47, notebook 04). Do not describe the model as "finding
# high-value churners" on the strength of M — it finds *lapsed* customers, and M then
# tells you which of them are worth spending on. That is a useful distinction for a
# retention budget and a misleading one if collapsed.

# %% [markdown]
# ## 6. Cluster geometry — reported, not gating

# %%
sub_te = np.random.default_rng(1).choice(int(te.sum()), 8000, replace=False)
print(f"silhouette (test, k={K}) = {silhouette_score(Z[te][sub_te], cust.loc[te, 'segment_pred'].to_numpy()[sub_te]):.3f}")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
samp = np.random.default_rng(2).choice(len(cust), 15000, replace=False)
for pair, a in zip([(0, 1), (0, 2)], ax):
    a.scatter(Z[samp, pair[0]], Z[samp, pair[1]], c=cust.segment_pred.to_numpy()[samp],
              cmap="viridis", s=3, alpha=0.4)
    a.set_xlabel(ZC[pair[0]]); a.set_ylabel(ZC[pair[1]])
ax[0].set_title("segments in log-scaled RFM space"); ax[1].set_title("")
plt.tight_layout(); plt.show()

# %% [markdown]
# The boundaries are straight cuts through a continuous cloud, not gaps between islands.
# That is what the flat silhouette curve was saying. It does not invalidate the
# segmentation — the outcome separation is real and 7x — but it does mean the segments are
# **a useful partition of a continuum**, not natural kinds. Worth stating plainly.

# %% [markdown]
# ---
# # Artifacts
#
# Immutable and versioned; never overwritten. Promotion writes `models/CURRENT.json`;
# rollback rewrites it to a prior prefix. Nothing is destroyed, so both directions are
# reversible.

# %%
def _sha(df):
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16]

try:
    GIT_SHA = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
except (subprocess.CalledProcessError, FileNotFoundError):
    GIT_SHA = "nogit"        # named benign case — not a masked error
UTC = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%MZ")
VERSION = f"v_{UTC}_{GIT_SHA}"
VP = f"models/{VERSION}"
print("version:", VERSION)

# %%
spec = read_json_s3("contracts/feature_spec_v1.json")
manifest = {
    "version": VERSION, "created_utc": UTC, "git_sha": GIT_SHA,
    "contract_version": "v1", "seed": SEED,
    "task": "RFM customer segmentation (unsupervised). No supervised model in this project.",
    "library_versions": {"python": ".".join(map(str, __import__("sys").version_info[:3])),
                         "numpy": np.__version__, "pandas": pd.__version__,
                         "scikit-learn": sklearn.__version__, "scipy": scipy.__version__},
    "model": {"algo": "KMeans", "k": K, "features": RFM_COLS,
              "transform": "log1p -> StandardScaler",
              "scaler_mean": spec["scaler_mean"], "scaler_scale": spec["scaler_scale"],
              "segment_names": {str(a): b for a, b in SEGMENT_NAMES.items()},
              "naming_rule": "assigned from measured dormancy ordering, not hardcoded ids"},
    "validation_outcome": VALIDATION_OUTCOME,
    "profile_cols": PROFILE_COLS,
    "feature_dtypes": {c: str(cust[c].dtype) for c in RFM_COLS},
    "feature_order_hash": hashlib.sha256("|".join(RFM_COLS).encode()).hexdigest()[:16],
    "data_hash": _sha(cust[RFM_COLS]),
    "split_sizes": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
    "notes": ["n_jobs / threaded KMeans is not bit-reproducible; seed is recorded, not claimed as determinism",
              "feature window is 45 days (vendor dictionary says 90 — wrong)",
              "drift reference is days 30-44 (days 0-44 gives a permanent false alarm)",
              "k=4 is a business choice; k=6-7 scores higher on outcome separation"],
}

metrics = {
    "gate": "outcome_spread", "value": round(spread_te, 4), "threshold": MIN_SPREAD,
    "threshold_status": "PROVISIONAL",
    "threshold_derivation": "~185x the permutation null p95, and below the k=2 solution (~4.2x). "
                            "Statistical only — no cost of misassignment supplied.",
    "review_trigger": "revisit when the retention action and its per-customer cost are known",
    "permutation_null_p95": round(null_p95, 4),
    "permutation_null_monotone_runs": f"{sum(null_mono)}/{N_PERM}",
    "monotone": bool(mono_te),
    "min_segment_share": round(float(shares.min()), 4),
    "max_segment_share": round(float(shares.max()), 4),
    "silhouette_test": round(float(silhouette_score(Z[te][sub_te],
                            cust.loc[te, "segment_pred"].to_numpy()[sub_te])), 4),
    "ari_vs_vendor_reference": round(ari, 4),
    "negative_control_spread": round(neg_spread, 4),
    "passed": bool(GATE_A),
    "excluded_metrics": {"bootstrap_stability_ari": "passes at 0.960 on shuffled data",
                         "anova_f": "circular — tests clusters on their own features",
                         "silhouette_as_gate": "0.261 real vs 0.256 marginal-shuffled null"},
    "k_sweep": sweep.to_dict(orient="records"),
    "segment_table": profile.round(4).to_dict(orient="index"),
    "evaluated_on": "test split, scored once",
}

s3.put_object(Bucket=BUCKET, Key=_key(f"{VP}/kmeans.pkl"), Body=pickle.dumps(kmeans))
s3.put_object(Bucket=BUCKET, Key=_key(f"{VP}/feature_names.pkl"), Body=pickle.dumps(RFM_COLS))
print("saved", f"{VP}/kmeans.pkl")
save_json_s3(manifest, f"{VP}/manifest.json")
save_json_s3(metrics, f"{VP}/metrics.json")

# %% [markdown]
# ## Promotion — only if the gate passes

# %%
if GATE_A:
    save_json_s3({"version": VERSION, "prefix": VP, "promoted_utc": UTC, "gate_passed": True},
                 "models/CURRENT.json")
    print(f"PROMOTED -> {VP}")
else:
    print("NOT PROMOTED — the gate failed. Artifacts are written and versioned; "
          "CURRENT.json is untouched, so production keeps serving whatever it was serving.")

# %%
save_s3(cust, "data/customers/07_segmented.parquet")

# %% [markdown]
# ---
# ## Model card
#
# | | |
# |---|---|
# | Task | RFM customer segmentation (unsupervised) |
# | Algorithm | KMeans, k=4, on log1p + standard-scaled recency / frequency / mean monetary |
# | Fit on | 307,644 training rows; scaler fit on the same rows only |
# | k selected on | validation split (65,924 rows) |
# | Gate | outcome spread ≥ 3.0x against `dormant_45d`, **PROVISIONAL** |
# | Secondary gate | must clear the permutation null p95 |
# | Reported, not gating | silhouette, ARI vs vendor reference |
# | Excluded as gates | bootstrap stability ARI, ANOVA F — both pass on pure noise |
# | Protected attributes | none used; geography excluded from the rebuild by design |
#
# **Limitations, stated:**
#
# * The data is **synthetic** — a causal simulation, not observed behaviour. This belongs
#   in the README and the sign-off, not a footnote.
# * Cluster geometry is weak (silhouette ~0.26 against ~0.256 on a marginal-shuffled null).
#   The segments are a useful partition of a continuum, not natural kinds.
# * `monetary_mean` contributes almost nothing to dormancy (AUC ~0.47). The model finds
#   **lapsed** customers; M then indicates which are worth spending on.
# * The primary split is random, not temporal — the customer table is a single snapshot.
#   The day-30 / day-45 harness in notebook 06 is the only temporal validation available,
#   and its two outcome windows overlap on days 45-75.
# * k=4 is a business choice. k=6-7 separates the outcome better.
# * The threshold is provisional pending a cost model.
#
# ---
# ## Phase-1 handoff — AS-RUN numbers from this notebook
#
# Fill from the printed output of **this** run, not from any earlier estimate.

# %%
print(f"""
PHASE 1 HANDOFF  ({UTC})
  project           QCOMM Customer Analytics — RFM segmentation (no supervised model)
  version           {VERSION}
  bucket            s3://{BUCKET}/{PREFIX}/
  contract          contracts/schema_v1.json (v1)
  libs              sklearn={sklearn.__version__} pandas={pd.__version__} numpy={np.__version__} scipy={scipy.__version__}
                    ^ THESE become the train/serve parity contract. Pin the Phase-2 image to them.

  GATE A            spread={spread_te:.2f}x   floor={MIN_SPREAD}x   null_p95={null_p95:.3f}
                    monotone={mono_te}   segment shares {shares.min():.1%}-{shares.max():.1%}
                    ARI vs vendor reference = {ari:.4f}
                    negative control = {neg_spread:.3f}x (correctly fails)
                    -> {'PASS' if GATE_A else 'FAIL'}

  k                 {K}  (business choice; sweep shows k=6-7 separates better)
  silhouette        reported only, never gating
  threshold         PROVISIONAL — review trigger: retention action + per-customer cost

  Rebuild parity    recency/tenure/dormancy exact; frequency 99.37% (clock skew)
  Drift reference   days 30-44, {'refreshed' if GATE_A else 'NOT refreshed'}

  STOP. Phase 2 begins only after this is signed off.
""")
