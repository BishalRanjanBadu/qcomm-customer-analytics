# %% [markdown]
# # Scaling and the Feature Pipeline
#
# **Project:** QCOMM Customer Analytics — RFM customer segmentation
# **Phase 1, notebook 06.** Scaler fit on train rows only. Chain B rebuilds the customer table from raw events — this becomes src/preprocess.py.
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
from sklearn.preprocessing import StandardScaler
cust   = read_s3("data/customers/05_hypothesis.parquet")
orders = read_s3("data/orders/03_imputed.parquet")
train  = cust.__split == "train"

# %% [markdown]
# ## 1. Feature-set declaration — by name, deterministic
#
# **No numerical threshold decides the feature set.** A rule like "drop where VIF is
# non-finite" can select a different set on a different library version — the same
# notebook on the same data producing a 25-feature model on one machine and 28 on another.
# The set is declared by name and asserted.

# %%
print("Clustering features (the model):", RFM_COLS)
print("Profile-only (describe segments, never define them):",
      [c for c in PROFILE_COLS if c not in RFM_COLS])
assert not (set(RFM_COLS) | set(PROFILE_COLS)) & set(FORBIDDEN)
assert "monetary_sum" not in RFM_COLS, "monetary_sum would double-count frequency in the distance metric"
for c in RFM_COLS + PROFILE_COLS:
    assert c in cust.columns, c
print("\nforbidden (never a feature):", FORBIDDEN)

# %% [markdown]
# ## 2. Scale — fit on TRAIN rows only
#
# The single most common leakage site. The `fit`/`transform` signature is not enough:
# what matters is the **call site**. `.fit()` sees `logA[train]` and nothing else.

# %%
logA = np.log1p(cust[RFM_COLS].to_numpy())
scaler = StandardScaler().fit(logA[train.to_numpy()])          # <-- TRAIN ONLY
Z = scaler.transform(logA)                                      # applied to everything
for i, c in enumerate(RFM_COLS):
    cust["z_" + c] = Z[:, i]

print("scaler.mean_ :", scaler.mean_.round(4))
print("scaler.scale_:", scaler.scale_.round(4))
print("\ntrain z-mean (must be ~0):", Z[train.to_numpy()].mean(0).round(6))
print("test  z-mean (drifts slightly — correct, it was not fitted):",
      Z[(cust.__split == 'test').to_numpy()].mean(0).round(4))
assert np.abs(Z[train.to_numpy()].mean(0)).max() < 1e-6

# %% [markdown]
# ### Leakage regression test
#
# Executable, not a comment. If the fitted statistic changes when test rows are appended
# to the fit input, the boundary has been broken. This test belongs in `tests/` in Phase 2.

# %%
sc_train_only = StandardScaler().fit(logA[train.to_numpy()])
sc_contaminated = StandardScaler().fit(logA)
assert np.allclose(sc_train_only.mean_, scaler.mean_)
assert not np.allclose(sc_contaminated.mean_, scaler.mean_), "no difference -> the split is not being honoured"
print("leakage regression: contaminated fit differs from train-only fit -> boundary is real")
print(f"  train-only mean_   : {sc_train_only.mean_.round(6)}")
print(f"  contaminated mean_ : {sc_contaminated.mean_.round(6)}")

# %%
save_s3(cust, "data/customers/06_scaled.parquet")
save_json_s3({"rfm_cols": RFM_COLS, "profile_cols": PROFILE_COLS,
              "validation_outcome": VALIDATION_OUTCOME,
              "forbidden": FORBIDDEN, "transform": "log1p -> StandardScaler",
              "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
              "fit_rows": "__split == 'train'", "n_fit_rows": int(train.sum())},
             "contracts/feature_spec_v1.json")

# %% [markdown]
# ---
# ## 3. Chain B — rebuild the customer table from raw events
#
# **This is the deliverable that makes the project an MLOps project.** The vendor's
# customer table is a static file; this function is the thing that produces one, at any
# snapshot date, from the event stream. In Phase 2 it becomes `src/preprocess.py`; in
# Phase 3 the retraining pipeline calls it on a rolling window.
#
# Note what it does *not* take: `city_id`, `delivery_city`, `billing_city`. Geography is
# the classic proxy for protected characteristics and is deliberately excluded (§2.11).

# %%
def build_snapshot(orders_df, snap_day, horizon_days=45):
    """Rebuild the customer feature table at an arbitrary snapshot.

    Row-preserving and target-free on the feature side. Features use orders on or before
    the snapshot; the outcome uses the horizon strictly after it. The two never overlap.
    """
    snap = DATA_START + pd.Timedelta(days=snap_day)
    horizon_end = snap + pd.Timedelta(days=horizon_days)
    if horizon_end > DATA_END:
        raise ValueError(f"snapshot {snap_day} + {horizon_days}d horizon ends {horizon_end.date()}, "
                         f"past the cutoff {DATA_END.date()} — the outcome would be censored")

    pre = orders_df[orders_df.event_time <= snap]
    g = pre.groupby("customer_id").agg(
        recency_days=("event_time", lambda s: (snap - s.max()).total_seconds() / 86400),
        frequency=("event_time", "size"),
        monetary_mean=("order_value", "mean"),
        monetary_sum=("order_value", "sum"),
        tenure_days=("event_time", lambda s: (snap - s.min()).total_seconds() / 86400),
        promo_rate=("promo_flag", "mean"),
        mobile_rate=("channel", lambda s: (s == "mobile_app").mean()))

    post = orders_df[(orders_df.event_time > snap) & (orders_df.event_time <= horizon_end)]
    nxt = post.groupby("customer_id").size()
    g["orders_next_45d"] = nxt.reindex(g.index).fillna(0).astype(int)
    g["dormant_45d"] = (g.orders_next_45d == 0).astype(int)
    return g.reset_index()

# %% [markdown]
# ### D2 — apply the join rule before rebuilding

# %%
flagged = set(cust.loc[cust.in_orders_sample == 1, "customer_id"])
ob = orders[orders.customer_id.isin(flagged)]
print(f"orders after join rule: {len(ob):,} rows, {ob.customer_id.nunique():,} customers")
assert ob.customer_id.nunique() == 35189

# %% [markdown]
# ### Parity check — the rebuild must reproduce the vendor's table
#
# This is the test that proves the feature pipeline is correct. It is the Phase-2
# transform-parity test in embryo.

# %%
snap45 = build_snapshot(ob, 45)
ref = cust[cust.in_orders_sample == 1].set_index("customer_id")
chk = snap45.set_index("customer_id").join(
    ref[["recency_days", "frequency", "tenure_days", "orders_next_45d", "dormant_45d"]], rsuffix="_ref")

print(f"parity on {len(chk):,} customers:")
for c in ["recency_days", "frequency", "tenure_days", "orders_next_45d"]:
    m = (chk[c].sub(chk[c + "_ref"]).abs() < 1e-3).mean()
    print(f"  {c:16s} exact match = {m:.4f}")
dm = (chk.dormant_45d == chk.dormant_45d_ref).mean()
print(f"  {'dormant_45d':16s} exact match = {dm:.4f}")

assert (chk.recency_days.sub(chk.recency_days_ref).abs() < 1e-3).mean() == 1.0
assert dm == 1.0
assert (chk.frequency == chk.frequency_ref).mean() > 0.99

# %% [markdown]
# `recency_days`, `tenure_days` and `dormant_45d` reproduce **exactly**. `frequency`
# matches on 99.37% of customers — the residual is the clock skew flagged in notebook 02:
# `event_time` is device-reported while the vendor computed from true time, so orders
# within 900s of the snapshot fall on the other side of the boundary. That is the
# documented defect behaving as documented, not a pipeline error.

# %%
mismatch = chk[chk.frequency != chk.frequency_ref]
print(f"frequency mismatches: {len(mismatch):,} ({len(mismatch)/len(chk):.4%}), "
      f"max |diff| = {(mismatch.frequency - mismatch.frequency_ref).abs().max():.0f} order(s)")

# %% [markdown]
# ## 4. The temporal harness
#
# The customer table is a single snapshot, so a temporal split on it is impossible. The
# orders file makes one available: fit at day 30, validate at day 45.
#
# **Recorded caveat:** the two outcome windows overlap on days 45-75. This is not a clean
# holdout. It is still materially better than a random split, and it is the only temporal
# validation this dataset supports. A day-60 snapshot is *not* usable — its horizon would
# end on day 105, past the cutoff, so the outcome would be censored. The function above
# raises rather than silently producing a biased label.

# %%
snap30 = build_snapshot(ob, 30)
print(f"day-30 snapshot: {len(snap30):,} customers, dormancy={snap30.dormant_45d.mean():.4f}")
print(f"day-45 snapshot: {len(snap45):,} customers, dormancy={snap45.dormant_45d.mean():.4f}")
try:
    build_snapshot(ob, 60)
except ValueError as e:
    print(f"\nday-60 correctly refused: {e}")

# %%
save_s3(snap45, "data/orders/06_snapshot_d45.parquet")
save_s3(snap30, "data/orders/06_snapshot_d30.parquet")

# %% [markdown]
# ## 5. Drift reference — days 30-44, not 0-44
#
# `orders_30d` is a 30-day trailing count on a panel that starts at day 0. Its mean ramps
# from 0.13 to 8.00 across the first month. A reference window of days 0-44 puts that
# feature at PSI 0.2585 on the very first monitored window — a **permanent false alarm**
# on the highest-drift feature in the dataset. With days 30-44 it drops to 0.0013.

# %%
def psi(reference, current, bins=10):
    e = np.unique(np.nanquantile(reference.dropna(), np.linspace(0, 1, bins + 1)))
    if len(e) < 3:
        return 0.0
    e[0], e[-1] = -np.inf, np.inf
    r = pd.cut(reference, e).value_counts(normalize=True).sort_index().clip(lower=1e-6)
    c = pd.cut(current, e).value_counts(normalize=True).sort_index().clip(lower=1e-6)
    return float(((c - r) * np.log(c / r)).sum())

cur = orders[orders.day_index.between(45, 51)]
print("PSI of the first monitored window (d45-51) against each candidate reference:")
for lo in [0, 15, 30]:
    ref_w = orders[orders.day_index.between(lo, 44)]
    print(f"  ref d{lo:>2}-44 : orders_30d PSI = {psi(ref_w.orders_30d, cur.orders_30d):.4f}"
          f"   device_age PSI = {psi(ref_w.device_age_days, cur.device_age_days):.4f}")
print("\nmean orders_30d by day (the warm-up ramp):",
      {d: round(orders.loc[orders.day_index == d, 'orders_30d'].mean(), 2) for d in [0, 10, 20, 30, 40, 50]})

# %%
drift_ref = orders[orders.day_index.between(*DRIFT_REF_DAYS)]
save_s3(drift_ref, "reference/drift_reference.parquet")
print(f"\ndrift reference: days {DRIFT_REF_DAYS[0]}-{DRIFT_REF_DAYS[1]}, {len(drift_ref):,} rows")
print("NOTE: this is refreshed on every promotion. A stale baseline measures drift against "
      "a model that is no longer deployed, and every alert becomes permanent.")

# %% [markdown]
# ---
# ### Notebook 06 findings
#
# | | |
# |---|---|
# | Feature set | declared **by name**, never by a numerical threshold |
# | Scaler | fit on 307,644 train rows only; leakage regression test passes |
# | Rebuild parity | recency/tenure/dormancy **exact**; frequency 99.37% (clock skew, documented) |
# | Temporal harness | day-30 fit / day-45 validate; day-60 refused (censored outcome) |
# | Drift reference | days **30-44** — days 0-44 gives a permanent false alarm at PSI 0.2585 |
# | Geography | excluded from the rebuild by design (§2.11 proxy risk) |
