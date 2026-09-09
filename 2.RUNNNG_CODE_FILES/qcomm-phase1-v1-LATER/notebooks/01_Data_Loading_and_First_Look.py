# %% [markdown]
# # Data Loading and First Look
#
# **Project:** QCOMM Customer Analytics — RFM segmentation + dormancy prediction
# **Phase 1, notebook 01.** Load both raw objects, emit the data contract, stamp the split, record the Phase-1 findings.
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

# %% [markdown]
# ## 1. Load the two raw objects
#
# `qcomm_customers_rfm.csv` is the **modeling table** — one row per customer, features
# already computed at the day-45 snapshot. `qcomm_orders.csv` is the **event stream** it
# was computed from. Both are needed: the customer table is the only source for 404,303
# of the 439,492 customers, and the orders file is the only thing that makes a *feature
# pipeline* (and Phase-3 drift) possible.

# %%
cust = read_s3("raw/qcomm_customers_rfm.csv")
print(f"customers: {cust.shape[0]:,} x {cust.shape[1]}")
cust.head(3)

# %%
orders = read_s3("raw/qcomm_orders.csv", low_memory=False)
for c in ["event_time", "ingestion_time", "label_available_at"]:
    orders[c] = pd.to_datetime(orders[c], format="ISO8601", utc=True)
orders["day_index"] = ((orders.event_time - DATA_START).dt.total_seconds() // 86400).astype("int16")
print(f"orders: {orders.shape[0]:,} x {orders.shape[1]}")
print(f"event_time: {orders.event_time.min()} -> {orders.event_time.max()}")

# %% [markdown]
# ## 2. Verify the raw objects before trusting them
#
# A key named `raw/` is not evidence that the contents match the documentation. The
# vendor `DATA_DICTIONARY.md` has five factual errors; each assertion below fails loudly
# if the file does not match what we measured.

# %% [markdown]
# ### D1 — the feature window is 45 days, not 90
#
# Data starts 2026-01-01, snapshot is 2026-02-15 = day 45. Both `tenure_days` and
# `recency_days` are hard-bounded by that gap, which is only possible with a 45-day window.

# %%
assert cust.tenure_days.max()  <= FEATURE_WINDOW_DAYS
assert cust.recency_days.max() <= FEATURE_WINDOW_DAYS
print(f"D1 OK  max tenure={cust.tenure_days.max():.2f}  max recency={cust.recency_days.max():.2f}"
      f"  -> window is {FEATURE_WINDOW_DAYS}d, NOT 90d")

# %% [markdown]
# ### D2 — the join predicate is `in_orders_sample == 1`
#
# The orders file holds 43,111 distinct customers, not the 35,189 documented. The extra
# 7,922 have their first order *after* the snapshot and therefore no RFM row. Joining on
# presence in the orders file silently turns every one of their features into NaN.

# %%
orphans = set(orders.customer_id) - set(cust.customer_id)
assert len(orphans) == 7922, len(orphans)
assert (orders[orders.customer_id.isin(orphans)].event_time <= SNAPSHOT).sum() == 0
flagged = set(cust.loc[cust.in_orders_sample == 1, "customer_id"])
assert len(flagged) == 35189 and not (flagged - set(orders.customer_id))
print(f"D2 OK  orders customers={orders.customer_id.nunique():,}  flagged={len(flagged):,}"
      f"  orphans={len(orphans):,} (all first-order-after-snapshot)")

# %% [markdown]
# ### D3 — `orders_next_45d` is the target under another name

# %%
assert ((cust.orders_next_45d == 0).astype(int) == cust.dormant_45d).all()
print("D3 OK  dormant_45d == (orders_next_45d == 0) on all rows -> identity leak, must drop")
print("       forbidden columns:", FORBIDDEN)

# %% [markdown]
# ## 3. Target leakage scan
#
# Every column is checked for a suspiciously strong univariate relationship with the
# target. Anything above ~0.95 AUC is a leak, not a feature.

# %%
from sklearn.metrics import roc_auc_score
y = cust.dormant_45d.to_numpy()
scan = []
for c in cust.columns:
    if c in ("customer_id", "segment_label", "dormant_45d"):
        continue
    try:
        auc = roc_auc_score(y, cust[c])
    except Exception:
        continue
    scan.append({"column": c, "auc": round(auc, 4), "abs_lift": round(abs(auc - 0.5), 4)})
scan = pd.DataFrame(scan).sort_values("abs_lift", ascending=False)
scan["verdict"] = np.where(scan.abs_lift > 0.45, "LEAK", np.where(scan.abs_lift > 0.05, "signal", "weak"))
print(scan.to_string(index=False))

# %% [markdown]
# `orders_next_45d` shows as a LEAK, as expected — it is the target. `recency_days` at
# ~0.77 is the genuine core RFM signal, not a leak: it is measured strictly before the
# snapshot and the outcome strictly after.

# %% [markdown]
# ## 4. Duplicates and nulls

# %%
print("customers: dup ids =", cust.customer_id.duplicated().sum(), "| nulls =", int(cust.isna().sum().sum()))
print("orders:    dup event_id =", orders.event_id.duplicated().sum())
print("\norders null rates:")
print((orders.isna().mean()[orders.isna().mean() > 0] * 100).round(2).to_string())

# %% [markdown]
# `days_since_prev_order` and `prev_late_delivery` are null on exactly 43,111 rows — one
# per customer, their first order. That is **structural**, not missing data, and gets an
# explicit flag in notebook 03 rather than an imputed value.

# %%
assert orders.days_since_prev_order.isna().sum() == orders.customer_id.nunique()
print("structural-null check OK")

# %% [markdown]
# ## 5. The order-level label is oracle for half the file
#
# `is_churn_45d` was reconstructed from the observable stream. It matches exactly where
# the 45-day forward window closes before the data cutoff, and disagrees on ~10% of rows
# where it does not — the generator saw a future this file does not contain.
#
# Consequence: with `AS_OF = data_end`, the trainable order-level set is **day 0-44 only**.
# The vendor's `label_available_at <= AS_OF` rule is load-bearing, not decoration.

# %%
o = orders.sort_values(["customer_id", "event_time"])
gap = (o.groupby("customer_id").event_time.shift(-1) - o.event_time).dt.total_seconds() / 86400
recon = ((gap.isna()) | (gap > 45)).astype(int)
censored = (o.event_time + pd.Timedelta(days=45)) > DATA_END
print(f"uncensored rows: {(~censored).sum():,}  agreement = {(recon[~censored] == o.is_churn_45d[~censored]).mean():.4f}")
print(f"censored rows:   { censored.sum():,}  agreement = {(recon[censored] == o.is_churn_45d[censored]).mean():.4f}")
ORDER_TRAIN_MAX_DAY = 44
print(f"\n-> order-level trainable set = day 0..{ORDER_TRAIN_MAX_DAY} "
      f"({(orders.day_index <= ORDER_TRAIN_MAX_DAY).sum():,} of {len(orders):,} rows)")

# %% [markdown]
# ## 6. Protected attributes and personal data (§2.11)
#
# **Finding, recorded rather than skipped.**
#
# * The customer table contains **no** gender, age, income, caste, religion or location.
#   Protected-attribute exclusion is therefore trivially satisfied for Model A and Model B.
# * The orders file contains `city_id`, `delivery_city`, `billing_city`. Geography is the
#   classic proxy for protected characteristics. **These are deliberately not propagated
#   into the rebuilt customer table** in notebook 06.
# * `customer_id`, `device_id`, `session_id` are pseudonymous identifiers, not direct
#   identifiers. No name, email, phone or address exists anywhere in the dataset.
# * The dataset is **synthetic**. No living person is described by any row. The applicable
#   regime (India DPDP Act) does not attach. The controls below are applied anyway, because
#   the production pipeline built on this shape will later carry real data.
# * Controls: SSE enabled on the bucket, Block Public Access on, retention to be set on
#   `predictions/` before Phase 3 writes to it.

# %%
DEMOGRAPHIC_COLS = ["gender", "age", "income", "occupation", "caste", "religion",
                    "date_of_birth", "email", "phone", "mobile", "address", "pincode"]
found = [c for c in cust.columns if any(d in c.lower() for d in DEMOGRAPHIC_COLS)]
assert not found, found
GEO_COLS_IN_ORDERS = [c for c in orders.columns if "city" in c.lower()]
print("demographic columns in customer table:", found or "NONE")
print("geo columns in orders (excluded from features):", GEO_COLS_IN_ORDERS)

# %% [markdown]
# ## 7. Emit the data contract
#
# One artefact, three consumers: training fails the run on violation, the serving API
# returns 4xx, and the drift job treats a schema change as a **different alert** from a
# distribution change. Without one file, the contract lives in three places and diverges.
#
# Categorical enums are derived from the **drift reference window (days 30-44)**, not the
# full file. Deriving `cuisine` from all 90 days would legalise the day-54 enum drift and
# the schema check could never fire on it.

# %%
def _col(name, s, required, note=None):
    d = {"name": name, "dtype": str(s.dtype), "nullable": bool(s.isna().any()),
         "null_rate": round(float(s.isna().mean()), 6), "required_at_inference": required}
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 25:
        d["range"] = {"min": float(np.nanmin(s)), "max": float(np.nanmax(s))}
    elif s.nunique() <= 25:
        d["allowed_values"] = sorted(map(str, s.dropna().unique()))
    if note:
        d["note"] = note
    return d

ref = orders[(orders.day_index >= DRIFT_REF_DAYS[0]) & (orders.day_index <= DRIFT_REF_DAYS[1])]
ENUM_FROM_REF = ["cuisine", "channel", "city_id", "delivery_city", "billing_city", "currency"]

contract = {
    "contract_version": "v1",
    "temporal": {
        "data_start": str(DATA_START.date()), "snapshot_date": str(SNAPSHOT.date()),
        "data_end": str(DATA_END.date()), "feature_window_days": FEATURE_WINDOW_DAYS,
        "correction": "Vendor dictionary states 90 days. Verified false: max(tenure_days) "
                      f"={cust.tenure_days.max():.2f} and max(recency_days)={cust.recency_days.max():.2f}."},
    "join_rules": {
        "customers_to_orders": "customers[in_orders_sample == 1] INNER JOIN orders ON customer_id",
        "verified_flagged": int(cust.in_orders_sample.sum()),
        "verified_orders_customers": int(orders.customer_id.nunique()),
        "verified_orphans": int(len(orphans))},
    "training_exclusions": {"target": "dormant_45d", "must_drop": FORBIDDEN,
                            "permitted_features": MODEL_B_FEATURES},
    "order_level_label_rule": {"rule": "orders[label_available_at <= AS_OF]",
                               "trainable_max_day": ORDER_TRAIN_MAX_DAY},
    "protected_attributes": {"demographics_present": False,
                             "geo_columns_excluded_from_features": GEO_COLS_IN_ORDERS,
                             "regime": "synthetic data; India DPDP Act does not attach",
                             "controls": ["SSE at rest", "Block Public Access",
                                          "retention on predictions/ before Phase 3"]},
    "reference_window_for_enums": {"days": list(DRIFT_REF_DAYS)},
    "tables": {
        "qcomm_customers_rfm": {"grain": "one row per customer", "primary_key": "customer_id",
            "rows": int(len(cust)),
            "columns": [_col(c, cust[c], c in MODEL_B_FEATURES) for c in cust.columns]},
        "qcomm_orders": {"grain": "one row per order", "primary_key": "event_id",
            "rows": int(len(orders)),
            "columns": [_col(c, ref[c] if c in ENUM_FROM_REF else orders[c], True)
                        for c in orders.columns if c != "day_index"]}},
}
save_json_s3(contract, "contracts/schema_v1.json")
print("cuisine enum from reference window:",
      [c for c in contract["tables"]["qcomm_orders"]["columns"] if c["name"] == "cuisine"][0]["allowed_values"])

# %% [markdown]
# ## 8. Stamp the split — once, here, and never again
#
# `__split` is written in notebook 01 and travels unchanged through every stage. Every
# downstream `.fit()` keys off `__split == "train"`. Re-splitting later would silently
# move rows across the leakage boundary.
#
# Stratified on `dormant_45d`. **A temporal split is impossible on this table** — it is a
# single snapshot, one date for every customer. The temporal harness is built in notebook
# 06 from the orders file (fit at day 30, validate at day 45) and is the only genuine
# temporal validation available.

# %%
from sklearn.model_selection import train_test_split
tr_idx, tmp = train_test_split(cust.index, test_size=0.30, random_state=SEED, stratify=cust.dormant_45d)
va_idx, te_idx = train_test_split(tmp, test_size=0.50, random_state=SEED,
                                  stratify=cust.loc[tmp, "dormant_45d"])
cust["__split"] = "train"
cust.loc[va_idx, "__split"] = "val"
cust.loc[te_idx, "__split"] = "test"
print(cust.__split.value_counts().to_string())
print("\ndormancy by split (must be near-identical):")
print(cust.groupby("__split").dormant_45d.mean().round(4).to_string())
assert cust.groupby("__split").dormant_45d.mean().std() < 0.001

# %% [markdown]
# ## 9. Save stage outputs

# %%
save_s3(cust, "data/customers/01_loaded.parquet")
save_s3(orders, "data/orders/01_loaded.parquet")

# %% [markdown]
# ---
# ### Notebook 01 findings
#
# | | |
# |---|---|
# | D1 feature window | **45 days**, not 90 — asserted |
# | D2 join predicate | `in_orders_sample == 1`; 7,922 orphans confirmed post-snapshot only |
# | D3 identity leak | `dormant_45d == (orders_next_45d == 0)` on all rows |
# | Order-label rule | trainable set is day 0-44; 48.9% of rows carry an oracle label |
# | Protected attributes | none present; geo deliberately excluded from features |
# | Split | 70/15/15 stratified, stamped once, dormancy identical across splits |
#
# **Next:** 02 cleaning. Chain A is already clean; chain B is where the real work is.
