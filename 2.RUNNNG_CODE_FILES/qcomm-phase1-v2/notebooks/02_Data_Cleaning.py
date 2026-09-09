# %% [markdown]
# # Data Cleaning
#
# **Project:** QCOMM Customer Analytics — RFM customer segmentation
# **Phase 1, notebook 02.** Chain A passes through with assertions. Chain B normalises the enum drift and flags clock skew.
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
cust   = read_s3("data/customers/01_loaded.parquet")
orders = read_s3("data/orders/01_loaded.parquet")
print(f"customers {len(cust):,} | orders {len(orders):,}")

# %% [markdown]
# ## Chain A — assert, do not clean
#
# The customer table has no nulls, no duplicate keys and no impossible values. There is
# nothing to clean. The right response is an assertion that fails if that ever changes,
# not a no-op cleaning function that hides a future regression.

# %%
assert cust.isna().sum().sum() == 0
assert not cust.customer_id.duplicated().any()
assert (cust.recency_days >= 0).all() and (cust.recency_days <= FEATURE_WINDOW_DAYS).all()
assert (cust.frequency >= 1).all()
assert (cust.monetary_mean > 0).all() and (cust.monetary_sum > 0).all()
assert cust.promo_rate.between(0, 1).all() and cust.mobile_rate.between(0, 1).all()
print("chain A: all invariants hold, passing through unchanged")
save_s3(cust, "data/customers/02_cleaned.parquet")

# %% [markdown]
# ## Chain B — enum normalisation
#
# **Order matters: normalise sentinels -> apply fixed maps -> impute.** `.map()` and
# categorical construction return NaN for every unmapped value. Normalising *after*
# imputation leaves those NaNs unfillable, and a downstream scaler computes statistics
# ignoring NaN then propagates it — the result looks plausible and is wrong.
#
# The drift itself is real: from day 54 the upstream system emits `North Indian` where it
# previously emitted `north_indian`. We normalise for modelling **and** record the raw
# value, because the *count* of violations is a Phase-3 schema alert with a different
# owner from a distribution shift.

# %%
REF_ENUM = sorted(orders.loc[orders.day_index.between(*DRIFT_REF_DAYS), "cuisine"].unique())
print(f"reference enum (days {DRIFT_REF_DAYS[0]}-{DRIFT_REF_DAYS[1]}): {len(REF_ENUM)} values")

orders["cuisine_raw"] = orders.cuisine
orders["schema_violation"] = (~orders.cuisine.isin(REF_ENUM)).astype("int8")
orders["cuisine"] = orders.cuisine.str.lower().str.replace(" ", "_", regex=False)

unmapped = set(orders.cuisine.unique()) - set(REF_ENUM)
assert not unmapped, f"normalisation incomplete: {unmapped}"
print(f"violations normalised: {orders.schema_violation.sum():,} rows "
      f"({orders.schema_violation.mean():.4%})")
first = int(orders.loc[orders.schema_violation == 1, "day_index"].min())
print(f"first violation on day {first}  (vendor dictionary says 'after day 55')")

# %% [markdown]
# ### Fixed categorical maps
#
# Categories come from the **reference enum**, not from whatever happens to be present in
# the current frame. A batch covering 3 of 12 cuisines must still carry all 12 categories,
# or the integer codes shift and every prediction is quietly wrong.

# %%
CUISINE_CATS = REF_ENUM
CHANNEL_CATS = sorted(orders.loc[orders.day_index.between(*DRIFT_REF_DAYS), "channel"].unique())
orders["cuisine"] = pd.Categorical(orders.cuisine, categories=CUISINE_CATS)
orders["channel"] = pd.Categorical(orders.channel, categories=CHANNEL_CATS)
assert orders.cuisine.isna().sum() == 0 and orders.channel.isna().sum() == 0
print("cuisine cats:", len(CUISINE_CATS), "| channel cats:", CHANNEL_CATS)

# %% [markdown]
# ## Chain B — device clock skew
#
# `event_time` is device-reported; `ingestion_time` is server-side. A negative lag means
# the device clock was ahead. **Flag it, do not silently correct it** — the feature values
# in the vendor's customer table were computed from true time, so this is exactly the
# ~0.6% of customers whose rebuilt `frequency` will differ by one order in notebook 06.
# Correcting it here would hide that, not fix it.

# %%
lag = (orders.ingestion_time - orders.event_time).dt.total_seconds()
orders["clock_skew"] = (lag < 0).astype("int8")
print(f"clock-skew rows: {orders.clock_skew.sum():,} ({orders.clock_skew.mean():.4%})")
print(f"ingestion lag seconds: median={lag.median():.1f}  p99={lag.quantile(0.99):.1f}  max={lag.max():.0f}")
assert lag.min() >= -900, "skew exceeds the documented 900s bound"

# %% [markdown]
# ## Drop nothing, derive nothing yet

# %%
assert len(orders) == 397436, len(orders)
save_s3(orders.drop(columns=["cuisine_raw"]), "data/orders/02_cleaned.parquet")

# %% [markdown]
# ---
# ### Notebook 02 findings
#
# | | |
# |---|---|
# | Chain A | clean; passed through with 6 invariant assertions |
# | Enum drift | 497 rows normalised, first on **day 54** (dictionary says 55) |
# | Categorical cats | fixed from the reference window, not the current frame |
# | Clock skew | 661 rows flagged (0.17%), not corrected — it explains the 06 parity gap |
# | Rows dropped | **zero** |
