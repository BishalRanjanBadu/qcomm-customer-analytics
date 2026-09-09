# %% [markdown]
# # Missing Values and Outliers
#
# **Project:** QCOMM Customer Analytics — RFM segmentation + dormancy prediction
# **Phase 1, notebook 03.** MNAR imputation on the order stream, outlier caps fit on training rows only.
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
cust   = read_s3("data/customers/02_cleaned.parquet")
orders = read_s3("data/orders/02_cleaned.parquet")

# %% [markdown]
# ## Chain B — the leakage boundary for the order stream
#
# For the order stream the boundary is the **label-availability rule**, not a random
# split: only orders whose 45-day forward window closes before the data cutoff have a
# label derivable from this file. Every statistic below is fit on day 0-44 rows only.

# %%
ORDER_TRAIN = orders.day_index <= 44
print(f"fit rows (day 0-44): {ORDER_TRAIN.sum():,} of {len(orders):,} ({ORDER_TRAIN.mean():.1%})")

# %% [markdown]
# ## MNAR — `order_value`
#
# Missingness is **informative**: blank rates rise monotonically across value deciles
# (0.16% in the bottom, 1.18% in the top). Mean- or median-imputation is therefore biased
# *downward* by construction, because the values that go missing are the large ones.
#
# The correct handling is not a cleverer imputer — it is to **carry the missingness as a
# feature** so a model can learn the bias rather than absorb it silently.

# %%
miss_by_decile = (orders.assign(d=pd.qcut(orders.groupby("customer_id").order_value.transform("median"),
                                          10, labels=False, duplicates="drop"))
                        .groupby("d").order_value.apply(lambda s: s.isna().mean()))
print("blank rate by customer-median decile:")
print((miss_by_decile * 100).round(3).to_string())
print(f"top/bottom ratio: {miss_by_decile.iloc[-1] / miss_by_decile.iloc[0]:.1f}x  -> MNAR confirmed")

# %%
ov_median_train = float(orders.loc[ORDER_TRAIN, "order_value"].median())
orders["order_value_missing"] = orders.order_value.isna().astype("int8")
orders["order_value"] = orders.order_value.fillna(ov_median_train)
print(f"observed mean = {orders.loc[orders.order_value_missing == 0, 'order_value'].mean():.2f}")
print(f"fill value    = {ov_median_train:.2f}   <- deliberately below the observed mean")
print("   the indicator column is what lets a model recover the bias")

# %% [markdown]
# ## MAR — `device_age_days`
#
# Missingness depends on an observed column: 4.15% on web, 4.09% on mweb, 0.40% on app.
# Channel-wise median, fit on training rows only.

# %%
print(orders.groupby("channel", observed=True).device_age_days
      .apply(lambda s: s.isna().mean() * 100).round(2).to_string())
dev_median = orders.loc[ORDER_TRAIN].groupby("channel", observed=True).device_age_days.median()
orders["device_age_missing"] = orders.device_age_days.isna().astype("int8")
orders["device_age_days"] = orders.device_age_days.fillna(orders.channel.map(dev_median))
print("\nchannel-wise fill (train only):", dev_median.round(1).to_dict())

# %% [markdown]
# ## MCAR — `items_n`
#
# 0.58-0.60% on every channel. Uniform, so a single median is defensible.

# %%
orders["items_n"] = orders.items_n.fillna(float(orders.loc[ORDER_TRAIN, "items_n"].median()))

# %% [markdown]
# ## Structural nulls — not missing data
#
# `days_since_prev_order` and `prev_late_delivery` are null on exactly one row per
# customer: their first order. There is no prior order, so there is no value to impute.
# Imputing a median here would invent a history that did not happen. Explicit sentinel
# plus a flag.

# %%
orders["is_first_order"] = orders.days_since_prev_order.isna().astype("int8")
assert orders.is_first_order.sum() == orders.customer_id.nunique()
orders["days_since_prev_order"] = orders.days_since_prev_order.fillna(-1.0)
orders["prev_late_delivery"] = orders.prev_late_delivery.fillna(-1.0)
print(f"first orders flagged: {orders.is_first_order.sum():,}")

# %% [markdown]
# ## Assert NaN-free
#
# This converts a silent accuracy leak into a loud failure.

# %%
remaining = orders.isna().sum()
remaining = remaining[remaining > 0]
assert remaining.empty, f"NaN survived in: {remaining.to_dict()}"
print("chain B: NaN-free")
save_s3(orders, "data/orders/03_imputed.parquet")

# %% [markdown]
# ## Chain A — outliers are the business, not noise
#
# The top spenders are the segment the whole project exists to find. **Dropping them
# would delete the answer.** Caps are computed at the training p99.5 and stored as
# *parallel* columns so the uncapped values remain available; the capped versions exist
# only to stop a single ₹66,106 mean order value from dominating a distance metric.

# %%
train = cust.__split == "train"
CAP_Q = 0.995
caps = {c: float(cust.loc[train, c].quantile(CAP_Q))
        for c in ["monetary_mean", "monetary_sum", "frequency"]}
for c, v in caps.items():
    cust[c + "_capped"] = cust[c].clip(upper=v)
    print(f"{c:15s} cap={v:>10.1f}  rows above={int((cust[c] > v).sum()):>5,} "
          f"({(cust[c] > v).mean():.2%})  max was {cust[c].max():.1f}")
print("\ncaps fit on TRAIN rows only, applied to the whole frame")

# %%
assert len(cust) == 439492
save_s3(cust, "data/customers/03_imputed.parquet")
save_json_s3({"order_value_median_train": ov_median_train,
              "device_age_median_by_channel": {str(k): float(v) for k, v in dev_median.items()},
              "cap_quantile": CAP_Q, "caps": caps,
              "fit_rows": "customers: __split=='train'; orders: day_index<=44"},
             "contracts/imputation_params_v1.json")

# %% [markdown]
# ---
# ### Notebook 03 findings
#
# | | |
# |---|---|
# | `order_value` | MNAR confirmed (7.2x top/bottom decile); indicator column carries the bias |
# | `device_age_days` | MAR by channel; channel-wise train median |
# | `items_n` | MCAR; single train median |
# | First-order nulls | structural — sentinel `-1` + `is_first_order` flag, never imputed |
# | Chain A outliers | **capped, never dropped** — high spenders are the target segment |
# | Rows dropped | **zero, both chains** |
