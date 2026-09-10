# %% [markdown]
# # Statistics and EDA
#
# **Project:** QCOMM Customer Analytics — RFM segmentation + dormancy prediction
# **Phase 1, notebook 04.** Exploratory analysis on TRAIN ROWS ONLY. Anything seen here influences downstream choices.
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
import matplotlib.pyplot as plt
cust = read_s3("data/customers/03_imputed.parquet")
train = cust.__split == "train"
d = cust.loc[train]
print(f"EDA on TRAIN only: {len(d):,} of {len(cust):,} rows")

# %% [markdown]
# **Why train-only matters here.** EDA is not passive. Anything you look at influences the
# transform you choose, the features you keep and the model family you pick. Looking at
# validation or test rows makes you the leakage vector — no `.fit()` required.

# %%
FEATS = MODEL_B_FEATURES
print(d[FEATS].describe().T.round(3).to_string())

# %% [markdown]
# ## Skew — the case for `log1p`

# %%
sk = d[FEATS].skew().sort_values(ascending=False)
print(sk.round(2).to_string())
print("\nlog1p skew:")
print(np.log1p(d[["frequency", "monetary_mean", "monetary_sum", "recency_days"]]).skew().round(2).to_string())

# %% [markdown]
# `monetary_mean` at skew ~31 is driven by a thin tail reaching ₹66,106. Untransformed it
# would dominate any Euclidean distance and the clustering would degenerate into
# "one giant cluster plus a handful of outliers". `frequency` at 2.6 and `monetary_sum` at
# 4.1 are the same problem in milder form. `recency_days` at ~1.0 is only mildly skewed
# but is transformed with the others for consistency — one transform, applied uniformly,
# is easier to keep symmetric between train and serve than three special cases.

# %%
fig, ax = plt.subplots(2, 3, figsize=(14, 7))
for i, c in enumerate(RFM_COLS):
    ax[0, i].hist(d[c], bins=60, color="#4C72B0")
    ax[0, i].set_title(f"{c} (raw, skew={d[c].skew():.1f})")
    ax[1, i].hist(np.log1p(d[c]), bins=60, color="#55A868")
    ax[1, i].set_title(f"log1p({c}), skew={np.log1p(d[c]).skew():.1f}")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Dormancy against each feature
#
# This is the relationship the segmentation has to reproduce. Recency is the dominant
# axis; monetary is nearly flat.

# %%
from sklearn.metrics import roc_auc_score
rows = []
for c in FEATS:
    a = roc_auc_score(d.dormant_45d, d[c])
    rows.append({"feature": c, "auc": round(a, 4), "oriented_auc": round(max(a, 1 - a), 4),
                 "direction": "higher -> more dormant" if a > 0.5 else "higher -> less dormant"})
print(pd.DataFrame(rows).sort_values("oriented_auc", ascending=False).to_string(index=False))

# %%
fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
for i, c in enumerate(RFM_COLS):
    g = d.groupby(pd.qcut(d[c], 10, labels=False, duplicates="drop")).dormant_45d.mean()
    ax[i].plot(g.index, g.values, marker="o", color="#C44E52")
    ax[i].set_title(f"dormancy by {c} decile"); ax[i].set_xlabel("decile"); ax[i].set_ylim(0, 0.75)
plt.tight_layout(); plt.show()

# %% [markdown]
# `monetary_mean` is visibly flat — it will contribute almost nothing to dormancy
# prediction (AUC ~0.47). It stays in the RFM triple because RFM is the stated framing and
# M genuinely separates *segments* by value, but its weakness is recorded here so nobody
# later claims the model "found high-value churners" on the strength of M.

# %% [markdown]
# ## Correlation among the RFM axes

# %%
print(d[FEATS].corr(method="spearman").round(3).to_string())

# %% [markdown]
# `monetary_sum` correlates strongly with `frequency` by construction (sum = mean x count).
# That redundancy is acceptable for Model B (gradient boosting is untroubled by it) but
# `monetary_sum` is deliberately **excluded** from the Model A clustering triple, which
# uses R, F and *mean* — otherwise F would be counted twice in the distance metric.

# %%
save_s3(cust, "data/customers/04_eda.parquet")

# %% [markdown]
# ---
# ### Notebook 04 findings
#
# | | |
# |---|---|
# | Skew | `monetary_mean` 31.0, `monetary_sum` 4.1, `frequency` 2.6 -> `log1p` before scaling |
# | Dominant axis | `recency_days`, oriented AUC ~0.77 |
# | Weakest axis | `monetary_mean`, AUC ~0.47 — recorded so it is not oversold later |
# | Clustering triple | R, F, **mean** M — `monetary_sum` excluded to avoid double-counting F |
