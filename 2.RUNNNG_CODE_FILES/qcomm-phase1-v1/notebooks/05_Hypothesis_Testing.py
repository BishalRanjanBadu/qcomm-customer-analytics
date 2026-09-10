# %% [markdown]
# # Hypothesis Testing
#
# **Project:** QCOMM Customer Analytics — RFM customer segmentation
# **Phase 1, notebook 05.** Effect sizes, not p-values. At n=307k every p is ~0 and tells you nothing.
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
from scipy import stats
cust = read_s3("data/customers/04_eda.parquet")
d = cust.loc[cust.__split == "train"]
print(f"tests on TRAIN only: {len(d):,} rows")

# %% [markdown]
# **What these tests are for.** There is no supervised model in this project, so these are
# not feature-selection tests. They establish how strongly each column relates to
# `dormant_45d` — the variable that will *referee* the segmentation in notebook 07 — so
# that the gate result there can be read in context rather than taken on faith.
#
# ## Why p-values are useless here
#
# With 307,644 rows, a difference of no practical size returns p ≈ 0. Testing 7 features
# at α=0.05 also yields false positives by chance. **The decision rule is the effect
# size**; p is reported only to show it carries no information.
#
# Rank-biserial `r` is used because the distributions are heavily skewed and non-normal,
# so a t-test's assumptions do not hold. `|r| > 0.05` is the retention threshold.

# %%
rows = []
for c in PROFILE_COLS:
    g0 = d.loc[d.dormant_45d == 0, c]
    g1 = d.loc[d.dormant_45d == 1, c]
    u, p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
    r = 1 - 2 * u / (len(g0) * len(g1))
    rows.append({"feature": c, "median_active": round(g0.median(), 2),
                 "median_dormant": round(g1.median(), 2), "rank_biserial_r": round(r, 4),
                 "p_value": f"{p:.2e}", "verdict": "KEEP" if abs(r) > 0.05 else "weak"})
res = pd.DataFrame(rows).reindex(pd.DataFrame(rows).rank_biserial_r.abs().sort_values(ascending=False).index)
print(res.to_string(index=False))

# %% [markdown]
# Every p-value is effectively zero, including for `promo_rate` whose effect size is
# `r = +0.0005` — a feature with no relationship whatsoever. That is the entire argument
# against p-value gating in one line.

# %% [markdown]
# ## Does the reference segmentation separate the outcome?
#
# Kruskal-Wallis across the four supplied segments, plus the thing that actually matters:
# the dormancy rate itself.

# %%
groups = [g.dormant_45d.values for _, g in d.groupby("segment_label")]
h, p = stats.kruskal(*groups)
print(f"Kruskal-Wallis H={h:,.1f} p={p:.2e}")
tbl = d.groupby("segment_label").agg(n=("dormant_45d", "size"), dormancy=("dormant_45d", "mean"),
                                     med_R=("recency_days", "median"), med_F=("frequency", "median"),
                                     med_M=("monetary_mean", "median")).sort_values("dormancy")
print("\n" + tbl.round(4).to_string())
print(f"\nspread = {tbl.dormancy.iloc[-1] / tbl.dormancy.iloc[0]:.2f}x")

# %% [markdown]
# **The H statistic is not the evidence.** A Kruskal-Wallis across clusters built from
# these same features is circular — it would return p ≈ 0 on randomly shuffled data too.
# The evidence is the **spread**, tested against a permutation null in notebook 07.

# %% [markdown]
# ## Channel and promo behaviour

# %%
d2 = d.assign(mobile_bucket=pd.cut(d.mobile_rate, [-0.01, 0.25, 0.75, 1.0],
                                   labels=["low", "mixed", "high"]),
              promo_bucket=pd.cut(d.promo_rate, [-0.01, 0.2, 0.5, 1.0],
                                  labels=["rare", "some", "heavy"]))
for col in ["mobile_bucket", "promo_bucket"]:
    ct = pd.crosstab(d2[col], d2.dormant_45d)
    chi2, p, _, _ = stats.chi2_contingency(ct)
    v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    print(f"{col}: Cramer's V={v:.4f}  p={p:.2e}  "
          f"{'usable' if v > 0.05 else 'negligible despite p~0'}")
    print(d2.groupby(col, observed=True).dormant_45d.mean().round(4).to_string(), "\n")

# %%
save_s3(cust, "data/customers/05_hypothesis.parquet")

# %% [markdown]
# ---
# ### Notebook 05 findings
#
# | | |
# |---|---|
# | Retained (\|r\| > 0.05) | `recency_days` +0.53, `frequency` -0.37, `monetary_sum` -0.35, `monetary_mean` -0.056, `mobile_rate` +0.053 |
# | Weak | `tenure_days` +0.013, `promo_rate` +0.0005 — profile columns only, never clustering inputs |
# | p-values | all ~0 including for a null feature; **not used as a decision rule** |
# | Reference segments | spread confirmed; the H statistic is circular and is not the evidence |
