"""Build the committed test fixture using THE PRODUCTION CODE PATH.

Part 0 #15. A fixture fitted differently from production certifies the wrong thing — in a
previous project a conftest that fitted on int64 while production used category dtype
produced 34 green tests and one live HTTP 500.

So: the fixture model is fitted by src.preprocess.transform() with the same dtypes and the
same scaler constants as the promoted artifact, and asserts the structural markers.

Regenerate from the real raw object:
    python tests/_make_fixture.py --source /path/to/qcomm_customers_rfm.csv
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess import FEATURE_DTYPES, FEATURES, align, transform  # noqa: E402

HERE = Path(__file__).resolve().parent
FIXTURE_ROWS = 4000          # large enough that k=4 behaves like production, not a toy
SEED = 42

# From the promoted manifest v_20260909T1228Z_nogit. The fixture uses the PRODUCTION
# scaler constants so the fixture model lives in the same scaled space as the real one.
SCALER_MEAN = [2.1396892482201015, 1.6204866490670535, 6.20592506637903]
SCALER_SCALE = [1.0873314965351688, 0.7438759245473856, 0.552499359350201]
SEGMENT_NAMES = {3: "Champions", 2: "Steady Regulars",
                 0: "High-Value At-Risk", 1: "Lost / Dormant"}
PROD_LIBS = {"numpy": "2.1.3", "pandas": "2.2.3",
             "scikit-learn": "1.6.1", "scipy": "1.16.3"}


def synthesise(n=FIXTURE_ROWS, seed=SEED) -> pd.DataFrame:
    """Stratified stand-in when the real extract is not on disk.

    Shapes are taken from the measured Phase-1 distributions so the fixture model is not a
    toy: recency right-skewed on [0,45], frequency Poisson-ish >=1, monetary lognormal.
    """
    rng = np.random.default_rng(seed)
    recency = np.clip(rng.gamma(1.6, 8.0, n), 0, 45)
    frequency = np.clip(rng.poisson(5.2, n), 1, 106)
    monetary = np.clip(rng.lognormal(6.2, 0.55, n), 195, 66106)
    return pd.DataFrame({"recency_days": recency.astype("float64"),
                         "frequency": frequency.astype("int64"),
                         "monetary_mean": monetary.astype("float64")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="path to qcomm_customers_rfm.csv; synthesised if absent")
    args = ap.parse_args()

    if args.source and Path(args.source).exists():
        raw = pd.read_csv(args.source, usecols=list(FEATURES))
        df = raw.sample(FIXTURE_ROWS, random_state=SEED).reset_index(drop=True)
        provenance = f"stratified sample of {args.source}"
    else:
        df = synthesise()
        provenance = "synthesised from measured Phase-1 distributions"

    df = align(df)
    for c, want in FEATURE_DTYPES.items():
        assert str(df[c].dtype) == want, (c, df[c].dtype, want)

    # PRODUCTION CODE PATH — same transform, same constants.
    Z = transform(df, SCALER_MEAN, SCALER_SCALE)
    model = KMeans(4, n_init=10, random_state=SEED).fit(Z)

    # Structural markers that prove the fixture matches production shape.
    assert model.cluster_centers_.shape == (4, len(FEATURES))
    assert model.n_features_in_ == len(FEATURES)

    df.to_csv(HERE / "fixtures" / "sample_customers.csv", index=False)
    (HERE / "fixtures" / "fixture_model.pkl").write_bytes(pickle.dumps(model))
    (HERE / "fixtures" / "fixture_manifest.json").write_text(json.dumps({
        "contract_version": "v1",
        "feature_order_hash": "a50fb3d15315ec03",
        # PRODUCTION versions, from the promoted manifest — not this machine's.
        # The fixture manifest stands in for the real one, so it must satisfy the same
        # check. The build environment is recorded separately, for provenance only.
        "library_versions": PROD_LIBS,
        "feature_dtypes": FEATURE_DTYPES,
        "model": {"algo": "KMeans", "k": 4, "features": list(FEATURES),
                  "transform": "log1p -> StandardScaler",
                  "scaler_mean": SCALER_MEAN, "scaler_scale": SCALER_SCALE,
                  "segment_names": {str(k): v for k, v in SEGMENT_NAMES.items()}},
        "provenance": provenance, "rows": len(df), "seed": SEED,
        "built_with": {"numpy": np.__version__, "pandas": pd.__version__,
                       "scikit-learn": __import__("sklearn").__version__,
                       "scipy": __import__("scipy").__version__},
    }, indent=2))
    print(f"fixture written: {len(df)} rows, {provenance}")
    print(f"  cluster sizes: {np.bincount(model.labels_).tolist()}")


if __name__ == "__main__":
    main()
