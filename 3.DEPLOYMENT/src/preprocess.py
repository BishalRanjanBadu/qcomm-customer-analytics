"""The one transform implementation. Training and serving both call it.

Two functions, deliberately separate:

  clean_for_training()  may filter rows, may see the validation outcome
  transform()           row-preserving, outcome-free, used by BOTH paths

A clean() that reads the outcome column will KeyError at inference. One that drops rows
silently returns fewer predictions than requests. Neither failure is loud.

The scaler is reconstructed from the manifest's mean/scale rather than unpickled. That is
deliberate: StandardScaler is (x - mean) / scale, so reconstructing removes an entire
cross-version-unpickle failure mode for zero loss of fidelity. test_parity.py pins the
reconstruction against sklearn's own transform to 1e-12.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Feature order is load-bearing. KMeans centroids are positional.
FEATURES: tuple[str, ...] = ("recency_days", "frequency", "monetary_mean")

# From the promoted manifest. Restored by align() before every predict.
FEATURE_DTYPES: dict[str, str] = {
    "recency_days": "float64",
    "frequency": "int64",
    "monetary_mean": "float64",
}


class ContractViolation(ValueError):
    """Input does not satisfy contracts/schema_v1.json. Serving returns 4xx, never a prediction."""


def align(df: pd.DataFrame, feature_order=FEATURES, dtypes=FEATURE_DTYPES) -> pd.DataFrame:
    """Reindex to the trained feature order and restore the trained dtypes.

    Order mismatch silently corrupts predictions — the centroid comparison is positional,
    so a swapped column produces a confident wrong segment with HTTP 200.
    Dtype mismatch raises inside the estimator and surfaces as an opaque 500.
    """
    missing = [c for c in feature_order if c not in df.columns]
    if missing:
        raise ContractViolation(f"missing required features: {missing}")

    out = df.reindex(columns=list(feature_order))
    for c in feature_order:
        want = dtypes[c]
        if str(out[c].dtype) == want:
            continue
        if want.startswith("int") and out[c].isna().any():
            raise ContractViolation(
                f"{c} is null but the trained dtype is {want}; nulls are rejected, not imputed"
            )
        try:
            out[c] = out[c].astype(want)
        except (ValueError, TypeError) as e:
            raise ContractViolation(f"{c} cannot be cast to the trained dtype {want}: {e}") from e
    return out


def transform(df: pd.DataFrame, scaler_mean, scaler_scale) -> np.ndarray:
    """log1p then standardise. Row-preserving, outcome-free.

    Identical to the notebook-06 path: np.log1p over the three RFM columns, then
    StandardScaler fitted on training rows only.
    """
    X = align(df)
    if (X.to_numpy() < 0).any():
        bad = [c for c in FEATURES if (X[c] < 0).any()]
        raise ContractViolation(f"negative values in {bad}; log1p is undefined below -1")
    logged = np.log1p(X.to_numpy(dtype="float64"))
    return (logged - np.asarray(scaler_mean)) / np.asarray(scaler_scale)


def clean_for_training(df: pd.DataFrame, split_col: str = "__split") -> pd.DataFrame:
    """Training-only path. May filter rows. Never called at serving time."""
    if split_col not in df.columns:
        raise ContractViolation(f"{split_col} absent; the split is stamped once in notebook 01")
    out = df.dropna(subset=list(FEATURES))
    return out
