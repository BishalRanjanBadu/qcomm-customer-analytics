"""Scoring. Wraps the estimator call and re-raises with the cause named.

An unhandled ValueError from .predict() reaches the caller as an opaque HTTP 500 with
/health still green. The wrapper turns that into a message naming the exception and the
observed dtypes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .preprocess import FEATURES, align, transform
from .registry import Artifact


class PredictionError(RuntimeError):
    pass


def score(df: pd.DataFrame, art: Artifact) -> pd.DataFrame:
    """Row-preserving. len(out) == len(df), always."""
    n_in = len(df)
    Z = transform(df, art.scaler_mean, art.scaler_scale)
    try:
        labels = art.model.predict(Z)
    except Exception as e:
        observed = {c: str(align(df)[c].dtype) for c in FEATURES}
        raise PredictionError(
            f"{type(e).__name__} during predict: {e}. observed dtypes={observed}, "
            f"trained dtypes={art.dtypes}") from e

    centroids = art.model.cluster_centers_
    dist = np.linalg.norm(Z - centroids[labels], axis=1)

    out = pd.DataFrame({
        "segment_id": labels.astype(int),
        "segment_name": [art.segment_names[int(i)] for i in labels],
        "distance_to_centroid": np.round(dist, 6),
    }, index=df.index)

    if len(out) != n_in:
        raise PredictionError(f"row count changed: {n_in} in, {len(out)} out")
    return out
