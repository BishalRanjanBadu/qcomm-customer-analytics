"""Model registry: alias resolution, manifest validation, artifact loading.

Promotion writes models/CURRENT.json. Rollback rewrites it to a prior prefix. No artifact
is ever overwritten, so both directions are reversible.

The manifest check runs at STARTUP and fails readiness on mismatch. A bad promotion then
fails readiness, the previous pods keep serving, and nothing rolls forward.
"""
from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass, field

import numpy as np

from . import s3_io

# Verified against the manifest of v_20260909T1228Z_nogit. These are the versions that
# produced the promoted pickle, NOT the latest on PyPI.
EXPECTED_LIBS = {
    "numpy": "2.1.3",
    "pandas": "2.2.3",
    "scikit-learn": "1.6.1",
    "scipy": "1.16.3",
}
EXPECTED_CONTRACT_VERSION = "v1"
EXPECTED_FEATURE_ORDER_HASH = "a50fb3d15315ec03"


class ManifestMismatch(RuntimeError):
    """Serving container refuses readiness. Old pods keep serving."""


@dataclass
class Artifact:
    version: str
    prefix: str
    model: object
    features: tuple[str, ...]
    dtypes: dict
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    segment_names: dict
    contract_version: str
    manifest: dict = field(repr=False, default_factory=dict)


def _installed_versions() -> dict:
    import pandas
    import scipy
    import sklearn
    return {"numpy": np.__version__, "pandas": pandas.__version__,
            "scikit-learn": sklearn.__version__, "scipy": scipy.__version__}


def validate_manifest(manifest: dict, *, strict_libs: bool = True) -> list[str]:
    """Return the list of mismatches. Empty means the artifact is safe to serve."""
    problems: list[str] = []

    if manifest.get("contract_version") != EXPECTED_CONTRACT_VERSION:
        problems.append(
            f"contract_version {manifest.get('contract_version')} != {EXPECTED_CONTRACT_VERSION}")

    if manifest.get("feature_order_hash") != EXPECTED_FEATURE_ORDER_HASH:
        problems.append(
            f"feature_order_hash {manifest.get('feature_order_hash')} "
            f"!= {EXPECTED_FEATURE_ORDER_HASH} — the feature set changed")

    baked = manifest.get("library_versions", {})
    for lib, expected in EXPECTED_LIBS.items():
        if baked.get(lib) != expected:
            problems.append(f"manifest {lib}=={baked.get(lib)} but this image expects {expected}")

    if strict_libs:
        for lib, installed in _installed_versions().items():
            want = baked.get(lib)
            if want and installed != want:
                problems.append(
                    f"{lib} installed {installed} but the artifact was trained on {want} — "
                    "a cross-version unpickle can appear to succeed and silently corrupt "
                    "fitted state")

    dt = manifest.get("feature_dtypes", {})
    from .preprocess import FEATURE_DTYPES
    for c, want in FEATURE_DTYPES.items():
        if dt.get(c) != want:
            problems.append(f"dtype for {c}: manifest says {dt.get(c)}, image expects {want}")

    return problems


def load_current(*, strict_libs: bool = True) -> Artifact:
    """Resolve the alias, validate, load. Raises ManifestMismatch rather than serving."""
    current = s3_io.get_json("models/CURRENT.json")
    prefix = current["prefix"]
    manifest = s3_io.get_json(f"{prefix}/manifest.json")

    problems = validate_manifest(manifest, strict_libs=strict_libs)
    if problems:
        raise ManifestMismatch(
            f"artifact {current['version']} rejected at load:\n  - " + "\n  - ".join(problems))

    model = pickle.loads(s3_io.get_bytes(f"{prefix}/kmeans.pkl"))
    m = manifest["model"]
    return Artifact(
        version=current["version"], prefix=prefix, model=model,
        features=tuple(m["features"]), dtypes=manifest["feature_dtypes"],
        scaler_mean=np.asarray(m["scaler_mean"], dtype="float64"),
        scaler_scale=np.asarray(m["scaler_scale"], dtype="float64"),
        segment_names={int(k): v for k, v in m["segment_names"].items()},
        contract_version=manifest["contract_version"], manifest=manifest,
    )
