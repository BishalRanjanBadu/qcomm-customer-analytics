"""Train/serve parity. The transform must be one implementation, provably."""
import numpy as np, pandas as pd, pytest
from sklearn.preprocessing import StandardScaler
from src.preprocess import FEATURES, FEATURE_DTYPES, align, transform, ContractViolation


def test_reconstructed_scaler_equals_sklearn(sample, artifact):
    """The manifest constants reproduce sklearn's own transform to 1e-12.

    This is what licenses NOT unpickling a StandardScaler, which removes an entire
    cross-version failure mode.
    """
    logged = np.log1p(align(sample).to_numpy(dtype="float64"))
    sk = StandardScaler()
    sk.mean_, sk.scale_ = artifact.scaler_mean, artifact.scaler_scale
    sk.n_features_in_, sk.var_ = 3, artifact.scaler_scale ** 2
    assert np.abs(sk.transform(logged) - transform(sample, artifact.scaler_mean,
                                                   artifact.scaler_scale)).max() < 1e-12


def test_align_restores_dtypes(sample):
    wrong = sample.copy()
    wrong["frequency"] = wrong["frequency"].astype("float64")
    wrong["recency_days"] = wrong["recency_days"].astype("float32")
    out = align(wrong)
    assert {c: str(out[c].dtype) for c in FEATURES} == FEATURE_DTYPES


def test_align_fixes_shuffled_column_order(sample, artifact):
    """Order is positional in a centroid comparison. A swap must not change the answer."""
    from src.predict import score
    shuffled = sample[list(reversed(FEATURES))]
    assert list(shuffled.columns) != list(FEATURES)
    assert (score(shuffled, artifact).segment_id.to_numpy()
            == score(sample, artifact).segment_id.to_numpy()).all()


def test_align_drops_extra_columns(sample, artifact):
    from src.predict import score
    extra = sample.assign(monetary_sum=1.0, promo_rate=0.3, __split="train")
    assert (score(extra, artifact).segment_id.to_numpy()
            == score(sample, artifact).segment_id.to_numpy()).all()


def test_align_rejects_missing_column(sample):
    with pytest.raises(ContractViolation, match="missing required features"):
        align(sample.drop(columns=["frequency"]))


def test_null_in_int_feature_is_rejected_not_imputed(sample):
    bad = sample.copy().astype({"frequency": "float64"})
    bad.loc[0, "frequency"] = np.nan
    with pytest.raises(ContractViolation, match="nulls are rejected"):
        align(bad)


def test_transform_is_row_preserving(sample, artifact):
    from src.predict import score
    for n in (1, 7, len(sample)):
        assert len(score(sample.head(n), artifact)) == n
