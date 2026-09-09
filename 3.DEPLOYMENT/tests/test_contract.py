"""Edge validation. A bad request returns 4xx, never a prediction."""
import numpy as np, pandas as pd, pytest
from src.preprocess import ContractViolation, align, transform


def test_negative_value_rejected(sample, artifact):
    bad = sample.copy(); bad.loc[0, "monetary_mean"] = -5.0
    with pytest.raises(ContractViolation, match="negative values"):
        transform(bad, artifact.scaler_mean, artifact.scaler_scale)


def test_unparseable_value_rejected(sample):
    bad = sample.copy().astype({"frequency": "object"})
    bad.loc[0, "frequency"] = "many"
    with pytest.raises(ContractViolation, match="cannot be cast"):
        align(bad)


@pytest.mark.parametrize("payload,field", [
    ({"recency_days": 46.0, "frequency": 3, "monetary_mean": 500.0}, "recency_days"),
    ({"recency_days": -1.0, "frequency": 3, "monetary_mean": 500.0}, "recency_days"),
    ({"recency_days": 5.0, "frequency": 0, "monetary_mean": 500.0}, "frequency"),
    ({"recency_days": 5.0, "frequency": 3, "monetary_mean": 0.0}, "monetary_mean"),
])
def test_pydantic_rejects_out_of_range(payload, field):
    from pydantic import ValidationError
    from src.api import Customer
    with pytest.raises(ValidationError, match=field):
        Customer(**payload)


def test_no_field_has_a_default():
    """A caller who omits an input must get 422, not a confident 200 from placeholders."""
    from pydantic import ValidationError
    from src.api import Customer
    for omit in ("recency_days", "frequency", "monetary_mean"):
        p = {"recency_days": 5.0, "frequency": 3, "monetary_mean": 500.0}
        p.pop(omit)
        with pytest.raises(ValidationError):
            Customer(**p)


def test_nan_is_not_valid_json_for_clients(sample):
    """pandas NaN is not JSON. Clients send null; the API rejects it for int features."""
    recs = sample.head(2).astype(object).where(pd.notna(sample.head(2)), None).to_dict("records")
    assert all(v is not None for r in recs for v in r.values())
