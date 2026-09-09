"""API-level checks against the fixture artifact, no S3 and no credentials."""
import numpy as np, pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(artifact, monkeypatch_session):
    """Patch load_current so the REAL startup path runs against the fixture artifact."""
    import src.api as api
    monkeypatch_session.setattr(api, "load_current", lambda **kw: artifact)
    with TestClient(api.app) as c:
        yield c


def test_live_does_no_io(client):
    assert client.get("/live").json() == {"status": "alive"}


def test_health_reports_version_and_features(client):
    b = client.get("/health").json()
    assert b["status"] == "ready" and b["k"] == 4
    assert b["features"] == ["recency_days", "frequency", "monetary_mean"]


def test_segment_happy_path(client):
    r = client.post("/v1/segment", json={"customers": [
        {"customer_id": "C1", "recency_days": 1.4, "frequency": 13, "monetary_mean": 583.5},
        {"customer_id": "C2", "recency_days": 21.6, "frequency": 2, "monetary_mean": 327.9}]})
    assert r.status_code == 200
    b = r.json()
    assert b["source"] == "model" and b["contract_version"] == "v1"
    assert len(b["results"]) == 2
    names = {x["segment_name"] for x in b["results"]}
    assert names <= {"Champions", "Steady Regulars", "High-Value At-Risk", "Lost / Dormant"}


def test_missing_field_is_422_not_200(client):
    r = client.post("/v1/segment", json={"customers": [{"recency_days": 5.0, "frequency": 3}]})
    assert r.status_code == 422


def test_out_of_window_recency_is_422(client):
    r = client.post("/v1/segment", json={"customers": [
        {"recency_days": 99.0, "frequency": 3, "monetary_mean": 500.0}]})
    assert r.status_code == 422


def test_null_is_422(client):
    r = client.post("/v1/segment", json={"customers": [
        {"recency_days": None, "frequency": 3, "monetary_mean": 500.0}]})
    assert r.status_code == 422


def test_empty_batch_is_422(client):
    assert client.post("/v1/segment", json={"customers": []}).status_code == 422


def test_row_count_matches_request(client, sample):
    recs = [{"customer_id": f"C{i}", **sample.iloc[i][["recency_days", "frequency",
            "monetary_mean"]].to_dict()} for i in range(25)]
    for r_ in recs:
        r_["frequency"] = int(r_["frequency"])
    b = client.post("/v1/segment", json={"customers": recs}).json()
    assert len(b["results"]) == 25


def test_failed_load_clears_the_artifact_and_returns_503(monkeypatch):
    """A pod whose load failed must refuse traffic — never serve a stale model as a new one."""
    import src.api as api
    def boom(**kw):
        raise RuntimeError("simulated manifest mismatch")
    monkeypatch.setattr(api, "load_current", boom)
    with TestClient(api.app, raise_server_exceptions=False) as c:
        assert api._ART is None, "a failed load must clear the cached artifact"
        assert c.get("/live").status_code == 200          # liveness unaffected
        assert c.get("/health").status_code == 503
        r = c.post("/v1/segment", json={"customers": [
            {"recency_days": 5.0, "frequency": 3, "monetary_mean": 500.0}]})
        assert r.status_code == 503
