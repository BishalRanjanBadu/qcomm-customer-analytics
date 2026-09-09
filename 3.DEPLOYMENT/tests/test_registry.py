"""Manifest validation. A mismatched artifact must fail readiness, not serve."""
import copy, pytest
from src.registry import validate_manifest, EXPECTED_FEATURE_ORDER_HASH


def test_good_manifest_passes(manifest):
    assert validate_manifest(manifest, strict_libs=False) == []


def test_wrong_contract_version_rejected(manifest):
    m = copy.deepcopy(manifest); m["contract_version"] = "v2"
    assert any("contract_version" in p for p in validate_manifest(m, strict_libs=False))


def test_changed_feature_order_hash_rejected(manifest):
    m = copy.deepcopy(manifest); m["feature_order_hash"] = "deadbeefdeadbeef"
    assert any("feature_order_hash" in p for p in validate_manifest(m, strict_libs=False))


def test_wrong_dtype_rejected(manifest):
    m = copy.deepcopy(manifest); m["feature_dtypes"]["frequency"] = "float64"
    assert any("dtype for frequency" in p for p in validate_manifest(m, strict_libs=False))


def test_library_skew_detected(manifest):
    """The defect that silently corrupts fitted state. Must be caught at load."""
    m = copy.deepcopy(manifest); m["library_versions"]["scikit-learn"] = "1.9.0"
    problems = validate_manifest(m, strict_libs=False)
    assert any("scikit-learn" in p for p in problems)
