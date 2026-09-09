import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.registry import Artifact

FIX = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def manifest():
    return json.loads((FIX / "fixture_manifest.json").read_text())


@pytest.fixture(scope="session")
def artifact(manifest):
    """The fixture artifact, assembled exactly as registry.load_current() assembles a real one."""
    m = manifest["model"]
    return Artifact(
        version="fixture", prefix="fixtures", 
        model=pickle.loads((FIX / "fixture_model.pkl").read_bytes()),
        features=tuple(m["features"]), dtypes=manifest["feature_dtypes"],
        scaler_mean=np.asarray(m["scaler_mean"]), scaler_scale=np.asarray(m["scaler_scale"]),
        segment_names={int(k): v for k, v in m["segment_names"].items()},
        contract_version=manifest["contract_version"], manifest=manifest)


@pytest.fixture(scope="session")
def sample():
    return pd.read_csv(FIX / "sample_customers.csv")


@pytest.fixture(scope="session")
def monkeypatch_session():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()
