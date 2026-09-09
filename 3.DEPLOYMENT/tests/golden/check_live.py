"""Compare a live endpoint against the recorded golden expectations.

Run locally (three-point parity) and automatically after every deploy.
"""
import argparse, json, sys, urllib.error, urllib.request
from pathlib import Path
HERE = Path(__file__).resolve().parent


def post(endpoint, body):
    req = urllib.request.Request(f"{endpoint}/v1/segment", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    a = ap.parse_args()

    exp_path = HERE / "expected.json"
    if not exp_path.exists():
        print("expected.json absent — recorded by runbook step 8. Skipping.")
        return 0
    exp = json.loads(exp_path.read_text())
    pay = json.loads((HERE / "payloads.json").read_text())
    tol = exp["tolerance"]
    failures = []

    health = json.loads(urllib.request.urlopen(f"{a.endpoint}/health", timeout=30).read())
    if health.get("model_version") != exp["model_version"]:
        failures.append(f"model_version: live {health.get('model_version')} "
                        f"!= golden {exp['model_version']}")

    customers = [{k: v for k, v in c.items() if k != "name"} for c in pay["accept"]]
    status, body = post(a.endpoint, {"customers": customers})
    if status != 200:
        failures.append(f"accept batch returned {status}: {body}")
    else:
        for r in body["results"]:
            want = exp["results"].get(r["customer_id"])
            if want is None:
                failures.append(f"{r['customer_id']}: not in golden set"); continue
            if r["segment_id"] != want["segment_id"]:
                failures.append(f"{r['customer_id']}: segment {r['segment_id']} != {want['segment_id']}")
            d = abs(r["distance_to_centroid"] - want["distance_to_centroid"])
            if d > tol:
                failures.append(f"{r['customer_id']}: distance drift {d:.3e} > {tol:.0e}")

    # A rejection that starts returning 200 is as serious as a wrong number.
    for case in pay["reject"]:
        st, _ = post(a.endpoint, {"customers": [case["payload"]]})
        if st != case["expect_status"]:
            failures.append(f"reject/{case['name']}: got {st}, expected {case['expect_status']}")

    if failures:
        print("GOLDEN CHECK FAILED", file=sys.stderr)
        for f in failures:
            print("  -", f, file=sys.stderr)
        return 1
    print(f"golden check PASSED against {exp['model_version']} "
          f"({len(exp['results'])} accept + {len(pay['reject'])} reject cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
