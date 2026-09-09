"""Record golden expectations from a LIVE endpoint serving the REAL promoted artifact."""
import argparse, json, sys, urllib.request
from pathlib import Path
HERE = Path(__file__).resolve().parent


def post(endpoint, body):
    req = urllib.request.Request(f"{endpoint}/v1/segment",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True, help="e.g. http://127.0.0.1:8000")
    a = ap.parse_args()

    pay = json.loads((HERE / "payloads.json").read_text())
    health = json.loads(urllib.request.urlopen(f"{a.endpoint}/health", timeout=30).read())
    if health.get("status") != "ready":
        sys.exit(f"endpoint not ready: {health}")

    customers = [{k: v for k, v in c.items() if k != "name"} for c in pay["accept"]]
    status, body = post(a.endpoint, {"customers": customers})
    if status != 200:
        sys.exit(f"accept batch returned {status}: {body}")

    expected = {
        "model_version": body["model_version"],
        "contract_version": body["contract_version"],
        "tolerance": 1e-6,
        "results": {r["customer_id"]: {"segment_id": r["segment_id"],
                                       "segment_name": r["segment_name"],
                                       "distance_to_centroid": r["distance_to_centroid"]}
                    for r in body["results"]},
    }
    (HERE / "expected.json").write_text(json.dumps(expected, indent=2))
    print(f"recorded {len(expected['results'])} golden results against {expected['model_version']}")
    for cid, r in expected["results"].items():
        print(f"  {cid}  {r['segment_name']:20s} d={r['distance_to_centroid']}")


if __name__ == "__main__":
    main()
