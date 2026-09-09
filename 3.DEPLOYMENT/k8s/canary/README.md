# Canary — correct, and DEFERRED

Canary needs at least 2 pods so traffic can split between tracks. A single `t3.medium`
(4 GiB, ~3.5 GiB allocatable, minus ~500 Mi of system pods) cannot host two pods at a
768 Mi request with headroom for a rolling surge.

**This is a documented deviation on constrained infrastructure. It does not waive the
requirement on real infrastructure.**

## What changes on a 2-node cluster

1. Apply `deployment-canary.yml` alongside `deployment.yml`.
2. Change `k8s/service.yml` selector from `{app: qcomm-rfm-api, track: stable}` to
   `{app: qcomm-rfm-api}` so the Service fronts **both** tracks; endpoint count then
   splits traffic by replica ratio (1 canary : 4 stable ≈ 20%).
3. Bake for the agreed window while comparing error rate, p99 latency and segment
   distribution against the stable track.
4. Ramp, or scale the canary to 0 and roll back.

## The bake monitor is ALSO deferred, and here is why

An automatic rollback needs a signal to read. Comparing segment distribution between
tracks requires prediction logging, and `LOG_PREDICTIONS` is `false` until Phase 3 flips
it together with the `s3:PutObject` scope on the IRSA policy.

**A rollout gate that reads a signal nothing produces fails closed on every deploy.** So
the guard is explicitly deferred rather than shipped inert. Phase 3 enables logging and
the guard in the same change.

Until then the rollback path is manual and rehearsed: `models/CURRENT.json` is rewritten
to a prior prefix and the deployment is restarted. Runbook step 19 drills it.
