# Phase 2 runbook — QCOMM Customer Analytics

Real values baked in. The only things you type are two credential values and a GitHub
token. Every step has a verify command; you should never wonder whether it worked.

```
Account    117211782845          Bucket   qcomm-rfm        Region  ap-south-2
Cluster    qcomm-rfm-cluster     ECR      qcomm-rfm-api    NS      prod
Repo       github.com/BishalRanjanBadu/qcomm-customer-analytics
Artifact   v_20260909T1228Z_nogit    (gate spread 7.5838x, passed)
Shell      Git Bash (MINGW64) on Windows

REPO ROOT  C:\Users\HP\Desktop\Bishal_\END TO END PROJECTS\ML PROJECTS\QCOMM_CUSTOMER_ANALYTICS
  1.RAW/                  raw CSVs — gitignored, S3 is the source of truth
  <notebooks folder>/     Phase-1 notebooks — committed, they are the documentation
  3.DEPLOYMENT/           Phase 2: src, tests, k8s, Dockerfile   <- already extracted
  .github/                MUST be moved here from 3.DEPLOYMENT (step 2)
  .gitignore              MUST be moved here from 3.DEPLOYMENT (step 2)
```

**That path has spaces in `END TO END PROJECTS` and `ML PROJECTS`.** Every reference is
quoted. Unquoted, `cd` fails at `END`.

**`3.DEPLOYMENT` is a subfolder, not the repo root.** GitHub reads workflows *only* from
the repository root, so `.github/` must move up one level or CI silently never runs — no
error, no red X, just nothing. `.gitignore` moves up for the same reason: a `.gitignore`
governs its own directory and below, and `1.RAW/` is a sibling of `3.DEPLOYMENT`.

Everything else stays where you put it. The CI workflow is already parameterised with
`PROJECT_DIR: 3.DEPLOYMENT`.

**Git Bash specifics baked in:** no `python3` (uses `$(command -v python3 || command -v python)`),
`venv/Scripts/activate`, `127.0.0.1` never `localhost`, everything written to the repo
directory never `/tmp`.

**Order matters. Steps 1–12 cost nothing. Step 13 starts the meter.**

---

## Step 0 — Confirm what you already extracted

You have `3.DEPLOYMENT/` populated with 14 items. Confirm it is the audited build, not the
first one:

```bash
export REPO="/c/Users/HP/Desktop/Bishal_/END TO END PROJECTS/ML PROJECTS/QCOMM_CUSTOMER_ANALYTICS"
export DEPLOY="$REPO/3.DEPLOYMENT"
ls -a "$DEPLOY"
grep -c QCOMM_PHASE2_V3 "$DEPLOY/MANIFEST.md"
```

The `grep` must print **3**. If it prints `0`, you extracted the superseded v1 archive —
re-download `qcomm-phase2-v3-layout.zip` and replace `3.DEPLOYMENT/` before continuing.
v1 has an unauthenticated admin endpoint reachable through the public LoadBalancer and a
CI pipeline that fails on its second run.

You must also see `.github`, `.gitignore` and `.dockerignore` in that listing. Windows
Explorer hides dotfiles by default, so trust `ls -a`, not the screenshot.

```bash
ls "$REPO"
```

Note the exact name of your notebooks folder — you will need it in step 2.

---

## Step 1 — Clone the repo at the ROOT (deferred from Phase 1)

`git clone` refuses a non-empty directory, so this clones to a temp folder and moves `.git`
in. **Verification precedes every destructive step.**

```bash
ls -a "$REPO"
```

Must list `1.RAW` and `3.DEPLOYMENT`. If it says "No such file or directory", fix the path
before continuing.

```bash
cd ~/Desktop
git clone https://github.com/BishalRanjanBadu/qcomm-customer-analytics.git _qcomm_tmp
ls -a ~/Desktop/_qcomm_tmp && echo "CLONE OK"
```

"You appear to have cloned an empty repository" is fine — `.git` is still created. You must
see `.git` in that listing before continuing.

```bash
cp -r ~/Desktop/_qcomm_tmp/. "$REPO"/
cd "$REPO"
ls -a
git remote -v
```

`.git` present **and** a remote pointing at `qcomm-customer-analytics`. Only once both are
confirmed:

```bash
rm -rf ~/Desktop/_qcomm_tmp && echo "TEMP REMOVED"
```

---

## Step 2 — Move `.github` and `.gitignore` to the repo root

These two, and only these two. Everything else stays in `3.DEPLOYMENT/`.

**`.gitignore` first, before any `git add`.** Once 127 MB of CSV is in the history,
removing it means rewriting commits.

```bash
cd "$REPO"
mv "$DEPLOY/.gitignore" .
mv "$DEPLOY/.github" .
ls -a
```

Verify both landed at the root and left the subfolder:

```bash
test -f .gitignore                            && echo "root .gitignore OK"  || echo "MISSING"
test -f .github/workflows/mlops_pipeline.yml  && echo "root CI OK"          || echo "MISSING"
test -f .github/scripts/render_and_verify.sh  && echo "render script OK"    || echo "MISSING"
test -e "$DEPLOY/.github" && echo "!!! .github STILL in 3.DEPLOYMENT — CI will not run" \
                          || echo "3.DEPLOYMENT/.github correctly gone"
```

Confirm the rest of Phase 2 is intact where it belongs:

```bash
test -f "$DEPLOY/Dockerfile"                          && echo "docker OK"  || echo "MISSING"
test -f "$DEPLOY/src/api.py"                          && echo "src OK"     || echo "MISSING"
test -f "$DEPLOY/tests/fixtures/sample_customers.csv" && echo "fixture OK" || echo "MISSING"
test -f "$DEPLOY/.dockerignore"                       && echo "dockerignore OK (stays with the Dockerfile)" || echo "MISSING"
```

`.dockerignore` **stays** in `3.DEPLOYMENT/` — Docker reads it from the build context, and
the context is `./3.DEPLOYMENT`.

### Prove the ignore rules before committing anything

```bash
git add -A
git status --porcelain | grep "1\.RAW" && echo "!!! RAW DATA WOULD BE COMMITTED — STOP" \
                                        || echo "raw data ignored — OK"
git check-ignore -v "1.RAW/qcomm_orders.csv"
git status --porcelain | grep "3.DEPLOYMENT/tests/fixtures/sample_customers.csv" \
  && echo "fixture staged — OK" || echo "!!! FIXTURE EXCLUDED — CI will fail"
git status --porcelain | grep ".github/workflows" && echo "CI staged — OK"
```

`check-ignore` must print the rule catching the raw CSV. The fixture must still stage:
`*.csv` is ignored, but `!3.DEPLOYMENT/tests/fixtures/*.csv` exempts it, and the negation
only works because it comes after the rule it exempts.

If your notebooks folder is not staged, add it — it is the Phase-1 documentation:

```bash
git status --porcelain | grep -i notebook
```

```bash
git config core.autocrlf false     # CRLF vs LF creates phantom diffs in every file
git commit -q -m "Phase 2: serving, tests, container, k8s manifests, CI at repo root"
git log --oneline
git ls-files | head -40
```

Don't push yet. Push in step 20, after the cluster exists — otherwise CI fires against
missing infrastructure and fails in a way that looks like a code problem.

---

## Step 3 — Python environment

```bash
cd "$DEPLOY"
PY="$(command -v python3 || command -v python)"
"$PY" --version
"$PY" -m venv venv
source venv/Scripts/activate       # Git Bash on Windows. NOT venv/bin/activate.
python --version
```

**Install `requirements.txt`, not the lockfile.** A Linux lock legitimately contains
packages with no Windows wheels; the lock is for the image.

```bash
pip install -r requirements-dev.txt
pip list | grep -E "numpy|pandas|scikit-learn|scipy|fastapi|uvicorn|boto3"
```

Expect `numpy 2.1.3`, `pandas 2.2.3`, `scikit-learn 1.6.1`, `scipy 1.16.3` — the versions
from the promoted manifest, not the latest on PyPI.

---

## Step 4 — Tests, with no credentials at all

This is the important property: the test job never touches AWS.

```bash
env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u S3_BUCKET pytest tests/ -q
```

Expect `29 passed`. If it fails on a missing fixture, `.gitignore` ate it — go back to
step 2.

---

## Step 5 — AWS credentials, locally

Keys land in `~/.aws/`, **outside the repo**. Never in `.env`, never in a file you commit.

```bash
aws configure
# AWS Access Key ID     [paste yours]
# AWS Secret Access Key [paste yours]
# Default region name   ap-south-2
# Default output format json
```

```bash
aws sts get-caller-identity
```

Must show account `117211782845`. If it errors with `InvalidAccessKeyId`, diagnose before
retrying — never blind-retry:

```bash
env | grep -i AWS_          # env vars SHADOW ~/.aws/credentials
aws configure list          # the TYPE column names the WINNING source
```

---

## Step 6 — Verify the promoted artifact is reachable

```bash
aws s3 cp "s3://qcomm-rfm/dev/models/CURRENT.json" - --region ap-south-2
aws s3 ls "s3://qcomm-rfm/dev/models/v_20260909T1228Z_nogit/" --region ap-south-2
```

You need `kmeans.pkl`, `feature_names.pkl`, `manifest.json`, `metrics.json`.

---

## Step 7 — Serve locally against the REAL artifact

```bash
cd "$DEPLOY"
source venv/Scripts/activate
export AWS_REGION=ap-south-2 S3_BUCKET=qcomm-rfm ENV_PREFIX=dev STRICT_LIBS=true
```

Background it with a log file. A foreground server occupies its terminal and dies with it,
and you cannot then tell whether it crashed or was closed.

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000 > api.log 2>&1 &
API_PID=$!
echo "api pid: $API_PID"
sleep 8
tail -20 api.log
```

Use `-sS`, never `-s`. Lowercase `-s` silences connection errors, so a dead server looks
identical to an empty response.

```bash
curl -sS http://127.0.0.1:8000/live
curl -sS http://127.0.0.1:8000/health
```

`/health` must report `"status":"ready"` and `"model_version":"v_20260909T1228Z_nogit"`.

**If it reports `not_ready` with a manifest mismatch, that is the check working.** The
message names exactly which library differs. Do not disable `STRICT_LIBS` to get past it —
that is the defect the check exists to catch.

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/segment \
  -H "Content-Type: application/json" \
  -d '{"customers":[{"customer_id":"T1","recency_days":1.4,"frequency":13,"monetary_mean":583.5}]}'
```

---

## Step 8 — Record the golden payloads

**Against the real artifact, served by the real code.** A fixture-derived `expected.json`
makes the post-deploy job compare the live endpoint to a different model and fail forever.

```bash
python "$DEPLOY/tests/golden/record.py" --endpoint http://127.0.0.1:8000
cat "$DEPLOY/tests/golden/expected.json"
```

Confirm `"model_version": "v_20260909T1228Z_nogit"`. Then verify the checker agrees with
what it just recorded — **parity point 1 of 3**:

```bash
python "$DEPLOY/tests/golden/check_live.py" --endpoint http://127.0.0.1:8000
```

Expect `golden check PASSED ... (6 accept + 5 reject cases)`.

```bash
git add "$DEPLOY/tests/golden/expected.json"
git commit -q -m "Record golden expectations against v_20260909T1228Z_nogit"
```

Stop the local server:

```bash
kill "$API_PID" 2>/dev/null; sleep 2; tail -3 api.log
```

---

## Step 9 — Build the image

```bash
cd "$DEPLOY"
docker --version
docker build -t qcomm-rfm-api:local .
```

The build runs a self-check in the **final** stage — the one that ships, and the only one
with `libgomp1`. Watch for:

```
libs 2.1.3 2.2.3 1.6.1 1.16.3
feature_contract_hash a50fb3d15315ec03
image self-check OK
```

That hash must match the manifest's `feature_order_hash`. If it doesn't, the image's code
disagrees with the promoted artifact and the build fails here rather than at readiness.

```bash
docker images qcomm-rfm-api:local --format "{{.Repository}}:{{.Tag}} {{.Size}}"
docker run --rm --entrypoint sh qcomm-rfm-api:local -c "id -un && echo HOME=\$HOME"
```

Must print `appuser` and `HOME=/home/appuser`.

---

## Step 10 — Container parity — parity point 2 of 3

Mount your credentials **read-only**. Never bake a key into an image.

```bash
docker run -d --name qcomm-local -p 8000:8000 \
  -v "$HOME/.aws:/home/appuser/.aws:ro" \
  -e AWS_REGION=ap-south-2 -e S3_BUCKET=qcomm-rfm -e ENV_PREFIX=dev \
  qcomm-rfm-api:local
sleep 15
docker logs qcomm-local | tail -20
curl -sS http://127.0.0.1:8000/health
python "$DEPLOY/tests/golden/check_live.py" --endpoint http://127.0.0.1:8000
```

**A container returning a different number from local step 8 is a corrupt artifact — and
both would still return HTTP 200.** That is the entire reason this step exists.

Verify the entrypoint fails fast on a bad role:

```bash
docker run --rm -e ROLE=streamlit qcomm-rfm-api:local; echo "exit code: $?"
```

Must print a message naming the expected value and exit `78`, not "command not found".

```bash
docker rm -f qcomm-local
```

---

## Step 11 — Render-verify the manifests, both directions

Prove the guard fails as well as passes. A gate tested in one direction has not been tested.

The render script now lives at the repo root and takes the manifest directory as its first
argument, so it is called with the subfolder path.

```bash
cd "$REPO"
export IMAGE="117211782845.dkr.ecr.ap-south-2.amazonaws.com/qcomm-rfm-api:testtag"
bash .github/scripts/render_and_verify.sh "3.DEPLOYMENT/k8s" rendered
```

Expect `render verification PASSED`.

```bash
IMAGE="117211782845.dkr.ecr.ap-south-2.amazonaws.com/qcomm-rfm-api:" \
  bash .github/scripts/render_and_verify.sh "3.DEPLOYMENT/k8s" rendered_bad; echo "exit: $?"
IMAGE="117211782845.dkr.ecr.ap-south-2.amazonaws.com/qcomm-rfm-api:latest" \
  bash .github/scripts/render_and_verify.sh "3.DEPLOYMENT/k8s" rendered_bad; echo "exit: $?"
```

Both must fail with exit 1 — one on `EMPTY TAG`, one on `:latest`. The empty-tag case is
the important one: `image: registry/repo:` is valid YAML and becomes `ImagePullBackOff`
minutes later, which looks nothing like a config error.

```bash
rm -rf rendered rendered_bad
```

---

## Step 12 — ECR repo and image push

ECR costs pennies. This is still before the meter starts.

```bash
aws ecr create-repository --repository-name qcomm-rfm-api --region ap-south-2 \
  --image-scanning-configuration scanOnPush=true \
  --image-tag-mutability IMMUTABLE
```

If it already exists you'll get `RepositoryAlreadyExistsException` — that is the expected
benign case, not an error to mask.

```bash
export ACCOUNT=117211782845 REGION=ap-south-2 REG=117211782845.dkr.ecr.ap-south-2.amazonaws.com
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $REG
export SHA=$(git rev-parse --short=7 HEAD)
docker tag qcomm-rfm-api:local $REG/qcomm-rfm-api:$SHA
docker push $REG/qcomm-rfm-api:$SHA
```

**Read the scan findings. A control that is enabled must be read.** `aws ecr wait
image-scan-complete` raises `ScanNotFoundException` in the seconds before a scan is
queued, so poll the status field in a bounded loop instead:

```bash
for i in $(seq 1 30); do
  ST=$(aws ecr describe-images --repository-name qcomm-rfm-api --region $REGION \
       --image-ids imageTag=$SHA --query 'imageDetails[0].imageScanStatus.status' --output text)
  echo "scan status: $ST"; [ "$ST" = "COMPLETE" ] && break; sleep 10
done
aws ecr describe-images --repository-name qcomm-rfm-api --region $REGION \
  --image-ids imageTag=$SHA --query 'imageDetails[0].imageScanFindingsSummary.findingSeverityCounts'
```

`scanOnPush` is already enabled, so `start-image-scan` returns `LimitExceededException` —
that is the expected path, not a failure.

**Triage into two buckets:** packages this Dockerfile installs are yours and usually
removable; packages the base image ships are a policy decision — a `+debNuN` version suffix
means the security update is already applied and no further fix exists. Record the counts
and package names in the sign-off and move on.

Record the digest — the OS layer is not byte-reproducible because of `apt-get upgrade`:

```bash
aws ecr describe-images --repository-name qcomm-rfm-api --region $REGION \
  --image-ids imageTag=$SHA --query 'imageDetails[0].imageDigest' --output text
```

---

## Step 13 — Instance-type and architecture pre-check ⚠️ THE METER STARTS NEXT STEP

```bash
aws ec2 describe-instance-type-offerings --region ap-south-2 \
  --location-type availability-zone \
  --filters Name=instance-type,Values=t3.medium \
  --query 'InstanceTypeOfferings[].Location' --output text
```

You need at least two AZs listed. If this returns nothing, stop — a restricted account can
fail to launch instances with **no error and no instances, in every region**, and
region-hopping never fixes it.

```bash
aws ec2 describe-instance-types --region ap-south-2 --instance-types t3.medium \
  --query 'InstanceTypes[0].{arch:ProcessorInfo.SupportedArchitectures,mem:MemoryInfo.SizeInMiB,vcpu:VCpuInfo.DefaultVCpus}'
```

Must show `x86_64`. Your image is amd64 — an arm64 node fails with `exec format error`.

---

## Step 14 — Create the cluster 💸

> **This costs money from the moment it succeeds.**
> EKS control plane **$0.10/hour ≈ $73/month ≈ ₹6,400/month**, per cluster, billed whether
> or not any pod runs. Never free tier, at any node size.
> Plus one `t3.medium` (~₹2,600/mo) and one LoadBalancer (~₹1,750/mo) from step 17.
> **Total ≈ ₹10,800/month.** Steps 20–21 tear it all down.

Set a calendar reminder now. A cluster left past 14 months moves to extended support at
$0.60/hour — about ₹38,000/month.

```bash
eksctl version
eksctl create cluster \
  --name qcomm-rfm-cluster \
  --region ap-south-2 \
  --nodegroup-name qcomm-ng \
  --node-type t3.medium \
  --nodes 1 --nodes-min 1 --nodes-max 2 \
  --with-oidc \
  --managed
```

15–20 minutes. **`--with-oidc` is not optional** — without it `eksctl` warns "OIDC is
disabled on the cluster" and *continues*, and step 16 then silently produces a
ServiceAccount that cannot assume its role.

**If it times out, read the real status before acting.** `eksctl` can exceed its own wait
while CloudFormation is still `CREATE_IN_PROGRESS`. Never tear down on a hypothesis:

```bash
aws cloudformation describe-stacks --region ap-south-2 \
  --query 'Stacks[?contains(StackName,`qcomm`)].{n:StackName,s:StackStatus}' --output table
aws cloudformation describe-stack-events --region ap-south-2 \
  --stack-name eksctl-qcomm-rfm-cluster-cluster \
  --query 'StackEvents[?ResourceStatus==`CREATE_FAILED`].[LogicalResourceId,ResourceStatusReason]' \
  --output text
```

Verify:

```bash
aws eks update-kubeconfig --name qcomm-rfm-cluster --region ap-south-2
kubectl get nodes -o wide
kubectl config current-context
```

---

## Step 15 — Node headroom

```bash
kubectl describe node | grep -A8 "Allocated resources"
```

Allocatable on a `t3.medium` is ~3.5 GiB. System pods (coredns, aws-node, kube-proxy)
claim roughly 500 Mi. Your pod requests 768 Mi. That fits with room for a rolling surge —
and it is why canary (which needs ≥2 pods) is deferred.

---

## Step 16 — Namespace, config, IRSA

```bash
kubectl create namespace prod
kubectl apply -n prod -f k8s/config.yml
kubectl -n prod get configmap qcomm-rfm-config -o jsonpath='{.data}' && echo
```

The IRSA policy is read-only, scoped to `dev/*`. Phase 3 adds a scoped `s3:PutObject` when
prediction logging is enabled — the two flip together, or `LOG_PREDICTIONS=true` against a
read-only role gives `AccessDenied` on every request.

```bash
cd "$REPO"
cat > irsa-policy.json <<'JSON'
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": "arn:aws:s3:::qcomm-rfm",
     "Condition": {"StringLike": {"s3:prefix": ["dev/*"]}}},
    {"Effect": "Allow", "Action": ["s3:GetObject"], "Resource": "arn:aws:s3:::qcomm-rfm/dev/*"}
  ]
}
JSON

aws iam create-policy --policy-name qcomm-rfm-serving-readonly \
  --policy-document file://irsa-policy.json --region ap-south-2

eksctl create iamserviceaccount \
  --name qcomm-rfm-sa --namespace prod \
  --cluster qcomm-rfm-cluster --region ap-south-2 \
  --attach-policy-arn arn:aws:iam::117211782845:policy/qcomm-rfm-serving-readonly \
  --approve
```

The manifests reference the ServiceAccount **by name only**. Never hand-define it in a
manifest *and* via eksctl — two sources of truth.

```bash
kubectl -n prod get sa qcomm-rfm-sa -o jsonpath='{.metadata.annotations}' && echo
```

Must show an `eks.amazonaws.com/role-arn` annotation. **If that annotation is missing,
IRSA is not wired** and the pod will fail to read S3 — fix it here, not after the deploy.

---

## Step 17 — Render, verify, apply 💸 (LoadBalancer starts billing)

`$SHA` was set in step 12. Step 14 takes ~20 minutes, so if you reopened the terminal it
is gone — re-derive it rather than assuming. An empty `$SHA` renders `image: repo:`, which
`render_and_verify.sh` rejects, but re-deriving is cheaper than diagnosing.

```bash
cd "$REPO"
export SHA=$(git rev-parse --short=7 HEAD)
echo "SHA=$SHA"
export IMAGE="117211782845.dkr.ecr.ap-south-2.amazonaws.com/qcomm-rfm-api:$SHA"
aws ecr describe-images --repository-name qcomm-rfm-api --region ap-south-2 \
  --image-ids imageTag=$SHA --query 'imageDetails[0].imageTags' --output text
bash .github/scripts/render_and_verify.sh "3.DEPLOYMENT/k8s" rendered
grep -n "image:" rendered/deployment.yml
kubectl apply -n prod -f rendered/
kubectl -n prod rollout status deploy/qcomm-rfm-api --timeout=300s
```

**If a pod is not Running, read the real reason. Never blind-retry.**

| symptom | first command |
|---|---|
| `ImagePullBackOff` | `kubectl -n prod describe pod -l app=qcomm-rfm-api \| tail -30` |
| `CrashLoopBackOff` | `kubectl -n prod logs -l app=qcomm-rfm-api --previous` |
| `ContainerCreating` stuck | `kubectl -n prod get sa qcomm-rfm-sa` — IRSA missing (step 16) |
| `0/1 Ready`, no restarts | `kubectl -n prod logs -l app=qcomm-rfm-api` — manifest mismatch at load |
| `Pending` | `kubectl -n prod describe pod ... \| grep -A5 Events` — insufficient memory |

```bash
kubectl -n prod get pods -o wide
kubectl -n prod logs -l app=qcomm-rfm-api | tail -20
```

---

## Step 18 — Live verification — parity point 3 of 3

```bash
for i in $(seq 1 30); do
  LB=$(kubectl -n prod get svc qcomm-rfm-api -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null)
  [ -n "$LB" ] && break; echo "waiting for LB ($i/30)"; sleep 10
done
echo "endpoint: http://$LB"
```

DNS takes another 2–4 minutes after the hostname appears.

```bash
curl -sS "http://$LB/live"
curl -sS "http://$LB/health"
python "$DEPLOY/tests/golden/check_live.py" --endpoint "http://$LB"
```

**This is the third parity point.** Local (step 8), container (step 10), and through the
load balancer must agree to the decimal on the same payload — including the five cases that
must be *rejected*. A rejection that starts returning 200 is as serious as a wrong number.

---

## Step 19 — Rollback drill (required before sign-off)

An untested rollback is not a rollback. Rollback here is an **alias write** — no artifact
is ever overwritten, so both directions are reversible.

The drill uses `kubectl rollout restart`, because the artifact loads at **startup**. There
is deliberately no `/admin/reload` endpoint: the Service publishes port 80 through an
internet-facing LoadBalancer, so an unauthenticated admin route would let anyone force a
model reload. Restart is also the stronger demonstration — with `replicas: 1` and
`maxUnavailable: 0`, a new pod that fails readiness **never receives traffic and the old
pod keeps serving throughout**.

```bash
cd "$REPO"
aws s3 cp "s3://qcomm-rfm/dev/models/CURRENT.json" ./CURRENT.backup.json --region ap-south-2
cat CURRENT.backup.json
```

### 19a — Point the alias at a version that does not exist

```bash
cat > current-bad.json <<'JSON'
{"version":"v_rollback_drill_bad","prefix":"models/v_does_not_exist","promoted_utc":"drill","gate_passed":true}
JSON
aws s3 cp current-bad.json "s3://qcomm-rfm/dev/models/CURRENT.json" --region ap-south-2
kubectl -n prod rollout restart deploy/qcomm-rfm-api
```

Watch the new pod fail while the old one keeps serving. Give it about 90 seconds:

```bash
sleep 90
kubectl -n prod get pods -l app=qcomm-rfm-api
kubectl -n prod logs -l app=qcomm-rfm-api --tail=15
```

You should see two pods: the old one `1/1 Running`, the new one `0/1 Running` with a log
line naming the load failure. Now the property that matters — **the endpoint is still
healthy**:

```bash
curl -sS "http://$LB/health"
python "$DEPLOY/tests/golden/check_live.py" --endpoint "http://$LB"
```

Both must still pass, against the **original** version. A bad promotion failed readiness,
nothing rolled forward, and no traffic ever reached a pod that could not serve it.

```bash
kubectl -n prod rollout status deploy/qcomm-rfm-api --timeout=90s; echo "exit: $?"
```

This is **expected to time out with a non-zero exit** — that is the rollout correctly
refusing to complete.

### 19b — Restore, and time it

```bash
date +%s
aws s3 cp CURRENT.backup.json "s3://qcomm-rfm/dev/models/CURRENT.json" --region ap-south-2
kubectl -n prod rollout restart deploy/qcomm-rfm-api
kubectl -n prod rollout status deploy/qcomm-rfm-api --timeout=180s
date +%s
```

Subtract the two timestamps and record the elapsed seconds in the sign-off.

```bash
kubectl -n prod get pods -l app=qcomm-rfm-api
curl -sS "http://$LB/health"
python "$DEPLOY/tests/golden/check_live.py" --endpoint "http://$LB"
rm -f current-bad.json
```

The golden check must pass against the restored version. That is what proves the rollback
restored the **right** model, not merely *a* model — `/health` alone would report ready for
any artifact that loads.

```bash
aws s3 cp "s3://qcomm-rfm/dev/models/CURRENT.json" - --region ap-south-2
```

Confirm it reads `v_20260909T1228Z_nogit` again before moving on.

---

## Step 20 — GitHub OIDC and the first CI run

No long-lived AWS key is stored anywhere.

```bash
aws iam list-open-id-connect-providers --region ap-south-2 | grep token.actions.githubusercontent.com \
  || aws iam create-open-id-connect-provider \
       --url https://token.actions.githubusercontent.com \
       --client-id-list sts.amazonaws.com \
       --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1
```

```bash
cd "$REPO"
cat > gha-trust.json <<'JSON'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Federated": "arn:aws:iam::117211782845:oidc-provider/token.actions.githubusercontent.com"},
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": {"token.actions.githubusercontent.com:aud": "sts.amazonaws.com"},
      "StringLike": {"token.actions.githubusercontent.com:sub": "repo:BishalRanjanBadu/qcomm-customer-analytics:*"}
    }
  }]
}
JSON

aws iam create-role --role-name qcomm-rfm-github-actions \
  --assume-role-policy-document file://gha-trust.json
aws iam attach-role-policy --role-name qcomm-rfm-github-actions \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser
aws iam attach-role-policy --role-name qcomm-rfm-github-actions \
  --policy-arn arn:aws:iam::117211782845:policy/qcomm-rfm-serving-readonly
```

The role must also be able to call `kubectl`. **If the CI role is not the cluster creator,
every `kubectl` call is `Unauthorized`:**

```bash
eksctl create iamidentitymapping \
  --cluster qcomm-rfm-cluster --region ap-south-2 \
  --arn arn:aws:iam::117211782845:role/qcomm-rfm-github-actions \
  --group system:masters --username github-actions
kubectl describe configmap aws-auth -n kube-system | grep -A5 mapRoles
```

**Secrets scan before pushing:**

```bash
cd "$REPO"
git grep -nE "(AKIA[0-9A-Z]{16}|aws_secret_access_key\s*=\s*[A-Za-z0-9/+=]{40})" -- . && \
  echo "!!! SECRET FOUND — remove and rewrite the commit, never use an unblock link" \
  || echo "no credential patterns found"
```

If a scanner flags something, read the actual file and line before advising rotation — a
placeholder can match a secret's shape.

```bash
git add -A
git commit -q -m "Phase 2 complete: verified locally, in container, and through the LB"
git push -u origin main
```

The password prompt wants a **Personal Access Token**, not your password. GitHub →
Settings → Developer settings → Personal access tokens → Fine-grained → repo
`qcomm-customer-analytics`, permission `Contents: Read and write`.

Watch the run: `https://github.com/BishalRanjanBadu/qcomm-customer-analytics/actions`

Four jobs: `test` → `build` → `deploy` → `golden`. The `golden` job now enforces, because
`expected.json` was committed in step 8.

---

## Step 21 — Capture the evidence, then tear down 💸

Capture before deleting. Everything valuable lives in artifacts and logs, not in a running
cluster.

```bash
cd "$REPO"
mkdir -p "$REPO/evidence"
kubectl -n prod get all              > "$REPO/evidence"/k8s_resources.txt
kubectl -n prod describe deploy qcomm-rfm-api > "$REPO/evidence"/deployment.txt
kubectl -n prod logs -l app=qcomm-rfm-api --tail=200 > "$REPO/evidence"/pod_logs.txt
curl -sS "http://$LB/health"         > "$REPO/evidence"/health.json
python "$DEPLOY/tests/golden/check_live.py" --endpoint "http://$LB" > "$REPO/evidence"/golden_live.txt 2>&1
aws ecr describe-images --repository-name qcomm-rfm-api --region ap-south-2 \
  --image-ids imageTag=$SHA > "$REPO/evidence"/image_digest_and_scan.json
git add evidence && git commit -q -m "Phase 2 evidence" && git push
```

**Delete the LoadBalancer Service BEFORE the cluster,** or the ELB is orphaned and keeps
billing after the cluster is gone:

```bash
kubectl -n prod delete svc qcomm-rfm-api
sleep 60
aws elbv2 describe-load-balancers --region ap-south-2 \
  --query 'LoadBalancers[].LoadBalancerName' --output text
```

That must come back empty. Only then:

```bash
eksctl delete cluster --name qcomm-rfm-cluster --region ap-south-2 --wait
```

Verify nothing is left billing:

```bash
aws eks list-clusters --region ap-south-2
aws cloudformation describe-stacks --region ap-south-2 \
  --query 'Stacks[?contains(StackName,`qcomm`)].{n:StackName,s:StackStatus}' --output table
aws ec2 describe-instances --region ap-south-2 \
  --filters Name=instance-state-name,Values=running \
  --query 'Reservations[].Instances[].InstanceId' --output text
```

`list-clusters` empty, no non-`DELETE_COMPLETE` stacks, no running instances.

Keep the ECR repo and the S3 bucket — pennies, and Phase 3 needs them.

---

## What is NOT in Phase 2

- No drift job, no `detect_drift.py`, no EventBridge or Lambda
- No SageMaker retraining pipeline
- No label collection or feedback loop
- No MLflow tracking backend
- No CloudWatch alarms or SLO thresholds
- **No prediction logging** — `LOG_PREDICTIONS=false`. Phase 3 enables it together with
  the `s3:PutObject` IRSA scope, in the same change.
- **No canary bake monitor** — see `k8s/canary/README.md`. A rollout gate that reads a
  signal nothing produces fails closed on every deploy.

## Phase-2 handoff

Record and paste back:

```
image tag           qcomm-rfm-api:<sha>
image digest        sha256:...
model promoted      v_20260909T1228Z_nogit
parity              local / container / LB  — all three match?  Y/N
rollback drill      elapsed seconds, golden check after restore  PASS/FAIL
CVE counts          HIGH=?  CRITICAL=?  accept/fix decision
CI run              4 jobs green?  URL
deviations          canary deferred (1 node); bake monitor deferred (no prediction log);
                    artifact git_sha is "nogit" (repo was not cloned at Phase-1 run time)
teardown            LB deleted before cluster?  clusters/stacks/instances all empty?
```

**STOP for recorded sign-off before Phase 3.**
