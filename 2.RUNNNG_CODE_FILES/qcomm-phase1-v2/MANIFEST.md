# qcomm-phase1-v2 — manifest

**Build marker:** `QCOMM_PHASE1_V2_2026-09-09`

Browsers cache archives by filename and sometimes serve a stale one. Check this marker,
not the filename.

## Expected tree

```
qcomm-phase1-v2/
├── MANIFEST.md                                  this file
├── PHASE1_RUNBOOK.md                            start here — real paths baked in
└── notebooks/
    ├── 01_Data_Loading_and_First_Look.ipynb     + .py
    ├── 02_Data_Cleaning.ipynb                   + .py
    ├── 03_Missing_Values_and_Outliers.ipynb     + .py
    ├── 04_Statistics_and_EDA.ipynb              + .py
    ├── 05_Hypothesis_Testing.ipynb              + .py
    ├── 06_Scaling_and_Feature_Pipeline.ipynb    + .py
    └── 07_Segmentation_Gates_and_Artifacts.ipynb + .py
```

16 files. The `.py` files are jupytext percent-format sources of the same notebooks —
they diff cleanly in git, where `.ipynb` JSON does not. Upload the `.ipynb` to Colab.

Verify after extraction:

```bash
ls notebooks/*.ipynb | wc -l     # -> 7
grep -c QCOMM_PHASE1_V2 MANIFEST.md   # -> 3
```

## Baked-in values

| | |
|---|---|
| Bucket / region / env | `qcomm-rfm` / `ap-south-2` / `dev` |
| Snapshot | 2026-02-15 (day 45) |
| Feature window | **45 days** (vendor dictionary says 90 — wrong) |
| Drift reference | days **30–44** (days 0–44 gives a permanent false alarm) |
| Seed | 42 |

No `<PLACEHOLDER>` anywhere. The only values you type are the two Colab secrets.

## What was verified before shipping

Every stage was executed locally against the real CSVs. These are measured, not predicted:

| check | result |
|---|---|
| D1 feature window | `max(tenure)=45.00`, `max(recency)=45.00` |
| D2 join rule | 43,111 orders customers / 35,189 flagged / 7,922 orphans, all post-snapshot |
| D3 identity leak | `dormant_45d == (orders_next_45d == 0)` on all 439,492 rows |
| Enum drift | 497 rows, first on day **54** |
| Clock skew | 661 rows (0.17%), max lag within the 900 s bound |
| MNAR `order_value` | 7.2× top/bottom decile blank rate |
| Split | 70/15/15 stratified; dormancy 0.2875 in all three |
| Rebuild parity | recency **1.0000**, tenure **1.0000**, dormancy **1.0000**, frequency 0.9937 |
| Drift reference | `orders_30d` PSI 0.2585 (d0–44) → **0.0013** (d30–44) |
| k sweep | 4.33× (k=2) → 7.93× (k=4) → 13.98× (k=7) → 12.72× (k=8), all monotone |
| **Gate A** | spread **7.58×**, null p95 **1.0214**, ARI **0.9903** vs vendor → **PASS** |
| Permutation null | monotone in **10/10** runs → monotonicity alone proves nothing |
| **Negative control** | random partition **1.021×** → correctly **FAILS** |
| Silhouette (test) | 0.265 — reported, never gating |

All seven notebooks parse under `ast.parse` and were converted with jupytext. Outputs are
stripped (0 output cells), so the repo does not bloat with embedded plot data.

## Known gaps — deliberate

- **Not run against your S3.** The stage logic was executed locally with the filesystem
  standing in for S3; the S3 helper itself is exercised the first time you run notebook 01.
- **No `requirements.txt`.** Colab's pinned versions become the train/serve parity
  contract, and I cannot read them from here. Notebook 07 prints them into the handoff.
  Nothing here is safe to copy into a Dockerfile yet.
- **No `src/`, tests, Dockerfile or CI.** Phase 2.
- **No `.gitignore`.** Git setup was deferred. It must exclude `1.RAW/` and `*.csv`
  **before** the first `git add`.
- **The gate threshold is PROVISIONAL** — statistically derived (~185× the null p95), no
  cost model. Review trigger recorded in `metrics.json`.

## Design decisions worth knowing

- **One model.** KMeans on log-scaled R/F/M. There is no supervised classifier.
- **`dormant_45d` is the referee, not a target.** It is measured on the 45 days after the
  snapshot — a window the clustering never sees — which makes it the only external
  validation available. It sits in `FORBIDDEN`, so it can never become an input: a
  segmentation built on the outcome would be validating against itself.
- **Two chains.** Chain A (customer table) fits the model. Chain B (orders → rebuilt
  snapshot) is the feature pipeline that becomes `src/preprocess.py` in Phase 2 and is the
  only source of drift for Phase 3.
- **`PROFILE_COLS` describe segments, they never define them.** `monetary_sum` is excluded
  from the clustering because sum = mean × count would count frequency twice in the
  distance metric.
- **`__split` is stamped once**, in notebook 01, and travels through every stage. Every
  `.fit()` keys off `__split == 'train'`. Notebook 06 carries an executable leakage
  regression test that proves the boundary is real.
- **Stability ARI and ANOVA F are excluded as gates.** Both pass on shuffled data where no
  signal exists by construction. Notebook 07 says so, in the notebook, so nobody
  reintroduces them later.
- **Geography is excluded** from the rebuilt customer table by design — `city_id`,
  `delivery_city`, `billing_city` are the classic proxy for protected characteristics.
- **Outliers are capped, never dropped.** The top spenders are the segment the project
  exists to find.

Build marker: `QCOMM_PHASE1_V2_2026-09-09`
