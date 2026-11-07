# MLOps Homework 01 — Customer Churn Prediction

A reproducible ML project for the telecom **churn** binary-classification task:
config-driven training, MLflow tracking, data versioning, tests, FastAPI
serving, Docker, drift monitoring and a CI/CD workflow.

- **Course:** Daneshkar AI Bootcamp — MLOps · **Exercise:** HW_MLOps_01
- **Target:** `churn` (0 = stayed, 1 = left), 12 numeric features
- **Primary metric:** `f1_score` (the classes are imbalanced, ~26% churn)
- Persian guide: **[README_FA.md](README_FA.md)**
- Looking for the commands? They are all in
  **[§12 How to run and test](#12-how-to-run-and-test)** at the end of this file.

---

## 1. Project structure

```
Project/
├── config/config.yaml              every parameter (no hard-coded values)
├── data/raw/                       dataset_v1.csv (400 rows), dataset_v2.csv (480)
├── data_versions/                  metadata.json (version_id, hash, ...) + snapshots
├── src/
│   ├── data/load_data.py           load_csv, validate_schema, split_data, hashing, versioning
│   ├── features/build_features.py  scale_features, select_features, build/transform pipeline
│   ├── models/train_model.py       get_model, train_model, evaluate_model, save/load_model
│   ├── monitoring/drift_detection.py  JSON prediction logs, KS + mean shift, report
│   └── serving/app.py              FastAPI: /health, /predict, /metadata
├── experiments/
│   ├── train_experiment.py         single full pipeline + MLflow
│   ├── run_experiments.py          5 tracked runs + comparison table
│   ├── logs/                       predictions_YYYY-MM-DD.json (JSONL)
│   └── results/                    per-run summary + comparison.md/json
├── tests/                          data, features, models, monitoring, serving
├── scripts/drift_report.py         drift report CLI (CI gate)
├── scripts/healthcheck.py          post-deployment contract check
├── models/                         model.joblib, preprocessor.joblib, metadata.json
├── reports/                        drift_report.json + drift_report.md
├── Dockerfile · requirements.txt · requirements-serve.txt
└── .github/workflows/ci.yml        quality → test → train → drift → docker smoke
```

## 2. Dataset

| version          | rows | churn rate | purpose                          |
| ---------------- | ---- | ---------- | -------------------------------- |
| `dataset_v1.csv` | 400  | 26.3%      | reference training data          |
| `dataset_v2.csv` | 480  | 48.1%      | shifted distribution, drift test |

Registered versions in `data_versions/metadata.json`: `v1`
`446b605d5006…ad7a`, `v2` `6bd52cc55b6c…8162`. `verify_data_version`
recomputes the SHA-256, and the same hash is stored in the model metadata, so
every artifact is traceable back to exact data bytes.

## 3. Preprocessing

1. Load the CSV, separate `churn` → `y`, the other 12 columns → `X`.
2. Stratified split, `test_size=0.2`, `random_state=42`.
3. Scale with `StandardScaler` (or `none` for tree models).
4. Select `k=10` features with `SelectKBest(f_classif)` — drops `num_services`
   and `age`.
5. Train the model configured in `config.yaml`, then report accuracy and F1.

**The scaler and the selector are fitted on the training split only**, so no
test information leaks into training. Every model uses
`class_weight="balanced"`: without it the forest predicts the majority class
and reaches ~0.70 accuracy with F1 ≈ 0.08 — which is exactly why F1 is the
primary metric here.

## 4. Results

`mlops_homework_01` holds 6 MLflow runs; each differs in data version, model
type or hyperparameters.

| Run               | Data | Model               | Scaling  | k   | Accuracy | F1         |
| ----------------- | ---- | ------------------- | -------- | --- | -------- | ---------- |
| `lr_v2_standard`  | v2   | logistic_regression | standard | 10  | 0.6979   | **0.6947** |
| `svm_v1_standard` | v1   | svm (rbf)           | standard | 10  | 0.6875   | 0.5098     |
| `lr_v1_k8`        | v1   | logistic_regression | standard | 8   | 0.6500   | 0.5000     |
| `lr_v1_standard`  | v1   | logistic_regression | standard | 10  | 0.6125   | 0.4746     |
| `rf_v1_none`      | v1   | random_forest       | none     | 10  | 0.5250   | 0.2083     |

The best row is the v2 run, but that is not a fair comparison — v2 is a
different, easier distribution. On the reference v1 data the ranking is
`svm > lr > rf`. The **served** artifact is the v1 logistic regression: it is
the config default, needs no probability calibration and its coefficients are
interpretable.

Each run logs parameters (data version + hash, model type, scaling, k, test
size, random state, all hyperparameters), metrics (accuracy, F1, precision,
recall, train time, v2 metrics), tags and artifacts (`model.joblib`,
`preprocessor.joblib`, `metadata.json`, run summary).

## 5. Data and model versioning

`create_data_version` writes one entry per version (`version_id`, `hash`,
`description`, `n_rows`, `created_at`) and freezes a CSV snapshot.

`models/metadata.json` describes the served artifact:

```json
{
  "model_version": "logistic_regression-v1-859",
  "data_version": "v1",
  "metric": { "name": "f1_score", "value": 0.4746 },
  "metrics": { "accuracy": 0.6125, "f1_score": 0.4746, "precision": 0.3684, "recall": 0.6667 },
  "selected_features": ["tenure_months", "..."],
  "data_hash": "446b605d5006…",
  "sklearn_version": "1.9.1"
}
```

The preprocessor travels with the model, so the API accepts the 12 raw
features and applies exactly the scaling/selection used in training.

## 6. Monitoring and drift

Prediction log — only operational fields are written (`SAFE_LOG_KEYS`); keys
that look sensitive (user, e-mail, tokens, raw features…) are dropped first:

```json
{"timestamp": "2026-11-06T09:41:12+00:00", "latency": 0.002241,
 "prediction": 1, "confidence": 0.613, "model_version": "logistic_regression-v1-2232"}
```

`src/monitoring/drift_detection.py` offers `detect_drift_ks` (two-sample KS
test), `detect_drift_mean_shift` (relative mean change) and
`detect_feature_drift` (per-feature combination; `drift_score` = share of
drifted features). Alert when `drift_score > 0.3`.

v1 → v2 result: **drift detected**, `drift_score = 0.3333` (4/12 features),
severity **medium**. Top features: `support_tickets` (+108.5%),
`monthly_charges` (+23.5%), `total_charges` (+31.6%), `credit_score` (−3.3%).
Recommended action: check the upstream pipeline for those features, compare
offline vs serving F1 and schedule a retrain on the current distribution.

## 7. Serving contract

| endpoint        | description                                                       |
| --------------- | ----------------------------------------------------------------- |
| `GET /health`   | `status`, `model_version`, `data_version`, `model_loaded`, uptime |
| `POST /predict` | 12 raw features → `prediction`, `confidence`, `latency`; logs it  |
| `GET /metadata` | metadata of the served artifact                                   |

Wrong feature count → `400`, non-numeric payload → `422`, missing model →
`503`. The response never echoes raw features.

## 8. Docker image

`python:3.12-slim`, installs only `requirements-serve.txt`, runs as UID 10001
(`appuser`), exposes 8000 and declares a `HEALTHCHECK` on `/health`. The image
copies `models/`, so train before building it.

## 9. CI/CD

`.github/workflows/ci.yml`: `code-quality` (ruff lint + format) → `test`
(pytest) → `train` (`run_experiments.py`, uploads models/results/mlflow.db) →
`drift-check` (fails on alert) → `docker-smoke` (build, run, wait for
`/health`, then `scripts/healthcheck.py` verifies the prediction contract and
input validation).

## 10. Known limitations

- 400 rows is small: the svm/lr F1 gap is inside noise on an 80-row test split.
- `SVC(probability=True)` raises a sklearn 1.9 `FutureWarning` (deprecated in
  favour of `CalibratedClassifierCV`); it still works, pinned to 1.9.1.
- The drift reference is a static snapshot; production should use a rolling
  window over the last N served samples.
- MLflow uses a local sqlite file — great for reproducibility, not a shared
  server.

---

## 11. Requirements

Python 3.11+ (built and tested on 3.14) and, for the Docker step only, Docker.

```bash
cd Project
python3 -m venv .venv && source .venv/bin/activate     # optional but recommended
pip install -r requirements.txt
```

## 12. How to run and test

All commands run from inside `Project/`.

### Step 1 — verify the code (tests, no training)

```bash
pytest tests/ -q                    # 63 tests, ~5 s
pytest tests/ -v                    # same, with test names
pytest tests/test_data.py -q        # a single module
```

Expected last line: `63 passed`. The tests read the two CSVs in `data/raw/`.

### Step 2 — train the model and print the metrics

```bash
python experiments/train_experiment.py
```

This single entry point runs the whole pipeline (load → split → scale → select
→ train → evaluate → save → MLflow) and prints:

```
accuracy   : 0.6125
f1_score   : 0.4746   <-- primary metric
v2 eval    : accuracy=0.6188 f1=0.6839 (distribution shift)
artifact   : .../Project/models/model.joblib
mlflow id  : dff75f2800884efd9710bb003e140862
```

It writes `models/model.joblib`, `models/preprocessor.joblib`,
`models/metadata.json`, a new entry in `data_versions/metadata.json` and one
run in `mlflow.db`. Any config value can be overridden for a single run:

```bash
python experiments/train_experiment.py --model-type svm
python experiments/train_experiment.py --model-type random_forest --scaling none
python experiments/train_experiment.py --data-file dataset_v2.csv --data-version v2
python experiments/train_experiment.py --no-mlflow          # skip tracking
```

### Step 3 — the batch of tracked experiments

```bash
python experiments/run_experiments.py
```

Runs 5 configurations (different data version / model / hyperparameter), logs
each to MLflow and writes `experiments/results/comparison.md` and
`comparison.json`.

### Step 4 — inspect the tracked runs

```bash
mlflow ui --backend-store-uri sqlite:///$(pwd)/mlflow.db    # http://localhost:5000
```

### Step 5 — drift report

```bash
python scripts/drift_report.py                  # writes reports/drift_report.{json,md}
python scripts/drift_report.py --fail-on-drift  # exit 1 when drift_score > 0.3
```

### Step 6 — serve the model and call it

```bash
uvicorn src.serving.app:app --port 8000
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' \
     -d '{"features":[22.85,74.43,1570.81,1,2,28.11,12.89,12,8.43,698,46,3]}'
# {"prediction":1,"confidence":0.581,"latency":0.023}
```

Every call appends one line to `experiments/logs/predictions_YYYY-MM-DD.json`.

### Step 7 — Docker

```bash
docker build -t mlops-serving:hw01 .
docker run -p 8000:8000 mlops-serving:hw01
python scripts/healthcheck.py --base-url http://127.0.0.1:8000
```
