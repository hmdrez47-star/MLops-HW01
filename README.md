# MLOps Homework 01 — Churn Prediction

A reproducible, modular, versioned ML project for the telecom **customer churn**
binary-classification task: config-driven training pipeline, MLflow experiment
tracking, data versioning, pytest suite, FastAPI serving, Docker image, drift
monitoring and a CI/CD pipeline.

- **Course:** Daneshkar AI Bootcamp — MLOps
- **Exercise:** HW_MLOps_01
- **Primary metric:** `f1_score` (the classes are imbalanced: 26% churn)

> Persian guide: see **[HINT_FA.md](HINT_FA.md)** for a complete step-by-step
> explanation of what was built and why.

---

## 1. Project structure

```
ANSWER/
├── config/config.yaml              # every parameter (no hard-coded values)
├── data/
│   ├── raw/                        # dataset_v1.csv (400 rows), dataset_v2.csv (480)
│   └── processed/                  # reserved for processed artifacts
├── data_versions/
│   ├── metadata.json               # version_id, hash, description, n_rows, created_at
│   ├── v1.csv / v2.csv             # frozen snapshots (gitignored)
├── src/
│   ├── data/load_data.py           # load_csv, split_data, compute_hash, versioning
│   ├── features/build_features.py  # scale_features, select_features, pipeline
│   ├── models/train_model.py       # get_model, train_model, evaluate_model, save_model
│   ├── monitoring/drift_detection.py  # JSON prediction logs, KS + mean shift, report
│   └── serving/app.py              # FastAPI: /health, /predict, /metadata
├── experiments/
│   ├── train_experiment.py         # config → data → features → train → eval → save → MLflow
│   ├── run_experiments.py          # batch of 5 tracked runs + comparison table
│   ├── logs/                       # predictions_YYYY-MM-DD.json (JSONL)
│   └── results/                    # per-run summary + comparison.md/json
├── tests/                          # pytest: data, features, models, monitoring, serving
├── scripts/
│   ├── drift_report.py             # drift report CLI (CI gate)
│   └── healthcheck.py              # post-deployment contract check
├── models/                         # model.joblib, preprocessor.joblib, metadata.json
├── reports/                        # drift_report.json + drift_report.md
├── Dockerfile / requirements-serve.txt
├── requirements.txt                # pinned full environment
└── .github/workflows/ci.yml        # quality → test → train → drift → docker smoke
```

## 2. Dataset

`churn` is the target (`0` = stayed, `1` = left) and there are 12 numeric
features: `tenure_months`, `monthly_charges`, `total_charges`, `num_services`,
`support_tickets`, `avg_session_minutes`, `login_frequency`, `contract_length`,
`payment_delay_days`, `credit_score`, `age`, `num_dependents`.

| version | rows | churn rate | purpose |
|---|---|---|---|
| `dataset_v1.csv` | 400 | 26.3% | reference training data |
| `dataset_v2.csv` | 480 | 48.1% | second version with distribution shift (drift testing) |

Registered fingerprints (`data_versions/metadata.json`):

| version_id | sha256 |
|---|---|
| v1 | `446b605d500600642b35459724d0a87fc0e47d25eba9167321acd29fb931ad7a` |
| v2 | `6bd52cc55b6ceaebee11d2123ad2dac234aaea46d713e42a80a95c60996f8162` |

## 3. Preprocessing

1. **Load** the CSV, separate `churn` → `y` and the 12 remaining columns → `X`.
2. **Split** `test_size=0.2`, `random_state=42`, stratified on the target.
3. **Scale** with `StandardScaler` fitted on the training split only
   (`none` is available for tree models).
4. **Select** the best `k=10` features with `SelectKBest(f_classif)`, also
   fitted on the training split only → `num_services` and `age` are dropped.
5. **Train** the model configured in `config.yaml`.
6. **Evaluate** with accuracy + F1 (F1 is the primary metric).

Because only ~26% of the rows are positive, every model uses
`class_weight="balanced"`. Without it the random forest predicts the majority
class and F1 collapses to ~0.08 while accuracy still looks like 0.70 — a clear
example of why F1 matters here.

## 4. Configuration

All parameters live in `config/config.yaml`: paths, `test_size`,
`random_state`, scaling method, `feature_selection_k`, model type and
hyperparameters, MLflow URI/experiment name, drift thresholds and serving port.
CLI flags override single values, e.g.

```bash
python experiments/train_experiment.py --model-type svm --scaling standard --k 10
python experiments/train_experiment.py --data-file dataset_v2.csv --data-version v2
```

Hyperparameter dicts are **replaced** (not merged) on override, so switching
model type never leaks another model's arguments.

## 5. Reproduce

```bash
cd ANSWER
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python experiments/train_experiment.py          # single reference run
python experiments/run_experiments.py           # 5 tracked runs + comparison
python scripts/drift_report.py                  # drift report → reports/
pytest tests/ -v                                # 63 tests
```

## 6. Results (MLflow: experiment `mlops_homework_01`, 6 runs)

| Run | Data | Model | Scaling | k | Accuracy | F1 | v2-set F1 |
|---|---|---|---|---|---|---|---|
| `lr_v2_standard` | v2 | logistic_regression | standard | 10 | 0.6979 | **0.6947** | 0.6848 |
| `svm_v1_standard` | v1 | svm (rbf) | standard | 10 | 0.6875 | 0.5098 | 0.6283 |
| `lr_v1_k8` | v1 | logistic_regression | standard | 8 | 0.6500 | 0.5000 | 0.6927 |
| `lr_v1_standard` | v1 | logistic_regression | standard | 10 | 0.6125 | 0.4746 | 0.6839 |
| `lr_v1_deployed` | v1 | logistic_regression | standard | 10 | 0.6125 | 0.4746 | 0.6839 |
| `rf_v1_none` | v1 | random_forest | none | 10 | 0.5250 | 0.2083 | 0.6695 |

The best score overall comes from the **v2** run, but that is not a fair
comparison: v2 is a different (easier, 48% positive) distribution. On the
reference v1 data the ranking is `svm (0.5098) > lr (0.4746) > rf (0.2083)`.
The deployed artifact is the `v1` logistic regression because it is the config
default, needs no probability calibration and its coefficients are
interpretable; switching the served model is a one-line config change followed
by a retrain.

`Drift-set F1` in `experiments/results/comparison.md` is each model's score on
v2, which is never seen during training. v1 models score *higher* on v2 purely
because v2 is easier — a reminder that a single holdout is not a drift monitor,
which is why the KS/mean-shift report below exists.

## 7. Experiment tracking

```bash
mlflow ui --backend-store-uri sqlite:///$(pwd)/mlflow.db   # http://localhost:5000
```

Each run logs: **parameters** (model type, data version + data hash, scaling,
k, test size, random state, every model hyperparameter), **metrics**
(accuracy, F1, precision, recall, train time, v2-set metrics), **tags**
(data version, model type, data hash) and **artifacts**
(`model.joblib`, `preprocessor.joblib`, `metadata.json`, run summary).
The tracking URI is local sqlite, so tracking works offline and the run history
is reproducible; set `experiments.mlflow_tracking_uri` to
`http://localhost:5000` to use a server instead.

## 8. Data versioning

`src/data/load_data.create_data_version` writes one entry per version into
`data_versions/metadata.json` (`version_id`, `hash`, `description`, `n_rows`,
`created_at`) and freezes a CSV snapshot. `verify_data_version` recomputes the
SHA-256 and proves a dataset still matches its recorded fingerprint, so a model
(`metadata.json → data_hash`) is always traceable back to exact bytes.

## 9. Model artifact

`models/model.joblib`, `models/preprocessor.joblib` and `models/metadata.json`:

```json
{
  "model_version": "logistic_regression-v1-2232",
  "created_at": "2026-11-06T09:39:39+00:00",
  "data_version": "v1",
  "metric": {"name": "f1_score", "value": 0.4746},
  "metrics": {"accuracy": 0.6125, "f1_score": 0.4746, "precision": 0.3684, "recall": 0.6667},
  "model_type": "LogisticRegression",
  "input_features": ["tenure_months", "..."],
  "selected_features": ["tenure_months", "..."],
  "scaling_method": "standard",
  "data_hash": "446b605d5006...",
  "sklearn_version": "1.9.1",
  "git_commit": "<commit>"
}
```

The preprocessor travels with the model, so the API accepts the 12 raw features
and applies exactly the scaling/selection used in training.

## 10. Monitoring & drift

**Prediction log** — one JSON object per line in
`experiments/logs/predictions_YYYY-MM-DD.json`:

```json
{"timestamp": "2026-11-06T09:41:12.418+00:00", "latency": 0.002241, "prediction": 1, "confidence": 0.613, "model_version": "logistic_regression-v1-2232"}
```

Only operational fields are allowed (`SAFE_LOG_KEYS`); anything that looks
sensitive (user id, e-mail, raw features, tokens…) is dropped before writing.

**Drift** — `detect_drift_ks` (two-sample KS test), `detect_drift_mean_shift`
(relative mean change) and `detect_feature_drift` (per-feature combination,
`drift_score` = share of features that drifted). Alert when
`drift_score > 0.3`.

`reports/drift_report.json` / `.md` compare v1 → v2:

| feature | KS statistic | p-value | relative shift |
|---|---|---|---|
| support_tickets | 0.4029 | <0.0001 | +108.5% |
| monthly_charges | 0.4017 | <0.0001 | +23.5% |
| credit_score | 0.1883 | <0.0001 | −3.3% |
| total_charges | 0.1721 | <0.0001 | +31.6% |

**Result:** drift detected, `drift_score = 0.3333` (4/12 features) > `0.3`,
severity **medium**. Recommended action: investigate the upstream pipeline for
those four features, compare offline vs serving F1 and schedule a retrain on
the current distribution. `scripts/drift_report.py --fail-on-drift` exits 1,
which is the CI gate.

## 11. Serving

```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' \
     -d '{"features":[22.85,74.43,1570.81,1,2,28.11,12.89,12,8.43,698,46,3]}'
# {"prediction":0,"confidence":0.62,"latency":0.0019}
```

| endpoint | description |
|---|---|
| `GET /health` | `status`, `model_version`, `data_version`, `model_loaded`, uptime |
| `POST /predict` | 12 raw features → `prediction`, `confidence`, `latency`; logs the call |
| `GET /metadata` | metadata of the served artifact |

Wrong feature counts return `400`, non-numeric payloads `422`, a missing model
`503`. The response never echoes raw features.

## 12. Docker

```bash
docker build -t mlops-serving:hw01 .
docker run -p 8000:8000 mlops-serving:hw01
python scripts/healthcheck.py --base-url http://127.0.0.1:8000
```

Image is `python:3.12-slim`, installs only `requirements-serve.txt`, runs as
UID 10001 (`appuser`), exposes 8000 and declares a `HEALTHCHECK` on `/health`.
Train before building: the image copies `models/`.

## 13. CI/CD (`.github/workflows/ci.yml`)

`code-quality` (ruff lint + format) → `test` (pytest) → `train`
(`run_experiments.py`, uploads models/results/mlflow.db) → `drift-check`
(report, fails on alert) → `docker-smoke` (build, run container, wait for
`/health`, then `scripts/healthcheck.py` verifies prediction contract and input
validation). Code quality and model health are therefore checked automatically
after every deployment.

## 14. Known limitations

- 400 rows is small: the F1 gap between svm/lr is within noise on an 80-row test
  split; nothing was tuned on the test set beyond a small grid on v1.
- `SVC(probability=True)` triggers a scikit-learn 1.9 `FutureWarning`
  (deprecated in favour of `CalibratedClassifierCV`); it still works and is
  pinned to sklearn 1.9.1.
- Drift reference is a static snapshot; in production it should be a rolling
  window on the last N served samples.
- The MLflow backend is a local sqlite file — great for reproducibility, not a
  shared server.
