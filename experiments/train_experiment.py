"""Config-driven training experiment.

    python experiments/train_experiment.py --model-type svm --scaling standard

Steps: load config -> load data -> version data -> split -> scale -> select
features -> train -> evaluate (test + drift set) -> save artifacts -> log to
MLflow -> print a summary.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # allow `python experiments/train_experiment.py`
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import (
    create_data_version,
    load_config,
    load_csv,
    resolve_path,
    split_data,
    utc_now,
    validate_schema,
)
from src.features.build_features import (
    build_feature_pipeline,
    transform_features,
)
from src.models.train_model import (
    evaluate_model,
    save_model,
    train_model,
)

DEFAULTS: dict[str, Any] = {
    "data": {
        "raw_path": "data/raw",
        "dataset_file": "dataset_v1.csv",
        "dataset_version": "v1",
        "drift_dataset_file": "dataset_v2.csv",
        "drift_dataset_version": "v2",
        "target_column": "churn",
        "test_size": 0.2,
        "random_state": 42,
        "stratify": True,
        "versions_dir": "data_versions",
    },
    "features": {"scaling": "standard", "feature_selection_k": 10},
    "model": {"type": "random_forest", "params": {}, "output_dir": "models"},
    "experiments": {
        "mlflow_tracking_uri": "sqlite:///mlflow.db",
        "experiment_name": "mlops_homework_01",
        "results_dir": "experiments/results",
    },
}


#: Keys whose value is replaced instead of merged. Hyperparameters are
#: model specific, so inheriting another model's keys (e.g. ``C`` for a random
#: forest) would be a bug, not a useful default.
REPLACE_KEYS = frozenset({"params"})


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overrides`` into a copy of ``base``."""
    merged = dict(base)
    for key, value in (overrides or {}).items():
        if key in REPLACE_KEYS:
            merged[key] = value
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _dataset_path(config: dict[str, Any], key: str = "dataset_file") -> Path:
    data_cfg = config["data"]
    return resolve_path(Path(data_cfg["raw_path"]) / data_cfg[key])


def _resolve_tracking_uri(uri: str) -> str:
    """Make relative sqlite tracking URIs absolute so runs land in the repo."""
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        relative = uri[len("sqlite:///") :]
        if not Path(relative).is_absolute():
            return f"sqlite:///{(PROJECT_ROOT / relative).as_posix()}"
    if uri.startswith("file:") and not uri.startswith("file:/"):
        return f"file:{(PROJECT_ROOT / uri[len('file:') :]).as_posix()}"
    return uri


def _log_to_mlflow(
    run_name: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    params: dict[str, Any],
    artifact_paths: list[Path],
    tags: dict[str, str],
) -> str | None:
    """Log params/metrics/artifacts to MLflow; returns the run id (or None)."""
    try:
        import mlflow
    except ImportError:  # pragma: no cover - mlflow is a hard requirement
        print("[warn] mlflow is not installed, skipping experiment tracking")
        return None

    tracking_uri = _resolve_tracking_uri(config["experiments"]["mlflow_tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["experiments"]["experiment_name"])

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {k: v for k, v in params.items() if isinstance(v, (str, int, float, bool))}
        )
        mlflow.log_metrics(
            {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        )
        mlflow.set_tags(tags)
        for path in artifact_paths:
            if path and Path(path).exists():
                mlflow.log_artifact(str(path))
        return run.info.run_id


def run_pipeline(
    config_path: str = "config/config.yaml",
    overrides: dict[str, Any] | None = None,
    run_name: str | None = None,
    use_mlflow: bool = True,
) -> dict[str, Any]:
    """Execute the whole pipeline and return a machine-readable summary."""
    started = time.perf_counter()

    print(f"[1/8] Loading config: {config_path}")
    config = deep_merge(DEFAULTS, load_config(config_path))
    config = deep_merge(config, overrides or {})

    data_cfg = config["data"]
    feature_cfg = config["features"]
    model_cfg = config["model"]
    target = data_cfg["target_column"]

    print(f"[2/8] Loading data: {_dataset_path(config)}")
    df = load_csv(_dataset_path(config))
    validate_schema(df, target_column=target)

    print(f"[3/8] Registering data version '{data_cfg['dataset_version']}'")
    version_entry = create_data_version(
        df,
        version_id=data_cfg["dataset_version"],
        description=f"Training dataset {data_cfg['dataset_file']} "
        f"({len(df)} rows, {df.shape[1] - 1} features)",
        output_dir=data_cfg["versions_dir"],
    )

    drift_df = None
    drift_path = _dataset_path(config, "drift_dataset_file")
    if drift_path.exists():
        drift_df = load_csv(drift_path)
        validate_schema(drift_df, target_column=target)
        create_data_version(
            drift_df,
            version_id=data_cfg["drift_dataset_version"],
            description=f"Drift / holdout dataset {data_cfg['drift_dataset_file']} "
            f"({len(drift_df)} rows, distribution shifted)",
            output_dir=data_cfg["versions_dir"],
        )

    print(
        f"[4/8] Splitting data (test_size={data_cfg['test_size']}, "
        f"random_state={data_cfg['random_state']})"
    )
    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column=target,
        test_size=float(data_cfg["test_size"]),
        random_state=int(data_cfg["random_state"]),
        stratify=bool(data_cfg.get("stratify", True)),
    )

    print(
        f"[5/8] Building features (scaling={feature_cfg['scaling']}, "
        f"k={feature_cfg['feature_selection_k']})"
    )
    bundle = build_feature_pipeline(
        X_train,
        y_train,
        X_test,
        scaling=feature_cfg["scaling"],
        k=int(feature_cfg["feature_selection_k"]),
    )

    print(
        f"[6/8] Training {model_cfg['type']} on "
        f"{bundle['X_train'].shape[0]} samples x {bundle['X_train'].shape[1]} features"
    )
    train_started = time.perf_counter()
    model = train_model(
        bundle["X_train"],
        y_train,
        model_cfg["type"],
        model_cfg.get("params", {}),
    )
    train_seconds = time.perf_counter() - train_started

    print("[7/8] Evaluating")
    metrics = evaluate_model(model, bundle["X_test"], y_test)
    metrics["train_seconds"] = round(train_seconds, 4)

    # Robustness check: how does the v1-trained model behave on the drifted v2?
    if drift_df is not None:
        drift_bundle = transform_features(
            drift_df.drop(columns=[target]), bundle["scaler"], bundle["selector"]
        )
        drift_metrics = evaluate_model(model, drift_bundle, drift_df[target])
        metrics["drift_set_accuracy"] = drift_metrics["accuracy"]
        metrics["drift_set_f1_score"] = drift_metrics["f1_score"]
    else:
        drift_metrics = {}

    print("[8/8] Saving artifacts + logging to MLflow")
    model_version = f"{model_cfg['type']}-{data_cfg['dataset_version']}-{int(started)}"
    preprocessor = {
        "scaler": bundle["scaler"],
        "selector": bundle["selector"],
        "input_features": bundle["input_features"],
        "selected_features": bundle["selected_features"],
        "scaling_method": bundle["scaling_method"],
    }
    model_path = save_model(
        model,
        metrics,
        data_version=data_cfg["dataset_version"],
        model_version=model_version,
        output_dir=model_cfg["output_dir"],
        preprocessor=preprocessor,
        metadata_extra={
            "feature_selection_k": int(feature_cfg["feature_selection_k"]),
            "test_size": float(data_cfg["test_size"]),
            "random_state": int(data_cfg["random_state"]),
            "data_hash": version_entry["hash"],
        },
    )

    results_dir = resolve_path(config["experiments"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    run_name = (
        run_name
        or f"{model_cfg['type']}_k{feature_cfg['feature_selection_k']}_{data_cfg['dataset_version']}"
    )
    summary: dict[str, Any] = {
        "run_name": run_name,
        "created_at": utc_now(),
        "model_type": model_cfg["type"],
        "model_params": model_cfg.get("params", {}),
        "data_version": data_cfg["dataset_version"],
        "data_file": data_cfg["dataset_file"],
        "data_hash": version_entry["hash"],
        "n_rows": len(df),
        "n_features_input": len(bundle["input_features"]),
        "selected_features": bundle["selected_features"],
        "feature_scores": bundle["feature_scores"],
        "scaling": feature_cfg["scaling"],
        "feature_selection_k": int(feature_cfg["feature_selection_k"]),
        "metrics": metrics,
        "drift_set_metrics": drift_metrics,
        "model_version": model_version,
        "model_path": str(model_path),
        "train_seconds": round(train_seconds, 4),
        "mlflow_run_id": None,
    }

    summary_path = results_dir / f"{run_name}.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    if use_mlflow:
        flat_params: dict[str, Any] = {
            "model_type": model_cfg["type"],
            "data_version": data_cfg["dataset_version"],
            "data_hash": version_entry["hash"],
            "data_file": data_cfg["dataset_file"],
            "scaling": feature_cfg["scaling"],
            "feature_selection_k": int(feature_cfg["feature_selection_k"]),
            "test_size": float(data_cfg["test_size"]),
            "random_state": int(data_cfg["random_state"]),
            "n_rows": len(df),
            "n_selected_features": len(bundle["selected_features"]),
        }
        for key, value in (model_cfg.get("params") or {}).items():
            flat_params[f"model_param_{key}"] = value

        metadata_path = Path(model_cfg["output_dir"]) / "metadata.json"
        preprocessor_path = Path(model_cfg["output_dir"]) / "preprocessor.joblib"
        summary["mlflow_run_id"] = _log_to_mlflow(
            run_name=run_name,
            config=config,
            metrics=metrics,
            params=flat_params,
            artifact_paths=[
                Path(model_path),
                resolve_path(metadata_path),
                resolve_path(preprocessor_path),
                summary_path,
            ],
            tags={
                "data_version": str(data_cfg["dataset_version"]),
                "model_type": str(model_cfg["type"]),
                "data_hash": version_entry["hash"],
            },
        )
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    print("\n" + "=" * 68)
    print(f"run        : {run_name}")
    print(f"model      : {model_cfg['type']} {model_cfg.get('params', {})}")
    print(
        f"data       : {data_cfg['dataset_version']} ({len(df)} rows, hash {version_entry['hash'][:12]}...)"
    )
    print(f"features   : {bundle['input_features']} -> {bundle['selected_features']}")
    print(f"accuracy   : {metrics['accuracy']:.4f}")
    print(f"f1_score   : {metrics['f1_score']:.4f}   <-- primary metric")
    if drift_metrics:
        print(
            f"v2 eval    : accuracy={drift_metrics['accuracy']:.4f} "
            f"f1={drift_metrics['f1_score']:.4f} (distribution shift)"
        )
    print(f"artifact   : {model_path}")
    print(f"mlflow id  : {summary['mlflow_run_id']}")
    print("=" * 68)

    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MLOps training pipeline")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--model-type", choices=["random_forest", "logistic_regression", "svm"]
    )
    parser.add_argument("--scaling", choices=["standard", "minmax", "none"])
    parser.add_argument("--k", type=int, help="number of features to keep")
    parser.add_argument("--data-file", help="CSV filename inside data/raw")
    parser.add_argument("--data-version", help="version id to register (e.g. v1, v2)")
    parser.add_argument("--run-name")
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)

    overrides: dict[str, Any] = {"model": {}, "features": {}, "data": {}}
    if args.model_type:
        overrides["model"]["type"] = args.model_type
        if args.model_type == "logistic_regression":
            overrides["model"]["params"] = {
                "C": 1.0,
                "max_iter": 2000,
                "class_weight": "balanced",
                "random_state": 42,
            }
        elif args.model_type == "svm":
            overrides["model"]["params"] = {
                "C": 1.0,
                "kernel": "rbf",
                "gamma": "scale",
                "class_weight": "balanced",
                "probability": True,
                "random_state": 42,
            }
        else:
            overrides["model"]["params"] = {
                "n_estimators": 300,
                "max_depth": 6,
                "min_samples_leaf": 2,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
            }
    if args.scaling:
        overrides["features"]["scaling"] = args.scaling
    if args.k is not None:
        overrides["features"]["feature_selection_k"] = args.k
    if args.data_file:
        overrides["data"]["dataset_file"] = args.data_file
    if args.data_version:
        overrides["data"]["dataset_version"] = args.data_version

    overrides = {section: value for section, value in overrides.items() if value}
    return run_pipeline(
        config_path=args.config,
        overrides=overrides,
        run_name=args.run_name,
        use_mlflow=not args.no_mlflow,
    )


if __name__ == "__main__":
    main()
