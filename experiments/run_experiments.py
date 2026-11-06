"""Run a batch of tracked experiments (>= 3 MLflow runs) and compare them.

Each run differs in at least one of: data version, model type or
hyperparameter - which is exactly what the assignment asks for.

    python experiments/run_experiments.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.train_experiment import parse_args, run_pipeline

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}
LR_PARAMS = {"C": 1.0, "max_iter": 2000, "class_weight": "balanced", "random_state": 42}
SVM_PARAMS = {
    "C": 1.0,
    "kernel": "rbf",
    "gamma": "scale",
    "class_weight": "balanced",
    "probability": True,
    "random_state": 42,
}

#: name -> overrides, plus what makes the run different from the reference
EXPERIMENTS: list[dict[str, Any]] = [
    {
        "run_name": "lr_v1_standard",
        "overrides": {
            "model": {"type": "logistic_regression", "params": LR_PARAMS},
            "features": {"scaling": "standard", "feature_selection_k": 10},
        },
        "differs_by": "reference: logistic_regression on data v1, standard scaling",
    },
    {
        "run_name": "rf_v1_none",
        "overrides": {
            "model": {"type": "random_forest", "params": RF_PARAMS},
            "features": {"scaling": "none", "feature_selection_k": 10},
        },
        "differs_by": "model type: random_forest (trees do not need scaling)",
    },
    {
        "run_name": "svm_v1_standard",
        "overrides": {
            "model": {"type": "svm", "params": SVM_PARAMS},
            "features": {"scaling": "standard", "feature_selection_k": 10},
        },
        "differs_by": "model type + hyperparameter: svm (rbf kernel)",
    },
    {
        "run_name": "lr_v1_k8",
        "overrides": {
            "model": {"type": "logistic_regression", "params": LR_PARAMS},
            "features": {"scaling": "standard", "feature_selection_k": 8},
        },
        "differs_by": "hyperparameter: feature_selection_k=8 instead of 10",
    },
    {
        "run_name": "lr_v2_standard",
        "overrides": {
            "data": {"dataset_file": "dataset_v2.csv", "dataset_version": "v2"},
            "model": {"type": "logistic_regression", "params": LR_PARAMS},
            "features": {"scaling": "standard", "feature_selection_k": 10},
        },
        "differs_by": "data version: v2 (480 rows, shifted distribution)",
    },
]


def render_comparison(summaries: list[dict[str, Any]], path: Path) -> str:
    """Write a Markdown comparison table and return it as text."""
    lines = [
        "# Experiment Comparison",
        "",
        "| Run | Data | Model | Scaling | k | Accuracy | F1 | Drift-set F1 | MLflow run |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    ranked = sorted(summaries, key=lambda s: s["metrics"]["f1_score"], reverse=True)
    for item in ranked:
        metrics = item["metrics"]
        drift_f1 = metrics.get("drift_set_f1_score")
        lines.append(
            "| {run} | {data} | {model} | {scaling} | {k} | {acc:.4f} | {f1:.4f} | {df1} | {rid} |".format(
                run=item["run_name"],
                data=item["data_version"],
                model=item["model_type"],
                scaling=item["scaling"],
                k=item["feature_selection_k"],
                acc=metrics["accuracy"],
                f1=metrics["f1_score"],
                df1=f"{drift_f1:.4f}" if isinstance(drift_f1, (int, float)) else "-",
                rid=(item.get("mlflow_run_id") or "-")[:12],
            )
        )

    best = ranked[0]
    lines += [
        "",
        (
            f"**Best run by F1:** `{best['run_name']}` "
            f"(F1 = {best['metrics']['f1_score']:.4f}, accuracy = {best['metrics']['accuracy']:.4f})."
        ),
        "",
        (
            "`Drift-set F1` is the score of each model on the second data version "
            "(v2), which is never used for training. A large drop there is an early "
            "warning that the deployed model will degrade."
        ),
        "",
        (
            "Every row above is a separate MLflow run; at least one of data version, "
            "model type or hyperparameters differs between them."
        ),
        "",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    args = parse_args(argv)
    summaries: list[dict[str, Any]] = []

    for experiment in EXPERIMENTS:
        print("\n" + "#" * 68)
        print(
            f"# Running experiment: {experiment['run_name']}  ({experiment['differs_by']})"
        )
        print("#" * 68)
        summaries.append(
            run_pipeline(
                config_path=args.config,
                overrides=experiment["overrides"],
                run_name=experiment["run_name"],
                use_mlflow=not args.no_mlflow,
            )
        )

    results_dir = PROJECT_ROOT / "experiments" / "results"
    comparison_path = results_dir / "comparison.md"
    table = render_comparison(summaries, comparison_path)

    with (results_dir / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(
            [
                {
                    "run_name": s["run_name"],
                    "data_version": s["data_version"],
                    "model_type": s["model_type"],
                    "model_params": s["model_params"],
                    "scaling": s["scaling"],
                    "feature_selection_k": s["feature_selection_k"],
                    "metrics": s["metrics"],
                    "mlflow_run_id": s["mlflow_run_id"],
                }
                for s in summaries
            ],
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + table)
    print(f"Comparison written to: {comparison_path}")
    return summaries


if __name__ == "__main__":
    main()
