"""Monitoring: structured prediction logging and distribution drift detection.

Two independent drift signals are combined per feature:

* **KS test** (``scipy.stats.ks_2samp``) - is the whole distribution different?
* **Mean shift** - did the average move by more than ``mean_shift_threshold``?

The overall ``drift_score`` is the *share of features that drifted* (0.0-1.0),
which is directly comparable with the alert threshold (default ``0.3`` = 30% of
the features). Each feature also gets its own severity so a report can say
exactly which column moved and how strongly.

Privacy: prediction logs may only contain operational fields. Sensitive data
(raw features, identifiers, e-mails, tokens) is dropped before anything is
written to disk.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

#: Only these optional keys may be attached to a prediction log entry.
SAFE_LOG_KEYS = ("model_version", "data_version", "endpoint", "status_code")

#: Keys that must never reach a log file, even if a caller passes them.
SENSITIVE_KEY_MARKERS = (
    "user",
    "email",
    "phone",
    "token",
    "password",
    "secret",
    "feature",
    "raw",
    "ip",
    "name",
    "address",
    "ssn",
    "card",
    "identity",
)

#: Severity boundaries (share of drifted features / KS statistic).
SEVERITY_ORDER = ("none", "low", "medium", "high")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path_like: str | os.PathLike) -> Path:
    path = Path(path_like).expanduser()
    return path if path.is_absolute() else project_root() / path


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


# ---------------------------------------------------------------------------
# Structured prediction logging
# ---------------------------------------------------------------------------
def _is_sensitive(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in SENSITIVE_KEY_MARKERS)


def log_prediction(
    prediction: int,
    confidence: float,
    latency: float,
    log_path: str | os.PathLike = "experiments/logs",
    **extra: Any,
) -> dict[str, Any]:
    """Append one structured JSON log line per prediction.

    The entry always contains ``timestamp``, ``latency``, ``prediction`` and
    ``confidence``. Extra fields are accepted only if they are whitelisted in
    :data:`SAFE_LOG_KEYS`; anything that looks sensitive is silently dropped.

    A daily file (``predictions_YYYY-MM-DD.json``) holds one JSON object per
    line, which is convenient to tail, ship and parse.
    """
    directory = resolve_path(log_path)
    directory.mkdir(parents=True, exist_ok=True)

    entry: dict[str, Any] = {
        "timestamp": utc_now(),
        "latency": round(float(latency), 6),
        "prediction": int(prediction),
        "confidence": round(float(confidence), 4),
    }

    for key, value in extra.items():
        if key in SAFE_LOG_KEYS and not _is_sensitive(key):
            entry[key] = value

    day = datetime.now(UTC).strftime("%Y-%m-%d")
    log_file = directory / f"predictions_{day}.json"

    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return entry


def read_prediction_logs(
    log_path: str | os.PathLike = "experiments/logs",
    day: str | None = None,
) -> list[dict[str, Any]]:
    """Read back the prediction log of a given day (default: today, UTC)."""
    day = day or datetime.now(UTC).strftime("%Y-%m-%d")
    log_file = resolve_path(log_path) / f"predictions_{day}.json"
    if not log_file.exists():
        return []

    entries: list[dict[str, Any]] = []
    with log_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


# ---------------------------------------------------------------------------
# Drift detection primitives
# ---------------------------------------------------------------------------
def _as_vector(data):
    values = np.asarray(data, dtype=float).ravel()
    values = values[~np.isnan(values)]
    if values.size < 2:
        raise ValueError("At least two non-NaN observations are required")
    return values


def detect_drift_ks(
    reference_data,
    current_data,
    threshold: float = 0.05,
) -> dict[str, Any]:
    """Two-sample Kolmogorov-Smirnov test.

    ``threshold`` is the p-value (alpha) below which the null hypothesis
    "both samples come from the same distribution" is rejected.
    """
    reference = _as_vector(reference_data)
    current = _as_vector(current_data)

    statistic, p_value = stats.ks_2samp(reference, current)

    return {
        "method": "ks_test",
        "statistic": round(float(statistic), 6),
        "p_value": round(float(p_value), 8),
        "threshold": float(threshold),
        "drift_detected": bool(p_value < threshold),
    }


def detect_drift_mean_shift(
    reference_data,
    current_data,
    threshold: float = 0.1,
) -> dict[str, Any]:
    """Relative change of the mean between two samples.

    ``threshold=0.1`` means "alert when the mean moved by more than 10%".
    """
    reference = _as_vector(reference_data)
    current = _as_vector(current_data)

    reference_mean = float(np.mean(reference))
    current_mean = float(np.mean(current))
    absolute_shift = float(current_mean - reference_mean)
    denominator = abs(reference_mean) if abs(reference_mean) > 1e-12 else 1.0
    relative_shift = abs(absolute_shift) / denominator

    return {
        "method": "mean_shift",
        "reference_mean": round(reference_mean, 6),
        "current_mean": round(current_mean, 6),
        "absolute_shift": round(absolute_shift, 6),
        "relative_shift": round(relative_shift, 6),
        "threshold": float(threshold),
        "drift_detected": bool(relative_shift > threshold),
    }


def _feature_severity(ks_statistic: float, feature_drift_score: float) -> str:
    if feature_drift_score <= 0:
        return "none"
    if ks_statistic >= 0.3 or feature_drift_score >= 0.5:
        return "high"
    if ks_statistic >= 0.15 or feature_drift_score >= 0.2:
        return "medium"
    return "low"


def _overall_severity(drift_share: float, max_ks: float) -> str:
    if drift_share <= 0:
        return "none"
    if drift_share >= 0.5 or max_ks >= 0.5:
        return "high"
    if drift_share >= 0.25 or max_ks >= 0.3:
        return "medium"
    return "low"


def detect_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    ks_threshold: float = 0.05,
    mean_shift_threshold: float = 0.1,
    features: list[str] | None = None,
) -> dict[str, Any]:
    """Run KS + mean-shift detection for every shared numeric feature.

    Returns a per-feature breakdown plus an aggregated``drift_score`` defined as
    the share of features that drifted.
    """
    if features is None:
        features = [c for c in reference_df.columns if c in current_df.columns]
        features = [
            c
            for c in features
            if pd.api.types.is_numeric_dtype(reference_df[c])
            and pd.api.types.is_numeric_dtype(current_df[c])
        ]

    missing = [c for c in features if c not in reference_df or c not in current_df]
    if missing:
        raise ValueError(f"Features missing in one of the datasets: {missing}")
    if not features:
        raise ValueError("No comparable numeric features were provided")

    per_feature: dict[str, Any] = {}
    drifted: list[str] = []

    for column in features:
        ks_result = detect_drift_ks(
            reference_df[column], current_df[column], ks_threshold
        )
        mean_result = detect_drift_mean_shift(
            reference_df[column], current_df[column], mean_shift_threshold
        )

        feature_score = max(
            ks_result["statistic"],
            min(1.0, mean_result["relative_shift"]),
        )
        is_drifted = bool(ks_result["drift_detected"] or mean_result["drift_detected"])

        per_feature[column] = {
            "ks_test": ks_result,
            "mean_shift": mean_result,
            "drift_score": round(feature_score, 4),
            "severity": _feature_severity(ks_result["statistic"], feature_score),
            "drift_detected": is_drifted,
        }
        if is_drifted:
            drifted.append(column)

    n_features = len(features)
    drift_share = len(drifted) / n_features
    mean_ks = float(np.mean([per_feature[c]["ks_test"]["statistic"] for c in features]))
    max_ks = float(max(per_feature[c]["ks_test"]["statistic"] for c in features))

    return {
        "method": "ks_test+mean_shift",
        "ks_p_value_threshold": float(ks_threshold),
        "mean_shift_threshold": float(mean_shift_threshold),
        "n_features": n_features,
        "n_drifted": len(drifted),
        "drift_score": round(drift_share, 4),
        "drift_share": round(drift_share, 4),
        "mean_ks_statistic": round(mean_ks, 4),
        "max_ks_statistic": round(max_ks, 4),
        "severity": _overall_severity(drift_share, max_ks),
        "drifted_features": drifted,
        "feature_drift": per_feature,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _recommended_action(
    drifted_features: list[str], drift_detected: bool, severity: str
) -> str:
    if not drift_detected:
        return (
            "No action required. Keep the scheduled KS/mean-shift check running "
            "and refresh the reference window after the next retraining."
        )

    features = ", ".join(drifted_features)
    if severity == "high":
        return (
            f"Retrain the model on fresh data now (drifted features: {features}), "
            "re-validate offline F1, then roll out the new artifact and keep the "
            "previous version for rollback. Audit the upstream pipeline for the "
            "listed columns before switching traffic."
        )
    return (
        f"Investigate the upstream pipeline of these features: {features}. "
        "Compare recent offline F1 with the serving F1; schedule a retraining on "
        "the current distribution if the metric drops or the drift persists for "
        "more than one monitoring window."
    )


def generate_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    reference_version: str = "v1",
    current_version: str = "v2",
    threshold: float = 0.3,
    ks_p_value_threshold: float = 0.05,
    mean_shift_threshold: float = 0.1,
    features: list[str] | None = None,
    output_dir: str | os.PathLike | None = "reports",
    write_files: bool = True,
) -> dict[str, Any]:
    """Full drift report: did drift happen, where, how strong, what to do.

    ``threshold`` is the alert threshold on ``drift_score`` (share of features
    that drifted).
    """
    detail = detect_feature_drift(
        reference_df,
        current_df,
        ks_threshold=ks_p_value_threshold,
        mean_shift_threshold=mean_shift_threshold,
        features=features,
    )

    drift_score = detail["drift_score"]
    drift_detected = bool(drift_score > threshold)
    severity = (
        detail["severity"] if drift_detected else ("low" if drift_score else "none")
    )

    affected = sorted(
        (name for name in detail["drifted_features"]),
        key=lambda name: detail["feature_drift"][name]["ks_test"]["statistic"],
        reverse=True,
    )
    affected_features = [
        {
            "feature": name,
            "ks_statistic": detail["feature_drift"][name]["ks_test"]["statistic"],
            "p_value": detail["feature_drift"][name]["ks_test"]["p_value"],
            "reference_mean": detail["feature_drift"][name]["mean_shift"][
                "reference_mean"
            ],
            "current_mean": detail["feature_drift"][name]["mean_shift"]["current_mean"],
            "relative_shift": detail["feature_drift"][name]["mean_shift"][
                "relative_shift"
            ],
            "severity": detail["feature_drift"][name]["severity"],
        }
        for name in affected
    ]

    if drift_detected:
        summary = (
            f"Drift detected: {detail['n_drifted']}/{detail['n_features']} features "
            f"({drift_score:.0%}) changed between {reference_version} and "
            f"{current_version}; drift_score={drift_score:.4f} > {threshold}. "
            f"Severity: {severity}."
        )
    else:
        summary = (
            f"No alert: {detail['n_drifted']}/{detail['n_features']} features "
            f"changed (drift_score={drift_score:.4f} <= threshold {threshold}). "
            f"Severity: {severity}."
        )

    report: dict[str, Any] = {
        "generated_at": utc_now(),
        "reference": {"version": reference_version, "n_rows": len(reference_df)},
        "current": {"version": current_version, "n_rows": len(current_df)},
        "method": detail["method"],
        "threshold": float(threshold),
        "ks_p_value_threshold": float(ks_p_value_threshold),
        "mean_shift_threshold": float(mean_shift_threshold),
        "drift_score": drift_score,
        "drift_share": detail["drift_share"],
        "mean_ks_statistic": detail["mean_ks_statistic"],
        "max_ks_statistic": detail["max_ks_statistic"],
        "n_features": detail["n_features"],
        "n_drifted": detail["n_drifted"],
        "drift_detected": drift_detected,
        "alert": drift_detected,
        "severity": severity,
        "summary": summary,
        "affected_features": affected_features,
        "recommended_action": _recommended_action(affected, drift_detected, severity),
        "feature_drift": detail["feature_drift"],
    }

    if write_files:
        directory = resolve_path(output_dir or "reports")
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "drift_report.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        (directory / "drift_report.md").write_text(
            _render_markdown(report), encoding="utf-8"
        )

    return report


def _render_markdown(report: dict[str, Any]) -> str:
    flag = "YES" if report["drift_detected"] else "NO"
    lines = [
        "# Drift Report",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Reference: `{report['reference']['version']}` ({report['reference']['n_rows']} rows)",
        f"- Current: `{report['current']['version']}` ({report['current']['n_rows']} rows)",
        (
            f"- Method: `{report['method']}` (KS alpha={report['ks_p_value_threshold']}, "
            f"mean shift={report['mean_shift_threshold']})"
        ),
        f"- Drift score: **{report['drift_score']:.4f}** (alert threshold {report['threshold']})",
        f"- Severity: **{report['severity']}**",
        f"- Drift detected: **{flag}**",
        "",
        "## Affected features",
        "",
    ]

    if report["affected_features"]:
        lines += [
            "| Feature | KS statistic | p-value | Reference mean | Current mean | Relative shift | Severity |",
            "|---|---|---|---|---|---|---|",
        ]
        for item in report["affected_features"]:
            lines.append(
                f"| {item['feature']} | {item['ks_statistic']:.4f} | {item['p_value']:.6f} | "
                f"{item['reference_mean']:.4f} | {item['current_mean']:.4f} | "
                f"{item['relative_shift']:.4f} | {item['severity']} |"
            )
    else:
        lines.append("_No feature exceeded the drift thresholds._")

    lines += [
        "",
        "## Summary",
        "",
        report["summary"],
        "",
        "## Recommended action",
        "",
        report["recommended_action"],
        "",
    ]
    return "\n".join(lines)
