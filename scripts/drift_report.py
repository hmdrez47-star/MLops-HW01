"""Generate the drift report between two data versions (CLI + CI gate).

python scripts/drift_report.py --fail-on-drift
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_config, load_csv, resolve_path
from src.monitoring.drift_detection import generate_drift_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a data drift report")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--reference", help="reference CSV path")
    parser.add_argument("--current", help="current CSV path")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="exit with code 1 when the alert threshold is crossed (CI gate)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]
    monitoring_cfg = config["monitoring"]

    reference_path = args.reference or str(
        resolve_path(Path(data_cfg["raw_path"]) / data_cfg["dataset_file"])
    )
    current_path = args.current or str(
        resolve_path(Path(data_cfg["raw_path"]) / data_cfg["drift_dataset_file"])
    )

    reference_df = load_csv(reference_path)
    current_df = load_csv(current_path)

    features = [
        column
        for column in reference_df.columns
        if column != data_cfg["target_column"] and column in current_df.columns
    ]

    report = generate_drift_report(
        reference_df,
        current_df,
        reference_version=data_cfg["dataset_version"],
        current_version=data_cfg["drift_dataset_version"],
        threshold=float(monitoring_cfg["drift_threshold"]),
        ks_p_value_threshold=float(monitoring_cfg["ks_p_value_threshold"]),
        mean_shift_threshold=float(monitoring_cfg["mean_shift_threshold"]),
        features=features,
        output_dir=args.output_dir,
    )

    print(
        json.dumps(
            {
                k: report[k]
                for k in ("drift_detected", "drift_score", "severity", "summary")
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Report: {resolve_path(args.output_dir) / 'drift_report.json'}")

    if args.fail_on_drift and report["drift_detected"]:
        print("[ALERT] drift threshold crossed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
