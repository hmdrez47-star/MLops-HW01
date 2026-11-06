"""Post-deployment health check for the serving container (used by CI).

Waits for ``/health``, asserts the model is loaded, sends a real sample to
``/predict`` and verifies the response contract. Exits non-zero on any failure,
which is what makes the CD job fail when a deployment is broken.

    python scripts/healthcheck.py --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_config, load_csv, resolve_path


def _request(url: str, payload: dict | None = None, timeout: float = 5.0):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read().decode())


def _wait_for_health(base_url: str, timeout: float) -> dict:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            status, body = _request(f"{base_url}/health")
            if status == 200:
                return body
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            last_error = exc
        time.sleep(1.0)
    raise SystemExit(f"service did not become healthy within {timeout}s: {last_error}")


def _sample_features() -> list[float]:
    config = load_config("config/config.yaml")
    dataset_path = resolve_path(
        Path(config["data"]["raw_path"]) / config["data"]["dataset_file"]
    )
    frame = load_csv(dataset_path)
    target = config["data"]["target_column"]
    return [float(value) for value in frame.drop(columns=[target]).iloc[0].tolist()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Health check a running model API")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-latency", type=float, default=2.0)
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    health = _wait_for_health(base_url, args.timeout)
    print("health:", json.dumps(health))
    if health.get("status") != "ok" or not health.get("model_loaded"):
        raise SystemExit(f"model is not loaded: {health}")

    features = _sample_features()
    status, body = _request(f"{base_url}/predict", {"features": features})
    print("predict:", json.dumps(body))

    if status != 200:
        raise SystemExit(f"unexpected /predict status: {status}")
    if body.get("prediction") not in (0, 1):
        raise SystemExit(f"invalid prediction value: {body}")
    if not 0.0 <= float(body.get("confidence", -1)) <= 1.0:
        raise SystemExit(f"invalid confidence: {body}")
    if float(body.get("latency", 1e9)) > args.max_latency:
        raise SystemExit(f"latency above budget: {body}")

    try:
        _request(f"{base_url}/predict", {"features": features[:-1]})
        raise SystemExit("wrong feature count was accepted - validation is broken")
    except urllib.error.HTTPError as exc:
        if exc.code != 400:
            raise SystemExit(f"expected HTTP 400 for malformed payload, got {exc.code}")

    print("OK: health, prediction contract and input validation verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
