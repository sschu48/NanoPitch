"""Verify technique training run artifacts from run_config.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .run_metadata import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify technique training run artifacts")
    parser.add_argument("--run-config", required=True, help="Path to run_config.json")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when run verification fails")
    return parser.parse_args()


def read_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _artifact_checks(name: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    path_value = metadata.get("path")
    expected_hash = metadata.get("sha256")
    expected_bytes = metadata.get("bytes")
    if not path_value:
        return [{"name": name, "ok": False, "detail": "missing artifact path"}]

    path = Path(str(path_value))
    if not path.is_file():
        return [{"name": name, "ok": False, "detail": f"file not found: {path}"}]

    actual_hash = sha256_file(path)
    actual_bytes = path.stat().st_size
    return [
        {
            "name": f"{name}:sha256",
            "ok": bool(expected_hash) and actual_hash == expected_hash,
            "detail": {
                "path": str(path),
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
            },
        },
        {
            "name": f"{name}:bytes",
            "ok": expected_bytes == actual_bytes,
            "detail": {
                "path": str(path),
                "expected_bytes": expected_bytes,
                "actual_bytes": actual_bytes,
            },
        },
    ]


def _collect_artifact_checks(prefix: str, value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict) and "path" in value:
        return _artifact_checks(prefix, value)
    if isinstance(value, dict):
        checks: list[dict[str, Any]] = []
        for key, child in sorted(value.items()):
            checks.extend(_collect_artifact_checks(f"{prefix}.{key}", child))
        return checks
    return []


def _run_dir_from_artifacts(artifacts: dict[str, Any]) -> Path | None:
    train_manifest = artifacts.get("train_manifest")
    if isinstance(train_manifest, dict) and train_manifest.get("path"):
        return Path(str(train_manifest["path"])).parent
    return None


def _metrics_history_checks(run_dir: Path | None) -> list[dict[str, Any]]:
    if run_dir is None:
        return []
    path = run_dir / "metrics_history.jsonl"
    if not path.is_file():
        return []

    checks: list[dict[str, Any]] = []
    epochs: list[int] = []
    errors: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: {exc}")
                continue
            epoch = record.get("epoch") if isinstance(record, dict) else None
            if not isinstance(epoch, int):
                errors.append(f"line {line_number}: missing integer epoch")
                continue
            epochs.append(epoch)

    checks.append(
        {
            "name": "metrics_history:jsonl",
            "ok": not errors,
            "detail": {"path": str(path), "errors": errors[:10], "error_count": len(errors)},
        }
    )
    checks.append(
        {
            "name": "metrics_history:unique_epochs",
            "ok": len(epochs) == len(set(epochs)),
            "detail": {
                "path": str(path),
                "epochs": epochs,
                "duplicate_epochs": sorted({epoch for epoch in epochs if epochs.count(epoch) > 1}),
            },
        }
    )
    checks.append(
        {
            "name": "metrics_history:contiguous_epochs",
            "ok": epochs == list(range(1, len(epochs) + 1)),
            "detail": {"path": str(path), "epochs": epochs},
        }
    )
    return checks


def verify_run_config(config: dict[str, Any]) -> dict[str, Any]:
    artifacts = config.get("artifacts")
    checks = _collect_artifact_checks("artifacts", artifacts) if isinstance(artifacts, dict) else []
    if isinstance(artifacts, dict):
        checks.extend(_metrics_history_checks(_run_dir_from_artifacts(artifacts)))
    if not checks:
        checks.append({"name": "artifacts", "ok": False, "detail": "run_config has no artifact metadata"})
    failed = [check for check in checks if not check["ok"]]
    return {
        "ok": not failed,
        "failed_checks": [check["name"] for check in failed],
        "checks": checks,
    }


def main() -> None:
    args = parse_args()
    report = verify_run_config(read_json(args.run_config))
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.strict and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
