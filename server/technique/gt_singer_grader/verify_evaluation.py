"""Verify technique evaluation artifacts and provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evaluation_artifacts import REQUIRED_EVALUATION_ARTIFACTS
from .verify_run import _artifact_checks, verify_run_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify technique evaluation artifacts")
    parser.add_argument("--eval-dir", required=True, help="Directory written by gt_singer_grader.evaluate")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when evaluation verification fails")
    return parser.parse_args()


def read_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def required_artifact_checks(eval_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(eval_dir)
    checks: list[dict[str, Any]] = []
    for name in REQUIRED_EVALUATION_ARTIFACTS:
        path = root / name
        checks.append(
            {
                "name": f"required_artifact:{name}",
                "ok": path.is_file(),
                "detail": str(path),
            }
        )
    return checks


def verify_evaluation_dir(eval_dir: str | Path) -> dict[str, Any]:
    root = Path(eval_dir)
    checks = required_artifact_checks(root)
    config_path = root / "evaluation_config.json"
    if config_path.is_file():
        try:
            config = read_json(config_path)
        except Exception as exc:
            checks.append({"name": "evaluation_config", "ok": False, "detail": str(exc)})
        else:
            checkpoint = config.get("checkpoint")
            manifest = config.get("manifest")
            run_config = config.get("run_config")
            if isinstance(checkpoint, dict):
                checks.extend(_artifact_checks("evaluation_config.checkpoint", checkpoint))
            else:
                checks.append({"name": "evaluation_config.checkpoint", "ok": False, "detail": "missing checkpoint metadata"})
            if isinstance(manifest, dict):
                checks.extend(_artifact_checks("evaluation_config.manifest", manifest))
            else:
                checks.append({"name": "evaluation_config.manifest", "ok": False, "detail": "missing manifest metadata"})
            if isinstance(run_config, dict):
                checks.extend(_artifact_checks("evaluation_config.run_config", run_config))
                path_value = run_config.get("path")
                if path_value:
                    try:
                        run_report = verify_run_config(read_json(path_value))
                    except Exception as exc:
                        checks.append(
                            {
                                "name": "evaluation_config.run_config:artifacts",
                                "ok": False,
                                "detail": str(exc),
                            }
                        )
                    else:
                        checks.append(
                            {
                                "name": "evaluation_config.run_config:artifacts",
                                "ok": run_report["ok"],
                                "detail": {
                                    "failed_checks": run_report["failed_checks"],
                                },
                            }
                        )

    failed = [check for check in checks if not check["ok"]]
    return {
        "ok": not failed,
        "failed_checks": [str(check["name"]) for check in failed],
        "checks": checks,
    }


def main() -> None:
    args = parse_args()
    report = verify_evaluation_dir(args.eval_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.strict and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
