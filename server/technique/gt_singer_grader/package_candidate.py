"""Package a promoted technique checkpoint with audit metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .constants import FAMILY_NAMES, TECHNIQUE_KEYS
from .evaluation_artifacts import REQUIRED_EVALUATION_ARTIFACTS
from .evaluation_artifacts import eval_artifact_hashes as current_eval_artifact_hashes
from .evaluation_artifacts import validate_eval_artifacts
from .run_metadata import file_metadata
from .verify_evaluation import verify_evaluation_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package a promoted technique checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Candidate checkpoint to copy")
    parser.add_argument("--comparison", required=True, help="compare_runs output JSON")
    parser.add_argument("--candidate-eval-dir", required=True, help="Candidate evaluation directory used in comparison")
    parser.add_argument(
        "--app-validation-audit",
        default=None,
        help="audit_app_validation JSON report required for product-facing packaging.",
    )
    parser.add_argument("--output-checkpoint", default="./gt_singer_grader/models/technique_demo_best.pth")
    parser.add_argument("--metadata", default="./gt_singer_grader/models/technique_demo_metadata.json")
    parser.add_argument(
        "--allow-ineligible",
        action="store_true",
        help="Write package files even when comparison promotion.eligible is false.",
    )
    return parser.parse_args()


def read_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def find_candidate(comparison: dict[str, Any], candidate_eval_dir: str) -> dict[str, Any]:
    candidate_path = _canonical_path(candidate_eval_dir)
    for candidate in comparison.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        if _canonical_path(str(candidate.get("path") or "")) == candidate_path:
            return candidate
    raise ValueError(f"candidate eval dir not found in comparison report: {candidate_eval_dir}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def comparison_eval_artifact_status(candidate: dict[str, Any], candidate_eval_dir: str | Path) -> dict[str, Any]:
    expected = candidate.get("evaluation_artifact_sha256")
    actual = current_eval_artifact_hashes(candidate_eval_dir)
    checks: list[dict[str, Any]] = []
    if not isinstance(expected, dict):
        checks.append(
            {
                "name": "comparison.evaluation_artifact_sha256",
                "ok": False,
                "detail": "comparison report has no evaluation artifact hashes",
            }
        )
    else:
        for name in REQUIRED_EVALUATION_ARTIFACTS:
            checks.append(
                {
                    "name": f"comparison.evaluation_artifact_sha256:{name}",
                    "ok": expected.get(name) == actual.get(name),
                    "detail": {
                        "expected_sha256": expected.get(name),
                        "actual_sha256": actual.get(name),
                    },
                }
            )
    failed = [check for check in checks if not check["ok"]]
    return {
        "ok": not failed,
        "failed_checks": [str(check["name"]) for check in failed],
        "expected": expected if isinstance(expected, dict) else {},
        "actual": actual,
        "checks": checks,
    }


def evaluated_checkpoint_status(checkpoint_path: str | Path, candidate_eval_dir: str | Path) -> dict[str, Any]:
    actual_path = Path(checkpoint_path)
    config_path = Path(candidate_eval_dir) / "evaluation_config.json"
    checks: list[dict[str, Any]] = []
    expected: dict[str, Any] = {}
    if not config_path.is_file():
        checks.append({"name": "evaluation_config", "ok": False, "detail": f"file not found: {config_path}"})
    else:
        try:
            config = read_json(config_path)
        except Exception as exc:
            checks.append({"name": "evaluation_config", "ok": False, "detail": str(exc)})
        else:
            checkpoint = config.get("checkpoint")
            if isinstance(checkpoint, dict):
                expected = checkpoint
                expected_sha256 = checkpoint.get("sha256")
                expected_bytes = checkpoint.get("bytes")
                actual_sha256 = sha256_file(actual_path) if actual_path.is_file() else None
                actual_bytes = actual_path.stat().st_size if actual_path.is_file() else None
                checks.extend(
                    [
                        {
                            "name": "packaged_checkpoint:evaluated_sha256",
                            "ok": bool(expected_sha256) and actual_sha256 == expected_sha256,
                            "detail": {
                                "expected_sha256": expected_sha256,
                                "actual_sha256": actual_sha256,
                            },
                        },
                        {
                            "name": "packaged_checkpoint:evaluated_bytes",
                            "ok": expected_bytes == actual_bytes,
                            "detail": {
                                "expected_bytes": expected_bytes,
                                "actual_bytes": actual_bytes,
                            },
                        },
                    ]
                )
            else:
                checks.append(
                    {
                        "name": "evaluation_config.checkpoint",
                        "ok": False,
                        "detail": "missing checkpoint metadata",
                    }
                )
    failed = [check for check in checks if not check["ok"]]
    return {
        "ok": not failed,
        "failed_checks": [str(check["name"]) for check in failed],
        "expected_checkpoint": expected,
        "checks": checks,
    }


def app_validation_audit_manifest_status(
    app_validation_audit: dict[str, Any],
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    expected = app_validation_audit.get("manifest")
    checks: list[dict[str, Any]] = []
    actual_path = Path(manifest_path) if manifest_path else None
    if not isinstance(expected, dict):
        checks.append(
            {
                "name": "app_validation_audit.manifest",
                "ok": False,
                "detail": "audit report has no manifest metadata",
            }
        )
        expected = {}
    else:
        expected_path = expected.get("path")
        if actual_path is None and expected_path:
            actual_path = Path(str(expected_path))
        if manifest_path is not None:
            checks.append(
                {
                    "name": "app_validation_audit.manifest:path",
                    "ok": bool(expected_path) and _canonical_path(expected_path) == _canonical_path(manifest_path),
                    "detail": {
                        "expected_path": expected_path,
                        "actual_path": str(manifest_path),
                    },
                }
            )

    actual = file_metadata(actual_path) if actual_path else {}
    checks.extend(
        [
            {
                "name": "app_validation_audit.manifest:exists",
                "ok": actual.get("exists") is True,
                "detail": actual,
            },
            {
                "name": "app_validation_audit.manifest:sha256",
                "ok": bool(expected.get("sha256")) and expected.get("sha256") == actual.get("sha256"),
                "detail": {
                    "expected_sha256": expected.get("sha256"),
                    "actual_sha256": actual.get("sha256"),
                },
            },
            {
                "name": "app_validation_audit.manifest:bytes",
                "ok": expected.get("bytes") == actual.get("bytes"),
                "detail": {
                    "expected_bytes": expected.get("bytes"),
                    "actual_bytes": actual.get("bytes"),
                },
            },
        ]
    )
    failed = [check for check in checks if not check["ok"]]
    return {
        "ok": not failed,
        "failed_checks": [str(check["name"]) for check in failed],
        "expected": expected,
        "actual": actual,
        "checks": checks,
    }


def _eval_config_manifest(eval_dir: str | Path) -> dict[str, Any]:
    config_path = Path(eval_dir) / "evaluation_config.json"
    config = read_json(config_path)
    manifest = config.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError(f"evaluation config has no manifest metadata: {config_path}")
    return manifest


def evaluation_manifest_match_status(
    eval_dir: str | Path,
    app_validation_audit: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    expected = app_validation_audit.get("manifest")
    checks: list[dict[str, Any]] = []
    if not isinstance(expected, dict):
        checks.append(
            {
                "name": f"{label}.manifest",
                "ok": False,
                "detail": "app validation audit has no manifest metadata",
            }
        )
        expected = {}
        actual = {}
    else:
        try:
            actual = _eval_config_manifest(eval_dir)
        except Exception as exc:
            actual = {}
            checks.append({"name": f"{label}.evaluation_config.manifest", "ok": False, "detail": str(exc)})

    if isinstance(expected, dict) and actual:
        checks.extend(
            [
                {
                    "name": f"{label}.manifest:sha256",
                    "ok": bool(expected.get("sha256")) and expected.get("sha256") == actual.get("sha256"),
                    "detail": {
                        "expected_sha256": expected.get("sha256"),
                        "actual_sha256": actual.get("sha256"),
                    },
                },
                {
                    "name": f"{label}.manifest:bytes",
                    "ok": expected.get("bytes") == actual.get("bytes"),
                    "detail": {
                        "expected_bytes": expected.get("bytes"),
                        "actual_bytes": actual.get("bytes"),
                    },
                },
            ]
        )
        expected_path = expected.get("path")
        actual_path = actual.get("path")
        if expected_path and actual_path:
            checks.append(
                {
                    "name": f"{label}.manifest:path",
                    "ok": _canonical_path(expected_path) == _canonical_path(actual_path),
                    "detail": {
                        "expected_path": expected_path,
                        "actual_path": actual_path,
                    },
                }
            )
    failed = [check for check in checks if not check["ok"]]
    return {
        "ok": not failed,
        "failed_checks": [str(check["name"]) for check in failed],
        "expected": expected,
        "actual": actual,
        "checks": checks,
    }


def app_domain_comparison_status(
    comparison: dict[str, Any],
    candidate_eval_dir: str | Path,
    app_validation_audit: dict[str, Any],
) -> dict[str, Any]:
    baseline = comparison.get("baseline") if isinstance(comparison.get("baseline"), dict) else {}
    baseline_path = baseline.get("path") if isinstance(baseline, dict) else None
    checks: list[dict[str, Any]] = [
        {
            "name": "comparison.baseline:path",
            "ok": bool(baseline_path),
            "detail": baseline_path or "comparison report has no baseline path",
        },
        {
            "name": "comparison.candidate:app_eval_dir",
            "ok": Path(candidate_eval_dir).name == "eval_app",
            "detail": str(candidate_eval_dir),
        },
    ]
    if baseline_path:
        checks.append(
            {
                "name": "comparison.baseline:app_eval_dir",
                "ok": Path(str(baseline_path)).name == "eval_app",
                "detail": str(baseline_path),
            }
        )
        expected_hashes = baseline.get("evaluation_artifact_sha256")
        actual_hashes = current_eval_artifact_hashes(str(baseline_path))
        if not isinstance(expected_hashes, dict):
            checks.append(
                {
                    "name": "comparison.baseline.evaluation_artifact_sha256",
                    "ok": False,
                    "detail": "comparison report has no baseline evaluation artifact hashes",
                }
            )
        else:
            for name in REQUIRED_EVALUATION_ARTIFACTS:
                checks.append(
                    {
                        "name": f"comparison.baseline.evaluation_artifact_sha256:{name}",
                        "ok": expected_hashes.get(name) == actual_hashes.get(name),
                        "detail": {
                            "expected_sha256": expected_hashes.get(name),
                            "actual_sha256": actual_hashes.get(name),
                        },
                    }
                )
        baseline_manifest = evaluation_manifest_match_status(
            str(baseline_path),
            app_validation_audit,
            label="comparison.baseline",
        )
        checks.extend(baseline_manifest["checks"])

    candidate_manifest = evaluation_manifest_match_status(
        candidate_eval_dir,
        app_validation_audit,
        label="comparison.candidate",
    )
    checks.extend(candidate_manifest["checks"])
    failed = [check for check in checks if not check["ok"]]
    return {
        "ok": not failed,
        "failed_checks": [str(check["name"]) for check in failed],
        "baseline_path": baseline_path,
        "checks": checks,
    }


def candidate_kind(candidate_eval_dir: str | Path) -> str:
    path_text = str(candidate_eval_dir).replace("\\", "/").lower()
    if "app_adapted" in path_text:
        return "app_adapted"
    if "balanced" in path_text:
        return "vocalset_balanced"
    if "vocalset" in path_text:
        return "vocalset"
    return "unknown"


def package_candidate(
    *,
    checkpoint: str,
    comparison_path: str,
    candidate_eval_dir: str,
    output_checkpoint: str,
    metadata_path: str,
    app_validation_audit_path: str | None = None,
    allow_ineligible: bool = False,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    comparison = read_json(comparison_path)
    candidate = find_candidate(comparison, candidate_eval_dir)
    kind = candidate_kind(candidate_eval_dir)
    if kind != "app_adapted" and not allow_ineligible:
        raise SystemExit(
            "product packaging requires an app-adapted candidate evaluated on app recordings; "
            f"candidate_kind={kind}. Use --allow-ineligible only for local demo packaging."
        )
    missing_eval_artifacts = validate_eval_artifacts(candidate_eval_dir)
    if missing_eval_artifacts and not allow_ineligible:
        raise SystemExit(
            "candidate evaluation directory is missing required artifact(s): "
            + ", ".join(missing_eval_artifacts)
            + ". Use --allow-ineligible only for local demo packaging."
        )
    evaluation_verification = verify_evaluation_dir(candidate_eval_dir)
    if not evaluation_verification["ok"] and not allow_ineligible:
        raise SystemExit(
            "candidate evaluation directory failed provenance verification: "
            + ", ".join(evaluation_verification["failed_checks"])
            + ". Use --allow-ineligible only for local demo packaging."
        )
    evaluated_checkpoint = evaluated_checkpoint_status(checkpoint_path, candidate_eval_dir)
    if not evaluated_checkpoint["ok"] and not allow_ineligible:
        raise SystemExit(
            "checkpoint does not match the checkpoint recorded by candidate evaluation: "
            + ", ".join(evaluated_checkpoint["failed_checks"])
            + ". Use --allow-ineligible only for local demo packaging."
        )
    comparison_eval_artifacts = comparison_eval_artifact_status(candidate, candidate_eval_dir)
    if not comparison_eval_artifacts["ok"] and not allow_ineligible:
        raise SystemExit(
            "comparison report does not match current candidate evaluation artifacts: "
            + ", ".join(comparison_eval_artifacts["failed_checks"])
            + ". Re-run compare_runs, or use --allow-ineligible only for local demo packaging."
        )

    promotion = candidate.get("promotion") if isinstance(candidate.get("promotion"), dict) else {}
    eligible = promotion.get("eligible") is True
    if not eligible and not allow_ineligible:
        failed = promotion.get("failed_gates") or []
        unknown = promotion.get("unknown_gates") or []
        raise SystemExit(
            "candidate is not promotion-eligible; "
            f"failed_gates={failed}, unknown_gates={unknown}. "
            "Use --allow-ineligible only for local demo packaging."
        )

    app_validation_audit: dict[str, Any] | None = None
    if not app_validation_audit_path and not allow_ineligible:
        raise SystemExit(
            "app validation audit is required for promoted packaging. "
            "Use --allow-ineligible only for local demo packaging."
        )
    if app_validation_audit_path:
        app_validation_audit = read_json(app_validation_audit_path)
        audit_ready = app_validation_audit.get("ready_for_mvp_validation") is True
        if not audit_ready and not allow_ineligible:
            warnings = app_validation_audit.get("warnings") or []
            raise SystemExit(
                "app validation audit is not ready for MVP validation; "
                f"warnings={warnings}. Use --allow-ineligible only for local demo packaging."
            )
        app_validation_manifest = app_validation_audit_manifest_status(app_validation_audit)
        if not app_validation_manifest["ok"] and not allow_ineligible:
            raise SystemExit(
                "app validation audit does not match its current manifest: "
                + ", ".join(app_validation_manifest["failed_checks"])
                + ". Re-run audit_app_validation, or use --allow-ineligible only for local demo packaging."
            )
        app_domain_comparison = app_domain_comparison_status(comparison, candidate_eval_dir, app_validation_audit)
        if not app_domain_comparison["ok"] and not allow_ineligible:
            raise SystemExit(
                "app-adapted packaging requires candidate and baseline app evaluations "
                "against the audited app validation manifest: "
                + ", ".join(app_domain_comparison["failed_checks"])
                + ". Re-run app-domain evaluation, compare_runs, or audit_app_validation."
            )
    else:
        app_validation_manifest = app_validation_audit_manifest_status({})
        app_domain_comparison = app_domain_comparison_status(comparison, candidate_eval_dir, {})

    output_path = Path(output_checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_path, output_path)
    app_validation_audit_sha256 = sha256_file(app_validation_audit_path) if app_validation_audit_path else None

    metadata = {
        "packaged_at": datetime.now(timezone.utc).isoformat(),
        "model_contract": {
            "name": "nanopitch_technique_detector",
            "version": 1,
            "axis": "technique",
            "families": list(FAMILY_NAMES),
            "techniques": list(TECHNIQUE_KEYS),
            "runtime_response": "axis_result",
            "candidate_kind": kind,
        },
        "source_checkpoint": str(checkpoint_path),
        "packaged_checkpoint": str(output_path),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "packaged_checkpoint_sha256": sha256_file(output_path),
        "comparison_report": str(comparison_path),
        "comparison_report_sha256": sha256_file(comparison_path),
        "candidate_eval_dir": str(candidate_eval_dir),
        "evaluation_artifact_sha256": current_eval_artifact_hashes(candidate_eval_dir),
        "comparison_evaluation_artifacts": {
            "ok": comparison_eval_artifacts["ok"],
            "failed_checks": comparison_eval_artifacts["failed_checks"],
        },
        "evaluation_verification": {
            "ok": evaluation_verification["ok"],
            "failed_checks": evaluation_verification["failed_checks"],
        },
        "evaluated_checkpoint": {
            "ok": evaluated_checkpoint["ok"],
            "failed_checks": evaluated_checkpoint["failed_checks"],
            "expected_checkpoint": evaluated_checkpoint["expected_checkpoint"],
        },
        "required_evaluation_artifacts": list(REQUIRED_EVALUATION_ARTIFACTS),
        "missing_evaluation_artifacts": missing_eval_artifacts,
        "app_validation_audit_report": str(app_validation_audit_path) if app_validation_audit_path else None,
        "app_validation_audit_sha256": app_validation_audit_sha256,
        "app_validation_audit": app_validation_audit or {},
        "app_validation_manifest": {
            "ok": app_validation_manifest["ok"],
            "failed_checks": app_validation_manifest["failed_checks"],
        },
        "app_domain_comparison": {
            "ok": app_domain_comparison["ok"],
            "failed_checks": app_domain_comparison["failed_checks"],
            "baseline_path": app_domain_comparison["baseline_path"],
        },
        "promotion": promotion,
        "metrics": candidate.get("metrics") or {},
        "operating_point": candidate.get("operating_point") or {},
        "delta_vs_baseline": candidate.get("delta_vs_baseline") or {},
        "gates": {
            "absolute": comparison.get("gates") or {},
            "regression": comparison.get("regression_gates") or {},
        },
        "packaging_note": "Comparison promotion.eligible was true."
        if eligible
        else "Packaged with --allow-ineligible for local demo use only.",
    }

    metadata_output = Path(metadata_path)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    args = parse_args()
    metadata = package_candidate(
        checkpoint=args.checkpoint,
        comparison_path=args.comparison,
        candidate_eval_dir=args.candidate_eval_dir,
        output_checkpoint=args.output_checkpoint,
        metadata_path=args.metadata,
        app_validation_audit_path=args.app_validation_audit,
        allow_ineligible=args.allow_ineligible,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
