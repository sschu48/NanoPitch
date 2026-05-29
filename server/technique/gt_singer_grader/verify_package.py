"""Verify packaged technique-model metadata and evidence hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .constants import FAMILY_NAMES, TECHNIQUE_KEYS
from .package_candidate import (
    REQUIRED_EVALUATION_ARTIFACTS,
    app_domain_comparison_status,
    app_validation_audit_manifest_status,
    comparison_eval_artifact_status,
    find_candidate,
    read_json,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify packaged technique-model metadata")
    parser.add_argument("--metadata", default="./gt_singer_grader/models/technique_demo_metadata.json")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when package verification fails")
    return parser.parse_args()


def check_hash(path_value: Any, expected_value: Any, *, name: str) -> dict[str, Any]:
    path = Path(str(path_value)) if path_value else None
    expected = str(expected_value) if expected_value else ""
    if path is None:
        return {"name": name, "ok": False, "detail": "missing path"}
    if not path.is_file():
        return {"name": name, "ok": False, "detail": f"file not found: {path}"}
    actual = sha256_file(path)
    return {
        "name": name,
        "ok": bool(expected) and actual == expected,
        "detail": {
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
        },
    }


def check_semantic(name: str, ok: bool, detail: Any) -> dict[str, Any]:
    return {
        "name": name,
        "ok": ok,
        "detail": detail,
    }


def check_checkpoint_matches_evaluation(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    evaluated_checkpoint = (
        metadata.get("evaluated_checkpoint") if isinstance(metadata.get("evaluated_checkpoint"), dict) else {}
    )
    expected = evaluated_checkpoint.get("expected_checkpoint")
    if not isinstance(expected, dict):
        return [
            check_semantic(
                "packaged_checkpoint_matches_evaluation",
                False,
                {"expected_checkpoint": expected},
            )
        ]

    packaged_checkpoint = Path(str(metadata.get("packaged_checkpoint") or ""))
    expected_sha256 = expected.get("sha256")
    expected_bytes = expected.get("bytes")
    if not packaged_checkpoint.is_file():
        return [
            check_semantic(
                "packaged_checkpoint_matches_evaluation",
                False,
                {"packaged_checkpoint": str(packaged_checkpoint), "reason": "file not found"},
            )
        ]

    actual_sha256 = sha256_file(packaged_checkpoint)
    actual_bytes = packaged_checkpoint.stat().st_size
    return [
        check_semantic(
            "packaged_checkpoint_matches_evaluation:sha256",
            bool(expected_sha256) and actual_sha256 == expected_sha256,
            {
                "packaged_checkpoint": str(packaged_checkpoint),
                "expected_sha256": expected_sha256,
                "actual_sha256": actual_sha256,
            },
        ),
        check_semantic(
            "packaged_checkpoint_matches_evaluation:bytes",
            expected_bytes == actual_bytes,
            {
                "packaged_checkpoint": str(packaged_checkpoint),
                "expected_bytes": expected_bytes,
                "actual_bytes": actual_bytes,
            },
        ),
    ]


def comparison_candidate_status(metadata: dict[str, Any]) -> dict[str, Any]:
    comparison_report = metadata.get("comparison_report")
    candidate_eval_dir = metadata.get("candidate_eval_dir")
    checks: list[dict[str, Any]] = []
    if not comparison_report:
        checks.append({"name": "comparison_report", "ok": False, "detail": "missing comparison report path"})
        comparison: dict[str, Any] = {}
    else:
        try:
            comparison = read_json(comparison_report)
        except Exception as exc:
            checks.append({"name": "comparison_report", "ok": False, "detail": str(exc)})
            comparison = {}

    candidate: dict[str, Any] = {}
    if not candidate_eval_dir:
        checks.append({"name": "comparison.candidate_eval_dir", "ok": False, "detail": "missing candidate eval dir"})
    elif comparison:
        try:
            candidate = find_candidate(comparison, str(candidate_eval_dir))
        except Exception as exc:
            checks.append({"name": "comparison.candidate", "ok": False, "detail": str(exc)})
        else:
            checks.append(
                {
                    "name": "comparison.candidate",
                    "ok": True,
                    "detail": {
                        "comparison_report": str(comparison_report),
                        "candidate_eval_dir": str(candidate_eval_dir),
                    },
                }
            )

    promotion = candidate.get("promotion") if isinstance(candidate.get("promotion"), dict) else {}
    if candidate:
        checks.append(
            {
                "name": "comparison.candidate.promotion",
                "ok": promotion.get("eligible") is True
                and not promotion.get("failed_gates")
                and not promotion.get("unknown_gates"),
                "detail": promotion,
            }
        )
        artifact_status = comparison_eval_artifact_status(candidate, str(candidate_eval_dir))
        checks.append(
            {
                "name": "comparison.candidate.evaluation_artifacts",
                "ok": artifact_status["ok"],
                "detail": artifact_status,
            }
        )

    failed = [check for check in checks if not check["ok"]]
    return {
        "ok": not failed,
        "failed_checks": [str(check["name"]) for check in failed],
        "checks": checks,
    }


def semantic_checks(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    promotion = metadata.get("promotion") if isinstance(metadata.get("promotion"), dict) else {}
    app_validation_audit = (
        metadata.get("app_validation_audit") if isinstance(metadata.get("app_validation_audit"), dict) else {}
    )
    missing_eval_artifacts = metadata.get("missing_evaluation_artifacts")
    artifact_hashes = metadata.get("evaluation_artifact_sha256")
    required_artifacts = metadata.get("required_evaluation_artifacts") or REQUIRED_EVALUATION_ARTIFACTS
    required_artifact_set = {str(name) for name in required_artifacts}
    expected_artifact_set = set(REQUIRED_EVALUATION_ARTIFACTS)
    hashed_artifact_set = set(artifact_hashes) if isinstance(artifact_hashes, dict) else set()
    evaluated_checkpoint = (
        metadata.get("evaluated_checkpoint") if isinstance(metadata.get("evaluated_checkpoint"), dict) else {}
    )
    evaluation_verification = (
        metadata.get("evaluation_verification")
        if isinstance(metadata.get("evaluation_verification"), dict)
        else {}
    )
    comparison_evaluation_artifacts = (
        metadata.get("comparison_evaluation_artifacts")
        if isinstance(metadata.get("comparison_evaluation_artifacts"), dict)
        else {}
    )
    app_validation_manifest = (
        metadata.get("app_validation_manifest")
        if isinstance(metadata.get("app_validation_manifest"), dict)
        else {}
    )
    model_contract = metadata.get("model_contract") if isinstance(metadata.get("model_contract"), dict) else {}
    current_app_validation_manifest = app_validation_audit_manifest_status(app_validation_audit)
    current_comparison_candidate = comparison_candidate_status(metadata)
    app_domain_comparison = (
        metadata.get("app_domain_comparison")
        if isinstance(metadata.get("app_domain_comparison"), dict)
        else {}
    )
    comparison_report = metadata.get("comparison_report")
    candidate_eval_dir = metadata.get("candidate_eval_dir")
    if comparison_report and candidate_eval_dir:
        try:
            comparison = read_json(comparison_report)
            current_app_domain_comparison = app_domain_comparison_status(
                comparison,
                str(candidate_eval_dir),
                app_validation_audit,
            )
        except Exception as exc:
            current_app_domain_comparison = {
                "ok": False,
                "failed_checks": ["app_domain_comparison"],
                "error": str(exc),
            }
    else:
        current_app_domain_comparison = {
            "ok": False,
            "failed_checks": ["app_domain_comparison"],
            "error": "missing comparison_report or candidate_eval_dir",
        }

    return [
        check_semantic(
            "model_contract",
            model_contract.get("name") == "nanopitch_technique_detector"
            and model_contract.get("version") == 1
            and model_contract.get("axis") == "technique"
            and model_contract.get("families") == list(FAMILY_NAMES)
            and model_contract.get("techniques") == list(TECHNIQUE_KEYS)
            and model_contract.get("runtime_response") == "axis_result"
            and model_contract.get("candidate_kind") == "app_adapted",
            model_contract,
        ),
        check_semantic(
            "promotion_eligible",
            promotion.get("eligible") is True
            and not promotion.get("failed_gates")
            and not promotion.get("unknown_gates"),
            promotion,
        ),
        check_semantic(
            "app_validation_audit_ready",
            bool(metadata.get("app_validation_audit_report"))
            and app_validation_audit.get("ready_for_mvp_validation") is True,
            {
                "app_validation_audit_report": metadata.get("app_validation_audit_report"),
                "ready_for_mvp_validation": app_validation_audit.get("ready_for_mvp_validation"),
                "warnings": app_validation_audit.get("warnings") or [],
            },
        ),
        check_semantic(
            "no_missing_evaluation_artifacts",
            missing_eval_artifacts == [],
            {"missing_evaluation_artifacts": missing_eval_artifacts},
        ),
        check_semantic(
            "required_evaluation_artifacts",
            expected_artifact_set.issubset(required_artifact_set)
            and expected_artifact_set.issubset(hashed_artifact_set),
            {
                "required_evaluation_artifacts": sorted(required_artifact_set),
                "hashed_evaluation_artifacts": sorted(hashed_artifact_set),
                "expected_evaluation_artifacts": sorted(expected_artifact_set),
            },
        ),
        check_semantic(
            "evaluation_verification_ok",
            evaluation_verification.get("ok") is True and not evaluation_verification.get("failed_checks"),
            evaluation_verification,
        ),
        check_semantic(
            "evaluated_checkpoint_match",
            evaluated_checkpoint.get("ok") is True and not evaluated_checkpoint.get("failed_checks"),
            evaluated_checkpoint,
        ),
        check_semantic(
            "comparison_evaluation_artifacts_match",
            comparison_evaluation_artifacts.get("ok") is True
            and not comparison_evaluation_artifacts.get("failed_checks"),
            comparison_evaluation_artifacts,
        ),
        check_semantic(
            "comparison_candidate_match",
            current_comparison_candidate.get("ok") is True,
            current_comparison_candidate,
        ),
        check_semantic(
            "app_validation_manifest_match",
            app_validation_manifest.get("ok") is True
            and not app_validation_manifest.get("failed_checks")
            and current_app_validation_manifest.get("ok") is True,
            {
                "packaged_status": app_validation_manifest,
                "current_status": current_app_validation_manifest,
            },
        ),
        check_semantic(
            "app_domain_comparison_match",
            app_domain_comparison.get("ok") is True
            and not app_domain_comparison.get("failed_checks")
            and current_app_domain_comparison.get("ok") is True,
            {
                "packaged_status": app_domain_comparison,
                "current_status": current_app_domain_comparison,
            },
        ),
    ]


def verify_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = [
        check_hash(
            metadata.get("packaged_checkpoint"),
            metadata.get("packaged_checkpoint_sha256"),
            name="packaged_checkpoint",
        )
    ]

    comparison_report = metadata.get("comparison_report")
    comparison_hash = metadata.get("comparison_report_sha256")
    if comparison_report or comparison_hash:
        checks.append(check_hash(comparison_report, comparison_hash, name="comparison_report"))

    app_audit = metadata.get("app_validation_audit_report")
    app_audit_hash = metadata.get("app_validation_audit_sha256")
    if app_audit or app_audit_hash:
        checks.append(check_hash(app_audit, app_audit_hash, name="app_validation_audit"))

    eval_dir = Path(str(metadata.get("candidate_eval_dir") or ""))
    artifact_hashes = metadata.get("evaluation_artifact_sha256")
    if isinstance(artifact_hashes, dict):
        for artifact in metadata.get("required_evaluation_artifacts") or REQUIRED_EVALUATION_ARTIFACTS:
            checks.append(
                check_hash(
                    eval_dir / str(artifact),
                    artifact_hashes.get(str(artifact)),
                    name=f"evaluation_artifact:{artifact}",
                )
            )

    checks.extend(check_checkpoint_matches_evaluation(metadata))
    checks.extend(semantic_checks(metadata))

    failed = [check for check in checks if not check["ok"]]
    return {
        "ok": not failed,
        "failed_checks": [check["name"] for check in failed],
        "checks": checks,
    }


def main() -> None:
    args = parse_args()
    report = verify_metadata(read_json(args.metadata))
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.strict and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
