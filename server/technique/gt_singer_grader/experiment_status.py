"""Report the next concrete step for technique-model experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .app_label_coverage import build_report as build_app_label_coverage_report
from .package_candidate import (
    REQUIRED_EVALUATION_ARTIFACTS,
    app_validation_audit_manifest_status,
    comparison_eval_artifact_status,
    find_candidate,
    read_json,
    validate_eval_artifacts,
)
from .preflight import build_report as build_preflight_report
from .preflight import TECHNIQUE_ROOT, resolve_audio_root, resolve_local_path
from .run_metadata import file_metadata
from .verify_evaluation import verify_evaluation_dir
from .verify_run import verify_run_config


TRAINING_ARTIFACTS = (
    "run_config.json",
    "metrics_history.jsonl",
    "best_metrics.json",
    "train_manifest.jsonl",
    "val_manifest.jsonl",
    "checkpoints/best.pth",
)


@contextmanager
def technique_working_dir() -> Any:
    previous = Path.cwd()
    os.chdir(TECHNIQUE_ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize technique experiment readiness and next steps")
    parser.add_argument("--gtsinger-root", default="./gt_singer_grader/data/GTSinger")
    parser.add_argument("--vocalset-root", default="./gt_singer_grader/data/VocalSet")
    parser.add_argument("--app-audio-dir", default="./gt_singer_grader/data/app_recordings/raw")
    parser.add_argument("--app-audio-root", default=".")
    parser.add_argument("--app-labels", default="./gt_singer_grader/data/app_recordings/review_labels.csv")
    parser.add_argument("--app-prepare-report", default="./gt_singer_grader/data/app_recordings/prepare_report.json")
    parser.add_argument(
        "--app-collection-plan",
        default="./gt_singer_grader/data/app_recordings/collection_plan.csv",
    )
    parser.add_argument(
        "--app-collection-plan-json",
        default="./gt_singer_grader/data/app_recordings/collection_plan.json",
    )
    parser.add_argument("--app-collection-root", default="./gt_singer_grader/data/app_recordings")
    parser.add_argument("--app-collection-checklist", default="./gt_singer_grader/data/app_recordings/collection_checklist.csv")
    parser.add_argument("--app-collection-missing", default="./gt_singer_grader/data/app_recordings/collection_missing.csv")
    parser.add_argument(
        "--app-collection-materialize-report",
        default="./gt_singer_grader/data/app_recordings/collection_materialize_report.json",
    )
    parser.add_argument("--app-collection-packet-dir", default="./gt_singer_grader/data/app_recordings/collection_packet")
    parser.add_argument(
        "--app-collection-packet-summary",
        default="./gt_singer_grader/data/app_recordings/collection_packet_summary.json",
    )
    parser.add_argument("--checkpoint", default="./gt_singer_grader/models/technique_demo_best.pth")
    parser.add_argument("--metadata", default="./gt_singer_grader/models/technique_demo_metadata.json")
    parser.add_argument("--baseline-split-group", choices=("song", "speaker"), default="song")
    parser.add_argument("--baseline-run-dir", default="./gt_singer_grader/runs/gtsinger_song_aug_v1")
    parser.add_argument("--baseline-plan", default="./gt_singer_grader/runs/gtsinger_song_aug_v1/training_plan.json")
    parser.add_argument("--baseline-eval-dir", default="./gt_singer_grader/runs/gtsinger_song_aug_v1/eval_val")
    parser.add_argument("--vocalset-manifest", default="./gt_singer_grader/manifests/vocalset.jsonl")
    parser.add_argument("--vocalset-run-dir", default="./gt_singer_grader/runs/gtsinger_vocalset_song_v2")
    parser.add_argument("--vocalset-plan", default="./gt_singer_grader/runs/gtsinger_vocalset_song_v2/training_plan.json")
    parser.add_argument("--vocalset-eval-dir", default="./gt_singer_grader/runs/gtsinger_vocalset_song_v2/eval_val")
    parser.add_argument("--vocalset-balanced-manifest", default="./gt_singer_grader/manifests/vocalset_balanced_120.jsonl")
    parser.add_argument(
        "--vocalset-balanced-summary",
        default="./gt_singer_grader/manifests/vocalset_balanced_120_summary.json",
    )
    parser.add_argument("--vocalset-balanced-max-per-family", type=int, default=120)
    parser.add_argument(
        "--vocalset-balanced-run-dir",
        default="./gt_singer_grader/runs/gtsinger_vocalset_balanced120_song_v2",
    )
    parser.add_argument(
        "--vocalset-balanced-plan",
        default="./gt_singer_grader/runs/gtsinger_vocalset_balanced120_song_v2/training_plan.json",
    )
    parser.add_argument(
        "--vocalset-balanced-eval-dir",
        default="./gt_singer_grader/runs/gtsinger_vocalset_balanced120_song_v2/eval_val",
    )
    parser.add_argument("--app-manifest", default="./gt_singer_grader/manifests/app_recordings.jsonl")
    parser.add_argument("--app-trainable-manifest", default="./gt_singer_grader/manifests/app_recordings_trainable.jsonl")
    parser.add_argument("--app-eval-only-manifest", default="./gt_singer_grader/manifests/app_recordings_eval_only.jsonl")
    parser.add_argument("--app-train-manifest", default="./gt_singer_grader/manifests/app_recordings_train.jsonl")
    parser.add_argument("--app-val-manifest", default="./gt_singer_grader/manifests/app_recordings_val.jsonl")
    parser.add_argument("--app-eval-manifest", default="./gt_singer_grader/manifests/app_recordings_eval.jsonl")
    parser.add_argument("--app-validation-audit", default="./gt_singer_grader/manifests/app_recordings_eval_audit.json")
    parser.add_argument(
        "--app-adapted-train-manifest",
        default="./gt_singer_grader/manifests/app_adapted_train.jsonl",
    )
    parser.add_argument("--app-baseline-eval-dir", default="./gt_singer_grader/runs/gtsinger_song_aug_v1/eval_app")
    parser.add_argument("--app-adapted-run-dir", default="./gt_singer_grader/runs/gtsinger_app_adapted_v1")
    parser.add_argument(
        "--app-adapted-plan",
        default="./gt_singer_grader/runs/gtsinger_app_adapted_v1/training_plan.json",
    )
    parser.add_argument("--app-adapted-eval-dir", default="./gt_singer_grader/runs/gtsinger_app_adapted_v1/eval_app")
    parser.add_argument("--comparison", default="./gt_singer_grader/runs/run_comparison_public_v2.json")
    parser.add_argument("--balanced-comparison", default="./gt_singer_grader/runs/run_comparison_balanced120_v2.json")
    parser.add_argument("--app-adapted-comparison", default="./gt_singer_grader/runs/run_comparison_app_adapted.json")
    parser.add_argument("--output-json", default=None, help="Optional path to write the JSON report")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero until the full MVP evidence set exists")
    return parser.parse_args()


def missing_files(root: str | Path, relative_paths: tuple[str, ...]) -> list[str]:
    root_path = resolve_local_path(root)
    return [name for name in relative_paths if not (root_path / name).is_file()]


def training_run_status(run_dir: str) -> dict[str, Any]:
    root = resolve_local_path(run_dir)
    missing = missing_files(root, TRAINING_ARTIFACTS)
    config_path = root / "run_config.json"
    verification: dict[str, Any] | None = None
    if config_path.is_file():
        try:
            with technique_working_dir():
                verification = verify_run_config(read_json(config_path))
        except Exception as exc:  # pragma: no cover - defensive CLI report path
            verification = {"ok": False, "failed_checks": ["run_config"], "error": str(exc)}

    complete = not missing and (verification is None or verification.get("ok") is True)
    return {
        "path": str(root),
        "complete": complete,
        "missing_artifacts": missing,
        "artifact_verification": verification,
    }


def evaluation_status(eval_dir: str) -> dict[str, Any]:
    eval_path = resolve_local_path(eval_dir)
    missing = validate_eval_artifacts(str(eval_path))
    verification: dict[str, Any] | None = None
    if not missing:
        try:
            with technique_working_dir():
                verification = verify_evaluation_dir(str(eval_path))
        except Exception as exc:  # pragma: no cover - defensive CLI report path
            verification = {"ok": False, "failed_checks": ["evaluation"], "error": str(exc)}
    return {
        "path": str(eval_path),
        "complete": not missing and (verification is None or verification.get("ok") is True),
        "required_artifacts": list(REQUIRED_EVALUATION_ARTIFACTS),
        "missing_artifacts": missing,
        "artifact_verification": verification,
    }


def file_status(path: str, source_paths: list[str | Path] | tuple[str | Path, ...] = ()) -> dict[str, Any]:
    path_obj = resolve_local_path(path)
    exists = path_obj.is_file()
    status: dict[str, Any] = {"path": str(path_obj), "exists": exists}
    if source_paths:
        sources = [resolve_local_path(source) for source in source_paths]
        source_status = [
            {
                "path": str(source),
                "exists": source.is_file(),
                "mtime": source.stat().st_mtime if source.is_file() else None,
            }
            for source in sources
        ]
        missing_sources = [source["path"] for source in source_status if not source["exists"]]
        stale_sources: list[str] = []
        if exists:
            output_mtime = path_obj.stat().st_mtime
            stale_sources = [
                str(source)
                for source in sources
                if source.is_file() and source.stat().st_mtime > output_mtime
            ]
            status["mtime"] = output_mtime
        status.update(
            {
                "source_files": source_status,
                "missing_source_files": missing_sources,
                "stale_source_files": stale_sources,
                "current_for_sources": exists and not missing_sources and not stale_sources,
            }
        )
    return status


def report_file_current(report: dict[str, Any], name: str, *, fallback: bool = True) -> bool:
    status = report.get(name)
    if not isinstance(status, dict):
        return fallback
    if not status.get("exists"):
        return False
    return status.get("current_for_sources", True) is True


def stale_artifact_message(status: dict[str, Any], command: str) -> str:
    stale = status.get("stale_source_files") or []
    missing = status.get("missing_source_files") or []
    details: list[str] = []
    if stale:
        details.append(f"newer sources={stale}")
    if missing:
        details.append(f"missing sources={missing}")
    suffix = "; ".join(details) if details else "source files changed"
    return f"{status.get('path')} is stale for its source files ({suffix}). Re-run: {command}"


def app_collection_plan_status(csv_path: str, json_path: str | None = None) -> dict[str, Any]:
    path_obj = resolve_local_path(csv_path)
    rows: list[dict[str, str]] = []
    status: dict[str, Any] = {
        "path": str(path_obj),
        "exists": path_obj.is_file(),
        "planned_records": 0,
        "planned_groups": 0,
        "intended_family_counts": {},
    }
    if path_obj.is_file():
        try:
            with path_obj.open("r", encoding="utf-8", newline="") as handle:
                rows = [dict(row) for row in csv.DictReader(handle)]
        except Exception as exc:  # pragma: no cover - defensive CLI report path
            status["error"] = str(exc)
            return status
        family_counts: dict[str, int] = {}
        groups: set[str] = set()
        for row in rows:
            family = (row.get("intended_family") or "").strip()
            if family:
                family_counts[family] = family_counts.get(family, 0) + 1
            singer_id = (row.get("singer_id") or "").strip()
            if singer_id:
                groups.add(singer_id)
        status.update(
            {
                "planned_records": len(rows),
                "planned_groups": len(groups),
                "intended_family_counts": dict(sorted(family_counts.items())),
            }
        )
    if json_path:
        json_status = file_status(json_path)
        status["json_path"] = json_status["path"]
        status["json_exists"] = json_status["exists"]
        if json_status["exists"]:
            try:
                payload = read_json(resolve_local_path(json_path))
            except Exception as exc:  # pragma: no cover - defensive CLI report path
                status["json_error"] = str(exc)
            else:
                status["thresholds"] = payload.get("thresholds") or {}
                status["needed"] = payload.get("needed") or {}
                status["ready_for_collection_target"] = payload.get("ready_for_collection_target") is True
                if not status["planned_records"] and isinstance(payload.get("planned_records"), int):
                    status["planned_records"] = payload["planned_records"]
                if not status["planned_groups"] and isinstance(payload.get("planned_groups"), int):
                    status["planned_groups"] = payload["planned_groups"]
    return status


def app_collection_materialize_status(
    checklist_path: str,
    report_path: str,
    missing_csv_path: str | None = None,
) -> dict[str, Any]:
    checklist = resolve_local_path(checklist_path)
    report = resolve_local_path(report_path)
    configured_missing_csv = resolve_local_path(missing_csv_path) if missing_csv_path else None
    status: dict[str, Any] = {
        "checklist_path": str(checklist),
        "checklist_exists": checklist.is_file(),
        "missing_csv": str(configured_missing_csv) if configured_missing_csv else str(checklist.with_name("collection_missing.csv")),
        "missing_csv_exists": configured_missing_csv.is_file() if configured_missing_csv else checklist.with_name("collection_missing.csv").is_file(),
        "missing_csv_records": 0,
        "report_path": str(report),
        "report_exists": report.is_file(),
        "planned_records": 0,
        "planned_groups": 0,
        "existing_audio_files": 0,
        "valid_audio_files": 0,
        "invalid_audio_files": 0,
        "missing_audio_files": 0,
        "duplicate_suggested_filenames": [],
        "ok": False,
    }
    if checklist.is_file():
        try:
            with checklist.open("r", encoding="utf-8", newline="") as handle:
                rows = [dict(row) for row in csv.DictReader(handle)]
        except Exception as exc:  # pragma: no cover - defensive CLI report path
            status["checklist_error"] = str(exc)
        else:
            status["checklist_records"] = len(rows)
            status["checklist_missing_audio_files"] = sum(1 for row in rows if row.get("exists") != "yes")
    missing_csv = Path(str(status["missing_csv"]))
    if missing_csv.is_file():
        try:
            with missing_csv.open("r", encoding="utf-8", newline="") as handle:
                status["missing_csv_records"] = len([row for row in csv.DictReader(handle)])
        except Exception as exc:  # pragma: no cover - defensive CLI report path
            status["missing_csv_error"] = str(exc)
    if report.is_file():
        try:
            payload = read_json(report)
        except Exception as exc:  # pragma: no cover - defensive CLI report path
            status["report_error"] = str(exc)
        else:
            for key in (
                "missing_csv",
                "planned_records",
                "planned_groups",
                "created_directories",
                "existing_audio_files",
                "valid_audio_files",
                "invalid_audio_files",
                "missing_audio_files",
            ):
                if key in payload:
                    status[key] = payload[key]
            status["duplicate_suggested_filenames"] = payload.get("duplicate_suggested_filenames") or []
            status["existing_audio_paths"] = payload.get("existing_audio_paths") or []
            status["invalid_audio_paths"] = payload.get("invalid_audio_paths") or []
            status["invalid_audio_reasons"] = payload.get("invalid_audio_reasons") or {}
            status["missing_audio_paths"] = payload.get("missing_audio_paths") or []
            status["missing_by_family"] = payload.get("missing_by_family") or {}
            status["missing_by_singer"] = payload.get("missing_by_singer") or {}
            status["intended_family_counts"] = payload.get("intended_family_counts") or {}
            status["ok"] = payload.get("ok") is True
            status["ready_for_review_csv"] = payload.get("ready_for_review_csv") is True
            status["wav_validation_enabled"] = payload.get("wav_validation_enabled") is True
            status["wav_duration_bounds_seconds"] = payload.get("wav_duration_bounds_seconds") or {}
    return status


def app_collection_packet_status(
    output_dir: str,
    summary_path: str,
    checklist_path: str | None = None,
) -> dict[str, Any]:
    output = resolve_local_path(output_dir)
    summary = resolve_local_path(summary_path)
    checklist_metadata = file_metadata(resolve_local_path(checklist_path)) if checklist_path else None
    status: dict[str, Any] = {
        "output_dir": str(output),
        "exists": output.is_dir(),
        "summary_path": str(summary),
        "summary_exists": summary.is_file(),
        "index_path": str(output / "index.md"),
        "index_exists": (output / "index.md").is_file(),
        "sheet_count": len([path for path in output.glob("*.md") if path.name != "index.md"]) if output.is_dir() else 0,
        "planned_records": 0,
        "planned_singers": 0,
        "existing_audio_files": 0,
        "missing_audio_files": 0,
        "duplicate_audio_paths": [],
        "source_checklist": None,
        "current_checklist": checklist_metadata,
        "checklist_match": checklist_metadata is None,
        "ok": False,
    }
    if summary.is_file():
        try:
            payload = read_json(summary)
        except Exception as exc:  # pragma: no cover - defensive CLI report path
            status["summary_error"] = str(exc)
        else:
            for key in ("planned_records", "planned_singers", "existing_audio_files", "missing_audio_files"):
                if key in payload:
                    status[key] = payload[key]
            index_path = payload.get("index_path")
            if isinstance(index_path, str) and index_path:
                index_resolved = resolve_local_path(index_path)
                status["index_path"] = str(index_resolved)
                status["index_exists"] = index_resolved.is_file()
            status["duplicate_audio_paths"] = payload.get("duplicate_audio_paths") or []
            status["intended_family_counts"] = payload.get("intended_family_counts") or {}
            source_checklist = payload.get("source_checklist")
            status["source_checklist"] = source_checklist if isinstance(source_checklist, dict) else None
            if checklist_metadata is not None and isinstance(source_checklist, dict):
                status["checklist_match"] = (
                    source_checklist.get("sha256") == checklist_metadata.get("sha256")
                    and source_checklist.get("bytes") == checklist_metadata.get("bytes")
                )
            status["ok"] = (
                payload.get("ok") is True
                and status["checklist_match"] is True
                and status["index_exists"] is True
            )
    return status


def app_label_coverage_status(csv_path: str, audio_root: str | Path) -> dict[str, Any]:
    path_obj = resolve_local_path(csv_path)
    if not path_obj.is_file():
        return {
            "path": str(path_obj),
            "exists": False,
            "ready_for_collection_target": False,
            "report": None,
        }
    try:
        report = build_app_label_coverage_report(
            str(path_obj),
            require_audio_files=True,
            audio_root=resolve_audio_root(audio_root, path_obj),
        )
    except Exception as exc:
        return {
            "path": str(path_obj),
            "exists": True,
            "ready_for_collection_target": False,
            "error": str(exc),
            "report": None,
        }
    return {
        "path": str(path_obj),
        "exists": True,
        "ready_for_collection_target": report["ready_for_collection_target"],
        "missing_audio_file_count": report["missing_audio_file_count"],
        "missing_target_families": report.get("missing_target_families") or {},
        "negative_shortfall": report.get("negative_shortfall"),
        "group_shortfall": report.get("group_shortfall"),
        "unlabeled_records": report.get("unlabeled_records"),
        "intended_family_mismatch_count": report.get("intended_family_mismatch_count"),
        "missing_reviewer_id_count": report.get("missing_reviewer_id_count"),
        "review_progress": report.get("review_progress") or {},
        "warnings": report["warnings"],
        "report": report,
    }


def app_prepare_report_status(path: str) -> dict[str, Any]:
    path_obj = resolve_local_path(path)
    if not path_obj.is_file():
        return {
            "path": str(path_obj),
            "exists": False,
            "collection_plan_fully_matched": False,
        }
    try:
        payload = read_json(path_obj)
    except Exception as exc:  # pragma: no cover - defensive CLI report path
        return {
            "path": str(path_obj),
            "exists": True,
            "collection_plan_fully_matched": False,
            "error": str(exc),
        }
    return {
        "path": str(path_obj),
        "exists": True,
        "records": payload.get("records"),
        "collection_plan_rows": payload.get("collection_plan_rows"),
        "collection_plan_matches": payload.get("collection_plan_matches"),
        "collection_plan_fully_matched": payload.get("collection_plan_fully_matched") is True,
        "missing_collection_plan_suggestions": payload.get("missing_collection_plan_suggestions") or [],
        "unplanned_audio_paths": payload.get("unplanned_audio_paths") or [],
    }


def preflight_check(report: dict[str, Any], name: str) -> dict[str, Any]:
    for check in report.get("preflight", {}).get("checks", []):
        if isinstance(check, dict) and check.get("name") == name:
            return check
    return {}


def report_file_exists(report: dict[str, Any], name: str, *, fallback: bool = False) -> bool:
    status = report.get(name)
    if not isinstance(status, dict):
        return fallback
    return status.get("exists") is True


def training_plan_status(path: str) -> dict[str, Any]:
    path_obj = resolve_local_path(path)
    if not path_obj.is_file():
        return {"path": str(path_obj), "exists": False, "ok": False}
    try:
        payload = read_json(path_obj)
    except Exception as exc:  # pragma: no cover - defensive CLI report path
        return {"path": str(path_obj), "exists": True, "ok": False, "error": str(exc)}
    errors = payload.get("errors") if isinstance(payload.get("errors"), list) else []
    return {
        "path": str(path_obj),
        "exists": True,
        "ok": payload.get("ok") is True,
        "source": payload.get("source"),
        "split": payload.get("split") if isinstance(payload.get("split"), dict) else {},
        "errors": errors,
    }


def app_audit_status(path: str, manifest_path: str) -> dict[str, Any]:
    path_obj = resolve_local_path(path)
    if not path_obj.is_file():
        return {
            "path": str(path_obj),
            "exists": False,
            "ready_for_mvp_validation": False,
            "manifest_match": False,
        }
    try:
        payload = read_json(path_obj)
    except Exception as exc:  # pragma: no cover - defensive CLI report path
        return {
            "path": str(path_obj),
            "exists": True,
            "ready_for_mvp_validation": False,
            "manifest_match": False,
            "error": str(exc),
        }
    manifest_status = app_validation_audit_manifest_status(payload, str(resolve_local_path(manifest_path)))
    return {
        "path": str(path_obj),
        "exists": True,
        "ready_for_mvp_validation": payload.get("ready_for_mvp_validation") is True and manifest_status["ok"],
        "audit_ready": payload.get("ready_for_mvp_validation") is True,
        "manifest_match": manifest_status["ok"],
        "manifest_failed_checks": manifest_status["failed_checks"],
        "records": payload.get("records"),
        "warnings": payload.get("warnings") or [],
        "missing_target_families": payload.get("missing_target_families") or {},
        "negative_shortfall": payload.get("negative_shortfall"),
        "group_shortfall": payload.get("group_shortfall"),
    }


def comparison_status(path: str, candidate_eval_dir: str) -> dict[str, Any]:
    path_obj = resolve_local_path(path)
    candidate_eval_path = resolve_local_path(candidate_eval_dir)
    if not path_obj.is_file():
        return {
            "path": str(path_obj),
            "exists": False,
            "candidate_found": False,
            "candidate_eligible": False,
        }
    try:
        payload = read_json(path_obj)
        with technique_working_dir():
            candidate = find_candidate(payload, str(candidate_eval_path))
    except Exception as exc:  # pragma: no cover - defensive CLI report path
        return {
            "path": str(path_obj),
            "exists": True,
            "candidate_found": False,
            "candidate_eligible": False,
            "error": str(exc),
        }
    promotion = candidate.get("promotion") if isinstance(candidate.get("promotion"), dict) else {}
    with technique_working_dir():
        artifact_status = comparison_eval_artifact_status(candidate, str(candidate_eval_path))
    promotion_eligible = promotion.get("eligible") is True
    return {
        "path": str(path_obj),
        "exists": True,
        "candidate_found": True,
        "candidate_eligible": promotion_eligible and artifact_status["ok"],
        "promotion_eligible": promotion_eligible,
        "promotion": promotion,
        "failed_gates": promotion.get("failed_gates") or [],
        "unknown_gates": promotion.get("unknown_gates") or [],
        "evaluation_artifact_match": artifact_status["ok"],
        "evaluation_artifact_failed_checks": artifact_status["failed_checks"],
    }


def candidate_failed_message(label: str, status: dict[str, Any]) -> str:
    failed = status.get("failed_gates") or []
    unknown = status.get("unknown_gates") or []
    return (
        f"{label} candidate is not promotion-eligible; "
        f"failed_gates={failed}, unknown_gates={unknown}."
    )


def selected_package_command(report: dict[str, Any], commands: dict[str, str]) -> str:
    if report.get("app_adapted_comparison", {}).get("candidate_eligible"):
        return commands["package_app_adapted_candidate"]
    return (
        "No packageable app-adapted candidate is available yet. "
        "Complete app-domain baseline evaluation, app-adapted training, "
        "app-domain evaluation, and app-adapted comparison before packaging."
    )


def app_prepare_report_message(status: dict[str, Any]) -> str | None:
    if not status.get("exists") or status.get("collection_plan_fully_matched"):
        return None
    missing = status.get("missing_collection_plan_suggestions") or []
    unplanned = status.get("unplanned_audio_paths") or []
    if not missing and not unplanned:
        return None
    return (
        "Latest app prepare report has collection-plan mismatches "
        f"(missing planned={len(missing)}, unplanned wavs={len(unplanned)}). "
        "Reconcile collection before review labels are treated as ready."
    )


def app_collection_plan_message(status: dict[str, Any]) -> str:
    if not status.get("exists"):
        return ""
    records = status.get("planned_records")
    groups = status.get("planned_groups")
    if not isinstance(records, int) or records <= 0:
        return ""
    if isinstance(groups, int) and groups > 0:
        return f"Current collection plan requests {records} clips across {groups} singer groups."
    return f"Current collection plan requests {records} clips."


def app_collection_materialize_message(status: dict[str, Any]) -> str:
    if not status.get("report_exists") and not status.get("checklist_exists"):
        return ""
    missing = status.get("missing_audio_files", status.get("checklist_missing_audio_files"))
    existing = status.get("existing_audio_files")
    invalid = status.get("invalid_audio_files")
    checklist = status.get("checklist_path")
    missing_csv = status.get("missing_csv")
    parts: list[str] = []
    if isinstance(missing, int):
        parts.append(f"{missing} missing audio files")
    if isinstance(existing, int):
        parts.append(f"{existing} existing audio files")
    if isinstance(invalid, int) and invalid > 0:
        parts.append(f"{invalid} invalid WAV files")
    counts = ", ".join(parts) if parts else "audio collection progress"
    locations: list[str] = []
    if isinstance(checklist, str) and checklist:
        locations.append(f"checklist at {checklist}")
    if isinstance(missing, int) and missing > 0 and isinstance(missing_csv, str) and missing_csv:
        locations.append(f"missing list at {missing_csv}")
    if locations:
        return f"Current collection checklist shows {counts}; {', '.join(locations)}."
    return f"Current collection checklist shows {counts}."


def app_label_coverage_message(status: dict[str, Any]) -> str:
    report = status.get("report") if isinstance(status.get("report"), dict) else {}
    missing_families = status.get("missing_target_families") or report.get("missing_target_families") or {}
    missing_audio = status.get("missing_audio_file_count", report.get("missing_audio_file_count"))
    negative_shortfall = status.get("negative_shortfall", report.get("negative_shortfall"))
    group_shortfall = status.get("group_shortfall", report.get("group_shortfall"))
    unlabeled = status.get("unlabeled_records", report.get("unlabeled_records"))
    mismatch_count = status.get("intended_family_mismatch_count", report.get("intended_family_mismatch_count"))
    missing_reviewer_ids = status.get("missing_reviewer_id_count", report.get("missing_reviewer_id_count"))
    progress = status.get("review_progress") if isinstance(status.get("review_progress"), dict) else {}
    labeled = progress.get("labeled_records")
    records = progress.get("records")
    warnings = status.get("warnings") or report.get("warnings") or []

    parts: list[str] = []
    if isinstance(missing_families, dict) and missing_families:
        family_text = ", ".join(f"{family}:{count}" for family, count in sorted(missing_families.items()))
        parts.append(f"missing target families {{{family_text}}}")
    if isinstance(negative_shortfall, int) and negative_shortfall > 0:
        parts.append(f"negative shortfall {negative_shortfall}")
    if isinstance(group_shortfall, int) and group_shortfall > 0:
        parts.append(f"group shortfall {group_shortfall}")
    if isinstance(unlabeled, int) and unlabeled > 0:
        parts.append(f"{unlabeled} unlabeled rows")
    if isinstance(missing_audio, int) and missing_audio > 0:
        parts.append(f"{missing_audio} missing audio files")
    if isinstance(mismatch_count, int) and mismatch_count > 0:
        parts.append(f"{mismatch_count} intended-family mismatches")
    if isinstance(missing_reviewer_ids, int) and missing_reviewer_ids > 0:
        parts.append(f"{missing_reviewer_ids} labeled rows missing reviewer_id")
    if isinstance(labeled, int) and isinstance(records, int):
        parts.append(f"reviewed {labeled}/{records} rows")
    if not parts and warnings:
        parts.append("warnings: " + ", ".join(str(warning) for warning in warnings))
    summary = "; ".join(parts) if parts else "coverage is below target"
    return f"App label coverage is not ready ({summary})."


def needs_app_collection_materialization(report: dict[str, Any]) -> bool:
    if not report.get("app_collection_plan", {}).get("exists"):
        return False
    materialized = report.get("app_collection_materialize")
    if not materialized:
        return True
    return not materialized.get("report_exists") or not materialized.get("checklist_exists")


def needs_app_collection_packet_export(report: dict[str, Any]) -> bool:
    materialized = report.get("app_collection_materialize") or {}
    if not materialized.get("checklist_exists"):
        return False
    packet = report.get("app_collection_packet")
    if not packet:
        return True
    return (
        not packet.get("summary_exists")
        or not packet.get("exists")
        or not packet.get("index_exists")
        or not packet.get("checklist_match", True)
    )


def args_value(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def command_list(args: argparse.Namespace) -> dict[str, str]:
    baseline_split_group = args_value(args, "baseline_split_group", "song")
    app_audio_dir = args_value(args, "app_audio_dir", "./gt_singer_grader/data/app_recordings/raw")
    app_audio_root = args_value(args, "app_audio_root", ".")
    app_collection_plan = args_value(
        args,
        "app_collection_plan",
        "./gt_singer_grader/data/app_recordings/collection_plan.csv",
    )
    app_collection_plan_json = args_value(
        args,
        "app_collection_plan_json",
        "./gt_singer_grader/data/app_recordings/collection_plan.json",
    )
    app_collection_root = args_value(args, "app_collection_root", "./gt_singer_grader/data/app_recordings")
    app_collection_checklist = args_value(
        args,
        "app_collection_checklist",
        "./gt_singer_grader/data/app_recordings/collection_checklist.csv",
    )
    app_collection_missing = args_value(
        args,
        "app_collection_missing",
        "./gt_singer_grader/data/app_recordings/collection_missing.csv",
    )
    app_collection_materialize_report = args_value(
        args,
        "app_collection_materialize_report",
        "./gt_singer_grader/data/app_recordings/collection_materialize_report.json",
    )
    app_collection_packet_dir = args_value(
        args,
        "app_collection_packet_dir",
        "./gt_singer_grader/data/app_recordings/collection_packet",
    )
    app_collection_packet_summary = args_value(
        args,
        "app_collection_packet_summary",
        "./gt_singer_grader/data/app_recordings/collection_packet_summary.json",
    )
    app_prepare_report = args_value(
        args,
        "app_prepare_report",
        str(Path(args.app_labels).with_name("prepare_report.json")),
    )
    vocalset_balanced_manifest = args_value(
        args,
        "vocalset_balanced_manifest",
        "./gt_singer_grader/manifests/vocalset_balanced_120.jsonl",
    )
    vocalset_balanced_summary = args_value(
        args,
        "vocalset_balanced_summary",
        "./gt_singer_grader/manifests/vocalset_balanced_120_summary.json",
    )
    vocalset_balanced_max_per_family = args_value(args, "vocalset_balanced_max_per_family", 120)
    vocalset_balanced_run_dir = args_value(
        args,
        "vocalset_balanced_run_dir",
        "./gt_singer_grader/runs/gtsinger_vocalset_balanced120_song_v2",
    )
    vocalset_balanced_plan = args_value(
        args,
        "vocalset_balanced_plan",
        "./gt_singer_grader/runs/gtsinger_vocalset_balanced120_song_v2/training_plan.json",
    )
    vocalset_balanced_eval_dir = args_value(
        args,
        "vocalset_balanced_eval_dir",
        "./gt_singer_grader/runs/gtsinger_vocalset_balanced120_song_v2/eval_val",
    )
    balanced_comparison = args_value(
        args,
        "balanced_comparison",
        "./gt_singer_grader/runs/run_comparison_balanced120_v2.json",
    )
    app_train_manifest = args_value(args, "app_train_manifest", "./gt_singer_grader/manifests/app_recordings_train.jsonl")
    app_adapted_train_manifest = args_value(
        args,
        "app_adapted_train_manifest",
        "./gt_singer_grader/manifests/app_adapted_train.jsonl",
    )
    app_baseline_eval_dir = args_value(args, "app_baseline_eval_dir", "./gt_singer_grader/runs/gtsinger_song_aug_v1/eval_app")
    app_adapted_run_dir = args_value(args, "app_adapted_run_dir", "./gt_singer_grader/runs/gtsinger_app_adapted_v1")
    app_adapted_plan = args_value(
        args,
        "app_adapted_plan",
        "./gt_singer_grader/runs/gtsinger_app_adapted_v1/training_plan.json",
    )
    app_adapted_eval_dir = args_value(args, "app_adapted_eval_dir", "./gt_singer_grader/runs/gtsinger_app_adapted_v1/eval_app")
    app_adapted_comparison = args_value(
        args,
        "app_adapted_comparison",
        "./gt_singer_grader/runs/run_comparison_app_adapted.json",
    )
    return {
        "preflight": "python3 -m gt_singer_grader.preflight",
        "plan_baseline": (
            "python3 -m gt_singer_grader.plan_training "
            f"--dataset-root {args.gtsinger_root} "
            f"--split-group {baseline_split_group} "
            f"--output-json {args.baseline_plan} --strict"
        ),
        "train_baseline": (
            "python3 -m gt_singer_grader.train "
            f"--dataset-root {args.gtsinger_root} "
            f"--output-dir {args.baseline_run_dir} "
            f"--training-plan {args.baseline_plan} "
            f"--split-group {baseline_split_group} --user-audio-augmentation --epochs 50 --batch-size 8 --quiet"
        ),
        "verify_baseline": (
            "python3 -m gt_singer_grader.verify_run "
            f"--run-config {args.baseline_run_dir}/run_config.json --strict"
        ),
        "evaluate_baseline": (
            "python3 -m gt_singer_grader.evaluate "
            f"--checkpoint {args.baseline_run_dir}/checkpoints/best.pth "
            f"--manifest {args.baseline_run_dir}/val_manifest.jsonl "
            f"--run-config {args.baseline_run_dir}/run_config.json "
            f"--output-dir {args.baseline_eval_dir} --max-control-fpr 0.25 --max-non-technique-fpr 0.25"
        ),
        "verify_baseline_eval": (
            "python3 -m gt_singer_grader.verify_evaluation "
            f"--eval-dir {args.baseline_eval_dir} --strict"
        ),
        "build_vocalset_manifest": (
            "python3 -m gt_singer_grader.build_manifest vocalset "
            f"--root {args.vocalset_root} --output {args.vocalset_manifest}"
        ),
        "plan_vocalset_candidate": (
            "python3 -m gt_singer_grader.plan_training "
            f"--dataset-root {args.gtsinger_root} "
            f"--split-group {baseline_split_group} "
            f"--extra-train-manifest {args.vocalset_manifest} "
            f"--output-json {args.vocalset_plan} --strict"
        ),
        "train_vocalset_candidate": (
            "python3 -m gt_singer_grader.train "
            f"--dataset-root {args.gtsinger_root} "
            f"--output-dir {args.vocalset_run_dir} "
            f"--training-plan {args.vocalset_plan} "
            f"--split-group {baseline_split_group} --user-audio-augmentation "
            f"--extra-train-manifest {args.vocalset_manifest} "
            "--epochs 50 --batch-size 8 --quiet"
        ),
        "evaluate_vocalset_candidate": (
            "python3 -m gt_singer_grader.evaluate "
            f"--checkpoint {args.vocalset_run_dir}/checkpoints/best.pth "
            f"--manifest {args.vocalset_run_dir}/val_manifest.jsonl "
            f"--run-config {args.vocalset_run_dir}/run_config.json "
            f"--output-dir {args.vocalset_eval_dir} --max-control-fpr 0.25 --max-non-technique-fpr 0.25"
        ),
        "verify_vocalset_eval": (
            "python3 -m gt_singer_grader.verify_evaluation "
            f"--eval-dir {args.vocalset_eval_dir} --strict"
        ),
        "compare_runs": (
            "python3 -m gt_singer_grader.compare_runs "
            f"--baseline {args.baseline_eval_dir} "
            f"--candidate {args.vocalset_eval_dir} "
            f"--candidate {vocalset_balanced_eval_dir} "
            f"--output-json {args.comparison}"
        ),
        "build_balanced_vocalset_manifest": (
            "python3 -m gt_singer_grader.sample_manifest "
            f"--input {args.vocalset_manifest} "
            f"--output {vocalset_balanced_manifest} "
            f"--summary-output {vocalset_balanced_summary} "
            f"--max-per-family {vocalset_balanced_max_per_family}"
        ),
        "plan_balanced_vocalset_candidate": (
            "python3 -m gt_singer_grader.plan_training "
            f"--dataset-root {args.gtsinger_root} "
            f"--split-group {baseline_split_group} "
            f"--extra-train-manifest {vocalset_balanced_manifest} "
            f"--output-json {vocalset_balanced_plan} --strict"
        ),
        "train_balanced_vocalset_candidate": (
            "python3 -m gt_singer_grader.train "
            f"--dataset-root {args.gtsinger_root} "
            f"--output-dir {vocalset_balanced_run_dir} "
            f"--training-plan {vocalset_balanced_plan} "
            f"--split-group {baseline_split_group} --user-audio-augmentation "
            f"--extra-train-manifest {vocalset_balanced_manifest} "
            "--epochs 50 --batch-size 8 --quiet"
        ),
        "evaluate_balanced_vocalset_candidate": (
            "python3 -m gt_singer_grader.evaluate "
            f"--checkpoint {vocalset_balanced_run_dir}/checkpoints/best.pth "
            f"--manifest {vocalset_balanced_run_dir}/val_manifest.jsonl "
            f"--run-config {vocalset_balanced_run_dir}/run_config.json "
            f"--output-dir {vocalset_balanced_eval_dir} --max-control-fpr 0.25 --max-non-technique-fpr 0.25"
        ),
        "verify_balanced_vocalset_eval": (
            "python3 -m gt_singer_grader.verify_evaluation "
            f"--eval-dir {vocalset_balanced_eval_dir} --strict"
        ),
        "compare_balanced_runs": (
            "python3 -m gt_singer_grader.compare_runs "
            f"--baseline {args.baseline_eval_dir} "
            f"--candidate {vocalset_balanced_eval_dir} "
            f"--output-json {balanced_comparison}"
        ),
        "build_app_manifest": (
            "python3 -m gt_singer_grader.build_manifest app-recordings "
            f"--csv {args.app_labels} --output {args.app_manifest}"
        ),
        "check_app_label_coverage": (
            "python3 -m gt_singer_grader.app_label_coverage "
            f"--csv {args.app_labels} --audio-root {app_audio_root} "
            "--require-audio-files --strict"
        ),
        "plan_app_collection": (
            "python3 -m gt_singer_grader.plan_app_collection "
            f"--csv {args.app_labels} "
            f"--output-json {app_collection_plan_json} "
            f"--output-csv {app_collection_plan} --clips-per-singer 7"
        ),
        "materialize_app_collection": (
            "python3 -m gt_singer_grader.materialize_app_collection "
            f"--plan {app_collection_plan} "
            f"--root {app_collection_root} "
            f"--checklist {app_collection_checklist} "
            f"--missing-csv {app_collection_missing} "
            f"--report-json {app_collection_materialize_report} --strict"
        ),
        "export_app_collection_packet": (
            "python3 -m gt_singer_grader.export_app_collection_packet "
            f"--checklist {app_collection_checklist} "
            f"--output-dir {app_collection_packet_dir} "
            f"--summary-json {app_collection_packet_summary} --strict"
        ),
        "check_app_collection": (
            "python3 -m gt_singer_grader.materialize_app_collection "
            f"--plan {app_collection_plan} "
            f"--root {app_collection_root} "
            f"--checklist {app_collection_checklist} "
            f"--missing-csv {app_collection_missing} "
            f"--report-json {app_collection_materialize_report} "
            "--strict --require-audio-files --validate-wav-files --min-wav-seconds 5 --max-wav-seconds 10"
        ),
        "prepare_app_review_csv": (
            "python3 -m gt_singer_grader.prepare_app_recordings "
            f"--audio-dir {app_audio_dir} "
            f"--output {args.app_labels} "
            f"--report-json {app_prepare_report} "
            f"--relative-to . --collection-plan {app_collection_plan} --singer-id-from-parent"
        ),
        "filter_app_manifest": (
            "python3 -m gt_singer_grader.filter_manifest "
            f"--input {args.app_manifest} "
            f"--trainable-output {args.app_trainable_manifest} "
            f"--eval-only-output {args.app_eval_only_manifest} "
            "--summary-output ./gt_singer_grader/manifests/app_recordings_filter_summary.json"
        ),
        "split_app_trainable_manifest": (
            "python3 -m gt_singer_grader.split_manifest "
            f"--input {args.app_trainable_manifest} "
            f"--train-output {app_train_manifest} "
            f"--val-output {args.app_val_manifest} "
            "--summary-output ./gt_singer_grader/manifests/app_recordings_split_summary.json "
            "--val-ratio 0.2 --strict-non-empty --strict-family-coverage"
        ),
        "merge_app_eval_manifest": (
            "python3 -m gt_singer_grader.merge_manifest "
            f"--input {args.app_val_manifest} "
            f"--input {args.app_eval_only_manifest} "
            f"--output {args.app_eval_manifest} "
            "--summary-output ./gt_singer_grader/manifests/app_recordings_eval_summary.json"
        ),
        "audit_app_validation": (
            "python3 -m gt_singer_grader.audit_app_validation "
            f"--manifest {args.app_eval_manifest} --output-json {args.app_validation_audit} --strict"
        ),
        "merge_app_adapted_train_manifest": (
            "python3 -m gt_singer_grader.merge_manifest "
            f"--input {args.baseline_run_dir}/train_manifest.jsonl "
            f"--input {app_train_manifest} "
            f"--output {app_adapted_train_manifest} "
            "--summary-output ./gt_singer_grader/manifests/app_adapted_train_summary.json"
        ),
        "plan_app_adapted_candidate": (
            "python3 -m gt_singer_grader.plan_training "
            f"--train-manifest {app_adapted_train_manifest} "
            f"--val-manifest {args.app_val_manifest} "
            "--require-train-dataset gtsinger "
            "--require-train-dataset app_recordings "
            "--require-val-dataset app_recordings "
            f"--output-json {app_adapted_plan} --strict"
        ),
        "train_app_adapted_candidate": (
            "python3 -m gt_singer_grader.train "
            f"--train-manifest {app_adapted_train_manifest} "
            f"--val-manifest {args.app_val_manifest} "
            f"--output-dir {app_adapted_run_dir} "
            f"--training-plan {app_adapted_plan} "
            "--user-audio-augmentation --epochs 50 --batch-size 8 --quiet"
        ),
        "evaluate_app_baseline": (
            "python3 -m gt_singer_grader.evaluate "
            f"--checkpoint {args.baseline_run_dir}/checkpoints/best.pth "
            f"--manifest {args.app_eval_manifest} "
            f"--run-config {args.baseline_run_dir}/run_config.json "
            f"--output-dir {app_baseline_eval_dir} --max-control-fpr 0.25 --max-non-technique-fpr 0.25"
        ),
        "verify_app_baseline_eval": (
            "python3 -m gt_singer_grader.verify_evaluation "
            f"--eval-dir {app_baseline_eval_dir} --strict"
        ),
        "evaluate_app_adapted_candidate": (
            "python3 -m gt_singer_grader.evaluate "
            f"--checkpoint {app_adapted_run_dir}/checkpoints/best.pth "
            f"--manifest {args.app_eval_manifest} "
            f"--run-config {app_adapted_run_dir}/run_config.json "
            f"--output-dir {app_adapted_eval_dir} --max-control-fpr 0.25 --max-non-technique-fpr 0.25"
        ),
        "verify_app_adapted_eval": (
            "python3 -m gt_singer_grader.verify_evaluation "
            f"--eval-dir {app_adapted_eval_dir} --strict"
        ),
        "compare_app_adapted_runs": (
            "python3 -m gt_singer_grader.compare_runs "
            f"--baseline {app_baseline_eval_dir} "
            f"--candidate {app_adapted_eval_dir} "
            f"--output-json {app_adapted_comparison}"
        ),
        "package_candidate": (
            "python3 -m gt_singer_grader.package_candidate "
            f"--checkpoint {args.vocalset_run_dir}/checkpoints/best.pth "
            f"--comparison {args.comparison} "
            f"--candidate-eval-dir {args.vocalset_eval_dir} "
            f"--app-validation-audit {args.app_validation_audit} "
            f"--output-checkpoint {args.checkpoint} "
            f"--metadata {args.metadata}"
        ),
        "package_balanced_candidate": (
            "python3 -m gt_singer_grader.package_candidate "
            f"--checkpoint {vocalset_balanced_run_dir}/checkpoints/best.pth "
            f"--comparison {balanced_comparison} "
            f"--candidate-eval-dir {vocalset_balanced_eval_dir} "
            f"--app-validation-audit {args.app_validation_audit} "
            f"--output-checkpoint {args.checkpoint} "
            f"--metadata {args.metadata}"
        ),
        "package_app_adapted_candidate": (
            "python3 -m gt_singer_grader.package_candidate "
            f"--checkpoint {app_adapted_run_dir}/checkpoints/best.pth "
            f"--comparison {app_adapted_comparison} "
            f"--candidate-eval-dir {app_adapted_eval_dir} "
            f"--app-validation-audit {args.app_validation_audit} "
            f"--output-checkpoint {args.checkpoint} "
            f"--metadata {args.metadata}"
        ),
    }


def next_actions(report: dict[str, Any], commands: dict[str, str]) -> list[str]:
    preflight = report["preflight"]
    if not preflight["ok"]:
        return list(preflight.get("required_next_steps") or preflight.get("next_steps") or [commands["preflight"]])
    if not report["baseline_plan"]["ok"]:
        return [commands["plan_baseline"]]
    if not report["baseline_run"]["complete"]:
        return [commands["train_baseline"]]
    if not report["baseline_eval"]["complete"]:
        if not report["baseline_eval"]["missing_artifacts"] and report["baseline_eval"].get("artifact_verification"):
            return [commands["verify_baseline_eval"], commands["evaluate_baseline"]]
        return [commands["verify_baseline"], commands["evaluate_baseline"]]
    if not report["vocalset_manifest"]["exists"]:
        return [commands["build_vocalset_manifest"]]
    if not report["vocalset_plan"]["ok"]:
        return [commands["plan_vocalset_candidate"]]
    if not report["vocalset_run"]["complete"]:
        return [commands["train_vocalset_candidate"]]
    if not report["vocalset_eval"]["complete"]:
        if not report["vocalset_eval"]["missing_artifacts"] and report["vocalset_eval"].get("artifact_verification"):
            return [commands["verify_vocalset_eval"], commands["evaluate_vocalset_candidate"]]
        return [commands["evaluate_vocalset_candidate"]]
    if not report["comparison"]["exists"]:
        return [commands["compare_runs"]]
    if report["comparison"].get("candidate_found") is False:
        error = report["comparison"].get("error")
        return [
            "Comparison report does not contain the configured candidate evaluation directory; "
            f"error={error}. Re-run: {commands['compare_runs']}"
        ]
    if not report["comparison"].get("evaluation_artifact_match", True):
        failed = report["comparison"].get("evaluation_artifact_failed_checks") or []
        return [
            "Comparison report is stale for the current candidate evaluation artifacts; "
            f"failed_checks={failed}. Re-run: {commands['compare_runs']}"
        ]
    if not report["comparison"]["candidate_eligible"]:
        if "vocalset_balanced_manifest" not in report:
            failed = report["comparison"].get("failed_gates") or []
            unknown = report["comparison"].get("unknown_gates") or []
            return [
                "Candidate is not promotion-eligible; review comparison gates "
                f"failed_gates={failed}, unknown_gates={unknown}, then retrain or recalibrate."
            ]
        if not report["vocalset_balanced_manifest"]["exists"]:
            return [commands["build_balanced_vocalset_manifest"]]
        if not report["vocalset_balanced_plan"]["ok"]:
            return [commands["plan_balanced_vocalset_candidate"]]
        if not report["vocalset_balanced_run"]["complete"]:
            return [commands["train_balanced_vocalset_candidate"]]
        if not report["vocalset_balanced_eval"]["complete"]:
            if not report["vocalset_balanced_eval"]["missing_artifacts"] and report["vocalset_balanced_eval"].get(
                "artifact_verification"
            ):
                return [commands["verify_balanced_vocalset_eval"], commands["evaluate_balanced_vocalset_candidate"]]
            return [commands["evaluate_balanced_vocalset_candidate"]]
        if not report["balanced_comparison"]["exists"]:
            return [commands["compare_balanced_runs"]]
        if report["balanced_comparison"].get("candidate_found") is False:
            error = report["balanced_comparison"].get("error")
            return [
                "Balanced comparison report does not contain the configured candidate evaluation directory; "
                f"error={error}. Re-run: {commands['compare_balanced_runs']}"
            ]
        if not report["balanced_comparison"].get("evaluation_artifact_match", True):
            failed = report["balanced_comparison"].get("evaluation_artifact_failed_checks") or []
            return [
                "Balanced comparison report is stale for the current candidate evaluation artifacts; "
                f"failed_checks={failed}. Re-run: {commands['compare_balanced_runs']}"
            ]
        if not report["balanced_comparison"]["candidate_eligible"] and not report["app_manifest"]["exists"]:
            app_labels = preflight_check(report, "app_recording_labels")
            if app_labels and not app_labels.get("ok"):
                detail = app_labels.get("detail")
                if isinstance(detail, str):
                    if not report.get("app_collection_plan", {}).get("exists"):
                        return [
                            candidate_failed_message("Full VocalSet", report["comparison"])
                            + " "
                            + candidate_failed_message("Balanced VocalSet", report["balanced_comparison"])
                            + " Generate an app recording collection plan from the release coverage targets: "
                            + commands["plan_app_collection"]
                        ]
                    prepare_message = app_prepare_report_message(report.get("app_prepare_report", {}))
                    if prepare_message:
                        return [
                            candidate_failed_message("Full VocalSet", report["comparison"])
                            + " "
                            + candidate_failed_message("Balanced VocalSet", report["balanced_comparison"])
                            + " "
                            + prepare_message
                            + " Refresh the starter CSV/report with: "
                            + commands["prepare_app_review_csv"]
                        ]
                    if needs_app_collection_materialization(report):
                        plan_message = app_collection_plan_message(report.get("app_collection_plan", {}))
                        return [
                            candidate_failed_message("Full VocalSet", report["comparison"])
                            + " "
                            + candidate_failed_message("Balanced VocalSet", report["balanced_comparison"])
                            + " "
                            + plan_message
                            + " Create the recording folders/checklist with: "
                            + commands["materialize_app_collection"]
                        ]
                    if needs_app_collection_packet_export(report):
                        materialize_message = app_collection_materialize_message(
                            report.get("app_collection_materialize", {})
                        )
                        return [
                            candidate_failed_message("Full VocalSet", report["comparison"])
                            + " "
                            + candidate_failed_message("Balanced VocalSet", report["balanced_comparison"])
                            + " "
                            + materialize_message
                            + " Export per-singer recording sheets with: "
                            + commands["export_app_collection_packet"]
                        ]
                    plan_message = app_collection_plan_message(report.get("app_collection_plan", {}))
                    materialize_message = app_collection_materialize_message(
                        report.get("app_collection_materialize", {})
                    )
                    return [
                        candidate_failed_message("Full VocalSet", report["comparison"])
                        + " "
                        + candidate_failed_message("Balanced VocalSet", report["balanced_comparison"])
                        + " Collect and label app recordings at "
                        f"{detail}, using gt_singer_grader/app_recordings_review_template.csv. "
                        + plan_message
                        + " "
                        + materialize_message
                        + " "
                        + "Use the generated collection checklist first. "
                        + "After WAVs are collected, validate readability and 5-10 second duration with: "
                        + commands["check_app_collection"]
                        + " Then prepare the starter CSV with: "
                        + commands["prepare_app_review_csv"]
                    ]
            if not report.get("app_label_coverage", {}).get("ready_for_collection_target", True):
                coverage_message = app_label_coverage_message(report.get("app_label_coverage", {}))
                return [
                    candidate_failed_message("Full VocalSet", report["comparison"])
                    + " "
                    + candidate_failed_message("Balanced VocalSet", report["balanced_comparison"])
                    + " "
                    + coverage_message
                    + " Refresh the app collection plan before fixing coverage: "
                    + commands["plan_app_collection"],
                    "Then recheck app label coverage before building the manifest: " + commands["check_app_label_coverage"],
                ]
            return [
                candidate_failed_message("Full VocalSet", report["comparison"])
                + " "
                + candidate_failed_message("Balanced VocalSet", report["balanced_comparison"])
                + " Build the app recording manifest next: "
                + commands["build_app_manifest"]
            ]
        if not report["balanced_comparison"]["candidate_eligible"]:
            # Public supplemental candidates are exhausted; continue with app-domain validation data.
            pass
    if not report_file_current(report, "app_manifest"):
        if report["app_manifest"].get("exists"):
            return [stale_artifact_message(report["app_manifest"], commands["build_app_manifest"])]
        app_labels = preflight_check(report, "app_recording_labels")
        if app_labels and not app_labels.get("ok"):
            detail = app_labels.get("detail")
            if isinstance(detail, str):
                if not report.get("app_collection_plan", {}).get("exists"):
                    return [
                        "Generate an app recording collection plan from the release coverage targets: "
                        + commands["plan_app_collection"]
                    ]
                prepare_message = app_prepare_report_message(report.get("app_prepare_report", {}))
                if prepare_message:
                    return [prepare_message + " Refresh the starter CSV/report with: " + commands["prepare_app_review_csv"]]
                if needs_app_collection_materialization(report):
                    plan_message = app_collection_plan_message(report.get("app_collection_plan", {}))
                    return [
                        plan_message
                        + " Create the recording folders/checklist with: "
                        + commands["materialize_app_collection"]
                    ]
                if needs_app_collection_packet_export(report):
                    materialize_message = app_collection_materialize_message(report.get("app_collection_materialize", {}))
                    return [
                        materialize_message
                        + " Export per-singer recording sheets with: "
                        + commands["export_app_collection_packet"]
                    ]
                plan_message = app_collection_plan_message(report.get("app_collection_plan", {}))
                materialize_message = app_collection_materialize_message(report.get("app_collection_materialize", {}))
                return [
                    "Collect and label app recordings at "
                    f"{detail}, using gt_singer_grader/app_recordings_review_template.csv. "
                    + plan_message
                    + " "
                    + materialize_message
                    + " "
                    + "Use the generated collection checklist first. "
                    "After WAVs are collected, validate readability and 5-10 second duration with: "
                    + commands["check_app_collection"]
                    + " Then prepare the starter CSV with: "
                    + commands["prepare_app_review_csv"]
                ]
            return [
                "Fix app recording label CSV errors, then rebuild: "
                + commands["build_app_manifest"]
            ]
        if not report.get("app_label_coverage", {}).get("ready_for_collection_target", True):
            coverage_message = app_label_coverage_message(report.get("app_label_coverage", {}))
            return [
                coverage_message + " Refresh the app collection plan before fixing coverage: " + commands["plan_app_collection"],
                "Then recheck app label coverage before building the manifest: " + commands["check_app_label_coverage"],
            ]
        return [commands["build_app_manifest"]]
    if not report_file_current(report, "app_trainable_manifest") or not report_file_current(
        report,
        "app_eval_only_manifest",
    ):
        if report["app_trainable_manifest"].get("exists") or report["app_eval_only_manifest"].get("exists"):
            stale_status = (
                report["app_trainable_manifest"]
                if not report_file_current(report, "app_trainable_manifest")
                else report["app_eval_only_manifest"]
            )
            return [stale_artifact_message(stale_status, commands["filter_app_manifest"])]
        return [commands["filter_app_manifest"]]
    if not report_file_current(
        report,
        "app_train_manifest",
        fallback=report_file_current(report, "app_val_manifest"),
    ) or not report_file_current(report, "app_val_manifest"):
        train_manifest = report.get("app_train_manifest", {})
        val_manifest = report.get("app_val_manifest", {})
        if train_manifest.get("exists") or val_manifest.get("exists"):
            stale_status = (
                train_manifest
                if not report_file_current(report, "app_train_manifest")
                else val_manifest
            )
            return [stale_artifact_message(stale_status, commands["split_app_trainable_manifest"])]
        return [commands["split_app_trainable_manifest"]]
    if not report_file_current(report, "app_eval_manifest"):
        if report["app_eval_manifest"].get("exists"):
            return [stale_artifact_message(report["app_eval_manifest"], commands["merge_app_eval_manifest"])]
        return [commands["merge_app_eval_manifest"]]
    if report["app_validation_audit"].get("exists") and not report["app_validation_audit"].get(
        "manifest_match", True
    ):
        failed = report["app_validation_audit"].get("manifest_failed_checks") or []
        return [
            "App validation audit is stale for the current app evaluation manifest; "
            f"failed_checks={failed}. Re-run: {commands['audit_app_validation']}"
        ]
    if not report["app_validation_audit"]["ready_for_mvp_validation"]:
        return [commands["audit_app_validation"]]
    if not report["app_baseline_eval"]["complete"]:
        if not report["app_baseline_eval"]["missing_artifacts"] and report["app_baseline_eval"].get(
            "artifact_verification"
        ):
            return [commands["verify_app_baseline_eval"], commands["evaluate_app_baseline"]]
        return [commands["evaluate_app_baseline"]]
    if not report_file_current(report, "app_adapted_train_manifest"):
        if report["app_adapted_train_manifest"].get("exists"):
            return [
                stale_artifact_message(
                    report["app_adapted_train_manifest"],
                    commands["merge_app_adapted_train_manifest"],
                )
            ]
        return [commands["merge_app_adapted_train_manifest"]]
    if not report["app_adapted_plan"]["ok"]:
        return [commands["plan_app_adapted_candidate"]]
    if not report["app_adapted_run"]["complete"]:
        return [commands["train_app_adapted_candidate"]]
    if not report["app_adapted_eval"]["complete"]:
        if not report["app_adapted_eval"]["missing_artifacts"] and report["app_adapted_eval"].get(
            "artifact_verification"
        ):
            return [commands["verify_app_adapted_eval"], commands["evaluate_app_adapted_candidate"]]
        return [commands["evaluate_app_adapted_candidate"]]
    if not report["app_adapted_comparison"]["exists"]:
        return [commands["compare_app_adapted_runs"]]
    if report["app_adapted_comparison"].get("candidate_found") is False:
        error = report["app_adapted_comparison"].get("error")
        return [
            "App-adapted comparison report does not contain the configured candidate evaluation directory; "
            f"error={error}. Re-run: {commands['compare_app_adapted_runs']}"
        ]
    if not report["app_adapted_comparison"].get("evaluation_artifact_match", True):
        failed = report["app_adapted_comparison"].get("evaluation_artifact_failed_checks") or []
        return [
            "App-adapted comparison report is stale for the current candidate evaluation artifacts; "
            f"failed_checks={failed}. Re-run: {commands['compare_app_adapted_runs']}"
        ]
    if not report["app_adapted_comparison"]["candidate_eligible"]:
        return [
            candidate_failed_message("Full VocalSet", report["comparison"])
            + " "
            + candidate_failed_message("Balanced VocalSet", report.get("balanced_comparison", {}))
            + " "
            + candidate_failed_message("App-adapted", report["app_adapted_comparison"])
            + " Add more labeled app recordings or adjust the training recipe before packaging."
        ]
    return [selected_package_command(report, commands)]


def current_stage(report: dict[str, Any]) -> dict[str, Any]:
    preflight = report["preflight"]
    if not preflight["ok"]:
        return {
            "name": "blocked_on_preflight",
            "ready": False,
            "detail": {"required_failures": preflight.get("required_failures") or []},
        }
    if not report["baseline_plan"]["ok"]:
        return {
            "name": "plan_gtsinger_baseline",
            "ready": True,
            "detail": {"plan_errors": report["baseline_plan"].get("errors") or []},
        }
    if not report["baseline_run"]["complete"]:
        return {"name": "train_gtsinger_baseline", "ready": True, "detail": {}}
    if not report["baseline_eval"]["complete"]:
        return {
            "name": "evaluate_gtsinger_baseline",
            "ready": True,
            "detail": {
                "missing_eval_artifacts": report["baseline_eval"]["missing_artifacts"],
                "failed_eval_checks": (
                    report["baseline_eval"].get("artifact_verification", {}) or {}
                ).get("failed_checks", []),
            },
        }
    if not report["vocalset_manifest"]["exists"]:
        return {"name": "build_vocalset_manifest", "ready": True, "detail": {}}
    if not report["vocalset_plan"]["ok"]:
        return {
            "name": "plan_vocalset_candidate",
            "ready": True,
            "detail": {"plan_errors": report["vocalset_plan"].get("errors") or []},
        }
    if not report["vocalset_run"]["complete"]:
        return {"name": "train_vocalset_candidate", "ready": True, "detail": {}}
    if not report["vocalset_eval"]["complete"]:
        return {
            "name": "evaluate_vocalset_candidate",
            "ready": True,
            "detail": {
                "missing_eval_artifacts": report["vocalset_eval"]["missing_artifacts"],
                "failed_eval_checks": (
                    report["vocalset_eval"].get("artifact_verification", {}) or {}
                ).get("failed_checks", []),
            },
        }
    if not report["comparison"]["exists"]:
        return {"name": "compare_candidates", "ready": True, "detail": {}}
    if report["comparison"].get("candidate_found") is False:
        return {
            "name": "comparison_candidate_missing",
            "ready": True,
            "detail": {
                "error": report["comparison"].get("error"),
                "path": report["comparison"].get("path"),
            },
        }
    if not report["comparison"].get("evaluation_artifact_match", True):
        return {
            "name": "comparison_evaluation_artifacts_stale",
            "ready": True,
            "detail": {
                "failed_checks": report["comparison"].get("evaluation_artifact_failed_checks") or [],
            },
        }
    if not report["comparison"]["candidate_eligible"]:
        if "vocalset_balanced_manifest" not in report:
            return {
                "name": "candidate_not_promotion_eligible",
                "ready": False,
                "detail": {
                    "failed_gates": report["comparison"].get("failed_gates") or [],
                    "unknown_gates": report["comparison"].get("unknown_gates") or [],
                },
            }
        if not report["vocalset_balanced_manifest"]["exists"]:
            return {"name": "build_balanced_vocalset_manifest", "ready": True, "detail": {}}
        if not report["vocalset_balanced_plan"]["ok"]:
            return {
                "name": "plan_balanced_vocalset_candidate",
                "ready": True,
                "detail": {"plan_errors": report["vocalset_balanced_plan"].get("errors") or []},
            }
        if not report["vocalset_balanced_run"]["complete"]:
            return {"name": "train_balanced_vocalset_candidate", "ready": True, "detail": {}}
        if not report["vocalset_balanced_eval"]["complete"]:
            return {
                "name": "evaluate_balanced_vocalset_candidate",
                "ready": True,
                "detail": {
                    "missing_eval_artifacts": report["vocalset_balanced_eval"]["missing_artifacts"],
                    "failed_eval_checks": (
                        report["vocalset_balanced_eval"].get("artifact_verification", {}) or {}
                    ).get("failed_checks", []),
                },
            }
        if not report["balanced_comparison"]["exists"]:
            return {"name": "compare_balanced_candidates", "ready": True, "detail": {}}
        if report["balanced_comparison"].get("candidate_found") is False:
            return {
                "name": "balanced_comparison_candidate_missing",
                "ready": True,
                "detail": {
                    "error": report["balanced_comparison"].get("error"),
                    "path": report["balanced_comparison"].get("path"),
                },
            }
        if not report["balanced_comparison"].get("evaluation_artifact_match", True):
            return {
                "name": "balanced_comparison_evaluation_artifacts_stale",
                "ready": True,
                "detail": {
                    "failed_checks": report["balanced_comparison"].get("evaluation_artifact_failed_checks") or [],
                },
            }
        if not report["balanced_comparison"]["candidate_eligible"] and not report["app_manifest"]["exists"]:
            app_labels = preflight_check(report, "app_recording_labels")
            if app_labels and not app_labels.get("ok"):
                if not report.get("app_collection_plan", {}).get("exists"):
                    return {
                        "name": "plan_app_recording_collection",
                        "ready": True,
                        "detail": {
                            "full_vocalset_failed_gates": report["comparison"].get("failed_gates") or [],
                            "balanced_vocalset_failed_gates": report["balanced_comparison"].get("failed_gates") or [],
                            "app_labels": app_labels.get("detail"),
                            "collection_plan": report.get("app_collection_plan", {}),
                        },
                    }
                if app_prepare_report_message(report.get("app_prepare_report", {})):
                    return {
                        "name": "collect_or_fix_app_recording_labels",
                        "ready": False,
                        "detail": {
                            "full_vocalset_failed_gates": report["comparison"].get("failed_gates") or [],
                            "balanced_vocalset_failed_gates": report["balanced_comparison"].get("failed_gates") or [],
                            "app_labels": app_labels.get("detail"),
                            "prepare_report": report.get("app_prepare_report", {}),
                            "collection_materialize": report.get("app_collection_materialize", {}),
                        },
                    }
                if needs_app_collection_materialization(report):
                    return {
                        "name": "materialize_app_collection",
                        "ready": True,
                        "detail": {
                            "full_vocalset_failed_gates": report["comparison"].get("failed_gates") or [],
                            "balanced_vocalset_failed_gates": report["balanced_comparison"].get("failed_gates") or [],
                            "collection_plan": report.get("app_collection_plan", {}),
                            "collection_materialize": report.get("app_collection_materialize", {}),
                        },
                    }
                if needs_app_collection_packet_export(report):
                    return {
                        "name": "export_app_collection_packet",
                        "ready": True,
                        "detail": {
                            "full_vocalset_failed_gates": report["comparison"].get("failed_gates") or [],
                            "balanced_vocalset_failed_gates": report["balanced_comparison"].get("failed_gates") or [],
                            "collection_materialize": report.get("app_collection_materialize", {}),
                            "collection_packet": report.get("app_collection_packet", {}),
                        },
                    }
                return {
                    "name": "collect_or_fix_app_recording_labels",
                    "ready": False,
                    "detail": {
                        "full_vocalset_failed_gates": report["comparison"].get("failed_gates") or [],
                        "balanced_vocalset_failed_gates": report["balanced_comparison"].get("failed_gates") or [],
                        "app_labels": app_labels.get("detail"),
                        "prepare_report": report.get("app_prepare_report", {}),
                        "collection_materialize": report.get("app_collection_materialize", {}),
                    },
                }
            if not report.get("app_label_coverage", {}).get("ready_for_collection_target", True):
                return {
                    "name": "fix_app_label_coverage",
                    "ready": False,
                    "detail": {
                        "full_vocalset_failed_gates": report["comparison"].get("failed_gates") or [],
                        "balanced_vocalset_failed_gates": report["balanced_comparison"].get("failed_gates") or [],
                        "coverage": report.get("app_label_coverage", {}),
                    },
                }
            return {
                "name": "build_app_manifest",
                "ready": True,
                "detail": {
                    "full_vocalset_failed_gates": report["comparison"].get("failed_gates") or [],
                    "balanced_vocalset_failed_gates": report["balanced_comparison"].get("failed_gates") or [],
                },
            }
    if not report_file_current(report, "app_manifest"):
        if report["app_manifest"].get("exists"):
            return {
                "name": "app_manifest_stale",
                "ready": True,
                "detail": {
                    "stale_source_files": report["app_manifest"].get("stale_source_files") or [],
                    "missing_source_files": report["app_manifest"].get("missing_source_files") or [],
                },
            }
        app_labels = preflight_check(report, "app_recording_labels")
        if app_labels and not app_labels.get("ok"):
            if not report.get("app_collection_plan", {}).get("exists"):
                return {
                    "name": "plan_app_recording_collection",
                    "ready": True,
                    "detail": {
                        "app_labels": app_labels.get("detail"),
                        "collection_plan": report.get("app_collection_plan", {}),
                    },
                }
            if app_prepare_report_message(report.get("app_prepare_report", {})):
                return {
                    "name": "collect_or_fix_app_recording_labels",
                    "ready": False,
                    "detail": {
                        "app_labels": app_labels.get("detail"),
                        "prepare_report": report.get("app_prepare_report", {}),
                        "collection_materialize": report.get("app_collection_materialize", {}),
                    },
                }
            if needs_app_collection_materialization(report):
                return {
                    "name": "materialize_app_collection",
                    "ready": True,
                    "detail": {
                        "app_labels": app_labels.get("detail"),
                        "collection_plan": report.get("app_collection_plan", {}),
                        "collection_materialize": report.get("app_collection_materialize", {}),
                    },
                }
            if needs_app_collection_packet_export(report):
                return {
                    "name": "export_app_collection_packet",
                    "ready": True,
                    "detail": {
                        "app_labels": app_labels.get("detail"),
                        "collection_materialize": report.get("app_collection_materialize", {}),
                        "collection_packet": report.get("app_collection_packet", {}),
                    },
                }
            return {
                "name": "collect_or_fix_app_recording_labels",
                "ready": False,
                "detail": {
                    "app_labels": app_labels.get("detail"),
                    "prepare_report": report.get("app_prepare_report", {}),
                    "collection_materialize": report.get("app_collection_materialize", {}),
                },
            }
        if not report.get("app_label_coverage", {}).get("ready_for_collection_target", True):
            return {
                "name": "fix_app_label_coverage",
                "ready": False,
                "detail": report.get("app_label_coverage", {}),
            }
        return {"name": "build_app_manifest", "ready": True, "detail": {}}
    if not report_file_current(report, "app_trainable_manifest") or not report_file_current(
        report,
        "app_eval_only_manifest",
    ):
        if report["app_trainable_manifest"].get("exists") or report["app_eval_only_manifest"].get("exists"):
            return {
                "name": "filtered_app_manifest_stale",
                "ready": True,
                "detail": {
                    "trainable": {
                        "stale_source_files": report["app_trainable_manifest"].get("stale_source_files") or [],
                        "missing_source_files": report["app_trainable_manifest"].get("missing_source_files") or [],
                    },
                    "eval_only": {
                        "stale_source_files": report["app_eval_only_manifest"].get("stale_source_files") or [],
                        "missing_source_files": report["app_eval_only_manifest"].get("missing_source_files") or [],
                    },
                },
            }
        return {"name": "filter_app_manifest", "ready": True, "detail": {}}
    if not report_file_current(
        report,
        "app_train_manifest",
        fallback=report_file_current(report, "app_val_manifest"),
    ) or not report_file_current(report, "app_val_manifest"):
        if report["app_train_manifest"].get("exists") or report["app_val_manifest"].get("exists"):
            return {
                "name": "app_train_val_split_stale",
                "ready": True,
                "detail": {
                    "train": {
                        "stale_source_files": report["app_train_manifest"].get("stale_source_files") or [],
                        "missing_source_files": report["app_train_manifest"].get("missing_source_files") or [],
                    },
                    "val": {
                        "stale_source_files": report["app_val_manifest"].get("stale_source_files") or [],
                        "missing_source_files": report["app_val_manifest"].get("missing_source_files") or [],
                    },
                },
            }
        return {"name": "split_app_trainable_manifest", "ready": True, "detail": {}}
    if not report_file_current(report, "app_eval_manifest"):
        if report["app_eval_manifest"].get("exists"):
            return {
                "name": "app_eval_manifest_stale",
                "ready": True,
                "detail": {
                    "stale_source_files": report["app_eval_manifest"].get("stale_source_files") or [],
                    "missing_source_files": report["app_eval_manifest"].get("missing_source_files") or [],
                },
            }
        return {"name": "merge_app_eval_manifest", "ready": True, "detail": {}}
    if report["app_validation_audit"].get("exists") and not report["app_validation_audit"].get(
        "manifest_match", True
    ):
        return {
            "name": "app_validation_audit_stale",
            "ready": True,
            "detail": {
                "failed_checks": report["app_validation_audit"].get("manifest_failed_checks") or [],
            },
        }
    if not report["app_validation_audit"]["ready_for_mvp_validation"]:
        return {
            "name": "audit_app_validation",
            "ready": True,
            "detail": {
                "warnings": report["app_validation_audit"].get("warnings") or [],
                "missing_target_families": report["app_validation_audit"].get("missing_target_families") or {},
                "negative_shortfall": report["app_validation_audit"].get("negative_shortfall"),
                "group_shortfall": report["app_validation_audit"].get("group_shortfall"),
            },
        }
    if not report["app_baseline_eval"]["complete"]:
        return {
            "name": "evaluate_app_baseline",
            "ready": True,
            "detail": {
                "missing_eval_artifacts": report["app_baseline_eval"]["missing_artifacts"],
                "failed_eval_checks": (
                    report["app_baseline_eval"].get("artifact_verification", {}) or {}
                ).get("failed_checks", []),
            },
        }
    if not report_file_current(report, "app_adapted_train_manifest"):
        if report["app_adapted_train_manifest"].get("exists"):
            return {
                "name": "app_adapted_train_manifest_stale",
                "ready": True,
                "detail": {
                    "stale_source_files": report["app_adapted_train_manifest"].get("stale_source_files") or [],
                    "missing_source_files": report["app_adapted_train_manifest"].get("missing_source_files") or [],
                },
            }
        return {"name": "merge_app_adapted_train_manifest", "ready": True, "detail": {}}
    if not report["app_adapted_plan"]["ok"]:
        return {
            "name": "plan_app_adapted_candidate",
            "ready": True,
            "detail": {"plan_errors": report["app_adapted_plan"].get("errors") or []},
        }
    if not report["app_adapted_run"]["complete"]:
        return {
            "name": "train_app_adapted_candidate",
            "ready": True,
            "detail": {
                "full_vocalset_failed_gates": report["comparison"].get("failed_gates") or [],
                "balanced_vocalset_failed_gates": report.get("balanced_comparison", {}).get("failed_gates") or [],
            },
        }
    if not report["app_adapted_eval"]["complete"]:
        return {
            "name": "evaluate_app_adapted_candidate",
            "ready": True,
            "detail": {
                "missing_eval_artifacts": report["app_adapted_eval"]["missing_artifacts"],
                "failed_eval_checks": (
                    report["app_adapted_eval"].get("artifact_verification", {}) or {}
                ).get("failed_checks", []),
            },
        }
    if not report["app_adapted_comparison"]["exists"]:
        return {"name": "compare_app_adapted_candidates", "ready": True, "detail": {}}
    if report["app_adapted_comparison"].get("candidate_found") is False:
        return {
            "name": "app_adapted_comparison_candidate_missing",
            "ready": True,
            "detail": {
                "error": report["app_adapted_comparison"].get("error"),
                "path": report["app_adapted_comparison"].get("path"),
            },
        }
    if not report["app_adapted_comparison"].get("evaluation_artifact_match", True):
        return {
            "name": "app_adapted_comparison_evaluation_artifacts_stale",
            "ready": True,
            "detail": {
                "failed_checks": report["app_adapted_comparison"].get("evaluation_artifact_failed_checks") or [],
            },
        }
    if not report["app_adapted_comparison"]["candidate_eligible"]:
        return {
            "name": "app_adapted_candidate_not_promotion_eligible",
            "ready": False,
            "detail": {
                "full_vocalset_failed_gates": report["comparison"].get("failed_gates") or [],
                "balanced_vocalset_failed_gates": report.get("balanced_comparison", {}).get("failed_gates") or [],
                "app_adapted_failed_gates": report["app_adapted_comparison"].get("failed_gates") or [],
            },
        }
    if report.get("app_adapted_comparison", {}).get("candidate_eligible"):
        return {"name": "package_app_adapted_candidate", "ready": True, "detail": {}}
    if report.get("balanced_comparison", {}).get("candidate_eligible"):
        return {"name": "package_balanced_candidate", "ready": True, "detail": {}}
    return {"name": "package_candidate", "ready": True, "detail": {}}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    vocalset_balanced_manifest = args_value(
        args,
        "vocalset_balanced_manifest",
        "./gt_singer_grader/manifests/vocalset_balanced_120.jsonl",
    )
    vocalset_balanced_plan = args_value(
        args,
        "vocalset_balanced_plan",
        "./gt_singer_grader/runs/gtsinger_vocalset_balanced120_song_v2/training_plan.json",
    )
    vocalset_balanced_run_dir = args_value(
        args,
        "vocalset_balanced_run_dir",
        "./gt_singer_grader/runs/gtsinger_vocalset_balanced120_song_v2",
    )
    vocalset_balanced_eval_dir = args_value(
        args,
        "vocalset_balanced_eval_dir",
        "./gt_singer_grader/runs/gtsinger_vocalset_balanced120_song_v2/eval_val",
    )
    balanced_comparison = args_value(
        args,
        "balanced_comparison",
        "./gt_singer_grader/runs/run_comparison_balanced120_v2.json",
    )
    app_train_manifest = args_value(args, "app_train_manifest", "./gt_singer_grader/manifests/app_recordings_train.jsonl")
    app_audio_root = args_value(args, "app_audio_root", ".")
    app_prepare_report = args_value(
        args,
        "app_prepare_report",
        str(Path(args.app_labels).with_name("prepare_report.json")),
    )
    app_collection_plan = args_value(
        args,
        "app_collection_plan",
        "./gt_singer_grader/data/app_recordings/collection_plan.csv",
    )
    app_collection_plan_json = args_value(
        args,
        "app_collection_plan_json",
        "./gt_singer_grader/data/app_recordings/collection_plan.json",
    )
    app_collection_checklist = args_value(
        args,
        "app_collection_checklist",
        "./gt_singer_grader/data/app_recordings/collection_checklist.csv",
    )
    app_collection_missing = args_value(
        args,
        "app_collection_missing",
        "./gt_singer_grader/data/app_recordings/collection_missing.csv",
    )
    app_collection_materialize_report = args_value(
        args,
        "app_collection_materialize_report",
        "./gt_singer_grader/data/app_recordings/collection_materialize_report.json",
    )
    app_collection_packet_dir = args_value(
        args,
        "app_collection_packet_dir",
        "./gt_singer_grader/data/app_recordings/collection_packet",
    )
    app_collection_packet_summary = args_value(
        args,
        "app_collection_packet_summary",
        "./gt_singer_grader/data/app_recordings/collection_packet_summary.json",
    )
    app_adapted_train_manifest = args_value(
        args,
        "app_adapted_train_manifest",
        "./gt_singer_grader/manifests/app_adapted_train.jsonl",
    )
    app_baseline_eval_dir = args_value(args, "app_baseline_eval_dir", "./gt_singer_grader/runs/gtsinger_song_aug_v1/eval_app")
    app_adapted_plan = args_value(
        args,
        "app_adapted_plan",
        "./gt_singer_grader/runs/gtsinger_app_adapted_v1/training_plan.json",
    )
    app_adapted_run_dir = args_value(args, "app_adapted_run_dir", "./gt_singer_grader/runs/gtsinger_app_adapted_v1")
    app_adapted_eval_dir = args_value(args, "app_adapted_eval_dir", "./gt_singer_grader/runs/gtsinger_app_adapted_v1/eval_app")
    app_adapted_comparison = args_value(
        args,
        "app_adapted_comparison",
        "./gt_singer_grader/runs/run_comparison_app_adapted.json",
    )
    preflight = build_preflight_report(
        Namespace(
            gtsinger_root=args.gtsinger_root,
            vocalset_root=args.vocalset_root,
            app_audio_dir=args_value(args, "app_audio_dir", "./gt_singer_grader/data/app_recordings/raw"),
            app_audio_root=app_audio_root,
            app_labels=args.app_labels,
            app_collection_plan=app_collection_plan,
            app_collection_plan_json=app_collection_plan_json,
            app_collection_root=args_value(args, "app_collection_root", "./gt_singer_grader/data/app_recordings"),
            app_collection_checklist=app_collection_checklist,
            app_collection_missing=app_collection_missing,
            app_collection_materialize_report=app_collection_materialize_report,
            app_collection_packet_dir=app_collection_packet_dir,
            app_collection_packet_summary=app_collection_packet_summary,
            checkpoint=args.checkpoint,
            metadata=args.metadata,
        )
    )
    report: dict[str, Any] = {
        "preflight": preflight,
        "baseline_plan": training_plan_status(args.baseline_plan),
        "baseline_run": training_run_status(args.baseline_run_dir),
        "baseline_eval": evaluation_status(args.baseline_eval_dir),
        "vocalset_manifest": file_status(args.vocalset_manifest),
        "vocalset_plan": training_plan_status(args.vocalset_plan),
        "vocalset_run": training_run_status(args.vocalset_run_dir),
        "vocalset_eval": evaluation_status(args.vocalset_eval_dir),
        "comparison": comparison_status(args.comparison, args.vocalset_eval_dir),
        "vocalset_balanced_manifest": file_status(vocalset_balanced_manifest),
        "vocalset_balanced_plan": training_plan_status(vocalset_balanced_plan),
        "vocalset_balanced_run": training_run_status(vocalset_balanced_run_dir),
        "vocalset_balanced_eval": evaluation_status(vocalset_balanced_eval_dir),
        "balanced_comparison": comparison_status(balanced_comparison, vocalset_balanced_eval_dir),
        "app_collection_plan": app_collection_plan_status(app_collection_plan, app_collection_plan_json),
        "app_collection_materialize": app_collection_materialize_status(
            app_collection_checklist,
            app_collection_materialize_report,
            app_collection_missing,
        ),
        "app_collection_packet": app_collection_packet_status(
            app_collection_packet_dir,
            app_collection_packet_summary,
            app_collection_checklist,
        ),
        "app_prepare_report": app_prepare_report_status(app_prepare_report),
        "app_label_coverage": app_label_coverage_status(args.app_labels, app_audio_root),
        "app_manifest": file_status(args.app_manifest, [args.app_labels]),
        "app_trainable_manifest": file_status(args.app_trainable_manifest, [args.app_manifest]),
        "app_eval_only_manifest": file_status(args.app_eval_only_manifest, [args.app_manifest]),
        "app_train_manifest": file_status(app_train_manifest, [args.app_trainable_manifest]),
        "app_val_manifest": file_status(args.app_val_manifest, [args.app_trainable_manifest]),
        "app_eval_manifest": file_status(args.app_eval_manifest, [args.app_val_manifest, args.app_eval_only_manifest]),
        "app_validation_audit": app_audit_status(args.app_validation_audit, args.app_eval_manifest),
        "app_adapted_train_manifest": file_status(
            app_adapted_train_manifest,
            [Path(args.baseline_run_dir) / "train_manifest.jsonl", app_train_manifest],
        ),
        "app_baseline_eval": evaluation_status(app_baseline_eval_dir),
        "app_adapted_plan": training_plan_status(app_adapted_plan),
        "app_adapted_run": training_run_status(app_adapted_run_dir),
        "app_adapted_eval": evaluation_status(app_adapted_eval_dir),
        "app_adapted_comparison": comparison_status(app_adapted_comparison, app_adapted_eval_dir),
    }
    report["next_actions"] = next_actions(report, command_list(args))
    report["current_stage"] = current_stage(report)
    report["ready_for_packaging_review"] = (
        report["preflight"]["ok"]
        and report["baseline_plan"]["ok"]
        and report["baseline_run"]["complete"]
        and report["baseline_eval"]["complete"]
        and report["vocalset_plan"]["ok"]
        and report["vocalset_run"]["complete"]
        and report["vocalset_eval"]["complete"]
        and report["app_baseline_eval"]["complete"]
        and report_file_current(report, "app_adapted_train_manifest")
        and report["app_adapted_plan"]["ok"]
        and report["app_adapted_run"]["complete"]
        and report["app_adapted_eval"]["complete"]
        and report["app_adapted_comparison"]["candidate_eligible"]
        and report["app_adapted_comparison"].get("evaluation_artifact_match", True)
        and report_file_current(report, "app_train_manifest")
        and report_file_current(report, "app_eval_manifest")
        and report["app_validation_audit"]["ready_for_mvp_validation"]
        and report["app_validation_audit"].get("manifest_match", True)
    )
    return report


def main() -> None:
    args = parse_args()
    report = build_report(args)
    output = json.dumps(report, indent=2, sort_keys=True)
    print(output)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
    if args.strict and not report["ready_for_packaging_review"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
