"""Audit app-recording reviewer CSV coverage before building training runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .audit_app_validation import DEFAULT_TARGET_FAMILIES, audit_records
from .build_manifest import build_app_recordings_manifest
from .manifest import record_families, validate_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report reviewer CSV coverage for app-recording collection")
    parser.add_argument("--csv", required=True, help="App recording reviewer CSV")
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--audio-root",
        default=".",
        help="Base directory for relative audio_path values when --require-audio-files is set",
    )
    parser.add_argument(
        "--require-audio-files",
        action="store_true",
        help="Require every reviewed audio_path to resolve to an existing file",
    )
    parser.add_argument("--min-per-family", type=int, default=20)
    parser.add_argument("--min-negative", type=int, default=20)
    parser.add_argument("--min-groups", type=int, default=3)
    parser.add_argument(
        "--target-family",
        action="append",
        default=[],
        help="Target family to require. Defaults to every non-control technique family.",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when app label coverage is not ready")
    return parser.parse_args()


def build_report(
    csv_path: str,
    *,
    target_families: list[str] | tuple[str, ...] = DEFAULT_TARGET_FAMILIES,
    min_per_family: int = 20,
    min_negative: int = 20,
    min_groups: int = 3,
    require_audio_files: bool = False,
    audio_root: str | Path = ".",
) -> dict[str, Any]:
    records = build_app_recordings_manifest(
        csv_path,
        "app_recordings",
        "app_user",
        allow_duplicate_review_rows=True,
    )
    audit = audit_records(
        records,
        target_families=target_families,
        min_per_family=min_per_family,
        min_negative=min_negative,
        min_groups=min_groups,
    )
    schema_errors: list[str] = []
    unlabeled_records = 0
    missing_audio_files = _missing_audio_files(records, audio_root) if require_audio_files else []
    duplicate_audio_paths = _duplicate_values(records, "audio_path")
    duplicate_recording_ids = _duplicate_values(records, "recording_id")
    intended_mismatches = _intended_family_mismatches(records)
    missing_reviewer_ids = _missing_reviewer_ids(records)
    review_progress = _review_progress(records)
    for index, record in enumerate(records, start=1):
        schema_errors.extend(validate_record(record, line_number=index))
        if not record_families(record):
            unlabeled_records += 1

    warnings = list(audit.get("warnings") or [])
    if unlabeled_records:
        warnings.append("review CSV has unlabeled rows")
    if schema_errors:
        warnings.append("review CSV has schema validation errors")
    if missing_audio_files:
        warnings.append("review CSV references missing audio files")
    if duplicate_audio_paths:
        warnings.append("review CSV has duplicate audio_path values")
    if duplicate_recording_ids:
        warnings.append("review CSV has duplicate recording_id values")
    if intended_mismatches:
        warnings.append("review CSV has intended_family/reviewer-label mismatches")
    if missing_reviewer_ids:
        warnings.append("review CSV has labeled rows without reviewer_id")

    return {
        **audit,
        "ready_for_collection_target": not warnings,
        "source_csv": str(Path(csv_path)),
        "audio_root": str(Path(audio_root)),
        "audio_files_checked": require_audio_files,
        "missing_audio_files": missing_audio_files,
        "missing_audio_file_count": len(missing_audio_files),
        "unlabeled_records": unlabeled_records,
        "duplicate_audio_paths": duplicate_audio_paths,
        "duplicate_recording_ids": duplicate_recording_ids,
        "intended_family_mismatches": intended_mismatches,
        "intended_family_mismatch_count": len(intended_mismatches),
        "missing_reviewer_ids": missing_reviewer_ids,
        "missing_reviewer_id_count": len(missing_reviewer_ids),
        "review_progress": review_progress,
        "schema_errors": schema_errors,
        "warnings": sorted(set(warnings)),
    }


def _missing_audio_files(records: list[dict[str, Any]], audio_root: str | Path) -> list[str]:
    root = Path(audio_root)
    missing: list[str] = []
    for record in records:
        audio_path = str(record.get("audio_path") or "")
        path = Path(audio_path)
        resolved = path if path.is_absolute() else root / path
        if not resolved.is_file():
            missing.append(audio_path)
    return sorted(missing)


def _duplicate_values(records: list[dict[str, Any]], field: str) -> list[str]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(field) or "").strip()
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return sorted(value for value, count in counts.items() if count > 1)


def _intended_family_mismatches(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for record in records:
        labels = record.get("labels") if isinstance(record.get("labels"), dict) else {}
        intended = str(labels.get("intended_family") or "").strip()
        if not intended:
            continue
        reviewed = record_families(record)
        reviewed_set = set(reviewed)
        if intended in {"none", "unclear"}:
            continue
        if intended == "control":
            mismatch = bool(reviewed_set - {"control", "none", "unclear"})
        else:
            mismatch = intended not in reviewed_set
        if mismatch:
            mismatches.append(
                {
                    "recording_id": record.get("recording_id"),
                    "audio_path": record.get("audio_path"),
                    "intended_family": intended,
                    "reviewed_families": reviewed,
                }
            )
    return mismatches


def _missing_reviewer_ids(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for record in records:
        if not record_families(record):
            continue
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        if str(metadata.get("reviewer_id") or "").strip():
            continue
        missing.append(
            {
                "recording_id": record.get("recording_id"),
                "audio_path": record.get("audio_path"),
                "families": record_families(record),
            }
        )
    return missing


def _counter_template() -> dict[str, int]:
    return {
        "records": 0,
        "labeled_records": 0,
        "unlabeled_records": 0,
        "missing_reviewer_id_records": 0,
    }


def _review_progress(records: list[dict[str, Any]]) -> dict[str, Any]:
    totals = _counter_template()
    by_intended_family: dict[str, dict[str, int]] = {}
    by_split_group: dict[str, dict[str, int]] = {}

    for record in records:
        labels = record.get("labels") if isinstance(record.get("labels"), dict) else {}
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        intended = str(labels.get("intended_family") or "").strip() or "unspecified"
        split_group = str(record.get("split_group") or metadata.get("singer_id") or "unspecified").strip() or "unspecified"
        families = record_families(record)
        reviewer_id = str(metadata.get("reviewer_id") or "").strip()

        buckets = [
            totals,
            by_intended_family.setdefault(intended, _counter_template()),
            by_split_group.setdefault(split_group, _counter_template()),
        ]
        for bucket in buckets:
            bucket["records"] += 1
            if families:
                bucket["labeled_records"] += 1
                if not reviewer_id:
                    bucket["missing_reviewer_id_records"] += 1
            else:
                bucket["unlabeled_records"] += 1

    return {
        **totals,
        "by_intended_family": dict(sorted(by_intended_family.items())),
        "by_split_group": dict(sorted(by_split_group.items())),
    }


def main() -> None:
    args = parse_args()
    target_families = args.target_family or list(DEFAULT_TARGET_FAMILIES)
    report = build_report(
        args.csv,
        target_families=target_families,
        min_per_family=args.min_per_family,
        min_negative=args.min_negative,
        min_groups=args.min_groups,
        require_audio_files=args.require_audio_files,
        audio_root=args.audio_root,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    if args.strict and not report["ready_for_collection_target"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
