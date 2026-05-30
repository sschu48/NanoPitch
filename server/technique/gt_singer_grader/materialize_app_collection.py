"""Create app-recording collection folders and a recording checklist."""

from __future__ import annotations

import argparse
import csv
import json
import wave
from pathlib import Path
from typing import Any


CHECKLIST_FIELDS = (
    "plan_id",
    "singer_id",
    "intended_family",
    "expected_audio_path",
    "directory",
    "filename",
    "exists",
    "review_goal",
    "minimum_review_strength",
    "notes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize app-recording collection folders from a plan CSV")
    parser.add_argument("--plan", required=True, help="Collection plan CSV from plan_app_collection")
    parser.add_argument(
        "--root",
        default="./gt_singer_grader/data/app_recordings",
        help="Base directory for plan suggested_filename values such as raw/<singer>/<take>.wav.",
    )
    parser.add_argument(
        "--checklist",
        default=None,
        help="Optional checklist CSV path. Defaults to <root>/collection_checklist.csv.",
    )
    parser.add_argument(
        "--missing-csv",
        default=None,
        help="Optional CSV path for only planned audio files that are still missing.",
    )
    parser.add_argument("--report-json", default=None, help="Optional JSON report path")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if the plan is empty or has duplicate suggested filenames.",
    )
    parser.add_argument(
        "--require-audio-files",
        action="store_true",
        help="Exit non-zero if any planned WAV file is still missing.",
    )
    parser.add_argument(
        "--validate-wav-files",
        action="store_true",
        help="Validate existing planned files as readable WAV audio before marking the collection review-ready.",
    )
    parser.add_argument("--min-wav-seconds", type=float, default=5.0, help="Minimum valid WAV duration when validating")
    parser.add_argument("--max-wav-seconds", type=float, default=10.0, help="Maximum valid WAV duration when validating")
    return parser.parse_args()


def read_plan(path: str | Path) -> list[dict[str, str]]:
    plan_path = Path(path)
    if not plan_path.is_file():
        raise FileNotFoundError(f"collection plan not found: {plan_path}")
    with plan_path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _normalized_relative_path(value: str) -> Path:
    stripped = value.strip()
    if not stripped:
        raise ValueError("suggested_filename is required")
    path = Path(stripped)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"suggested_filename must be a safe relative path: {value}")
    return Path(path.as_posix().lstrip("./"))


def materialize_collection(
    plan_rows: list[dict[str, str]],
    *,
    root: str | Path,
    checklist_path: str | Path | None = None,
    missing_csv_path: str | Path | None = None,
    validate_wav_files: bool = False,
    min_wav_seconds: float = 5.0,
    max_wav_seconds: float = 10.0,
) -> dict[str, Any]:
    root_path = Path(root)
    checklist = Path(checklist_path) if checklist_path else root_path / "collection_checklist.csv"
    checklist.parent.mkdir(parents=True, exist_ok=True)
    missing_csv = Path(missing_csv_path) if missing_csv_path else root_path / "collection_missing.csv"
    missing_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    groups: set[str] = set()
    directories: set[Path] = set()
    family_counts: dict[str, int] = {}
    duplicate_suggestions = _duplicate_suggestions(plan_rows)

    for row in plan_rows:
        relative_path = _normalized_relative_path(row.get("suggested_filename") or "")
        expected_audio_path = root_path / relative_path
        expected_audio_path.parent.mkdir(parents=True, exist_ok=True)
        directories.add(expected_audio_path.parent)

        singer_id = (row.get("singer_id") or "").strip()
        if singer_id:
            groups.add(singer_id)
        family = (row.get("intended_family") or "").strip()
        if family:
            family_counts[family] = family_counts.get(family, 0) + 1

        rows.append(
            {
                "plan_id": (row.get("plan_id") or "").strip(),
                "singer_id": singer_id,
                "intended_family": family,
                "expected_audio_path": expected_audio_path.as_posix(),
                "directory": expected_audio_path.parent.as_posix(),
                "filename": expected_audio_path.name,
                "exists": "yes" if expected_audio_path.is_file() else "no",
                "review_goal": (row.get("review_goal") or "").strip(),
                "minimum_review_strength": (row.get("minimum_review_strength") or "").strip(),
                "notes": (row.get("notes") or "").strip(),
            }
        )

    with checklist.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CHECKLIST_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    existing_audio = sum(1 for row in rows if row["exists"] == "yes")
    missing_audio = len(rows) - existing_audio
    missing_rows = [row for row in rows if row["exists"] != "yes"]
    with missing_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CHECKLIST_FIELDS))
        writer.writeheader()
        for row in missing_rows:
            writer.writerow(row)

    missing_family_counts: dict[str, int] = {}
    missing_singer_counts: dict[str, int] = {}
    for row in missing_rows:
        family = row["intended_family"]
        singer_id = row["singer_id"]
        if family:
            missing_family_counts[family] = missing_family_counts.get(family, 0) + 1
        if singer_id:
            missing_singer_counts[singer_id] = missing_singer_counts.get(singer_id, 0) + 1
    invalid_audio_reasons = _invalid_audio_reasons(
        [row["expected_audio_path"] for row in rows if row["exists"] == "yes"],
        enabled=validate_wav_files,
        min_seconds=min_wav_seconds,
        max_seconds=max_wav_seconds,
    )
    invalid_audio_paths = sorted(invalid_audio_reasons)
    valid_audio_files = existing_audio - len(invalid_audio_paths)
    return {
        "root": str(root_path),
        "checklist": str(checklist),
        "missing_csv": str(missing_csv),
        "planned_records": len(rows),
        "planned_groups": len(groups),
        "created_directories": len(directories),
        "existing_audio_files": existing_audio,
        "valid_audio_files": valid_audio_files,
        "invalid_audio_files": len(invalid_audio_paths),
        "missing_audio_files": missing_audio,
        "existing_audio_paths": [row["expected_audio_path"] for row in rows if row["exists"] == "yes"],
        "invalid_audio_paths": invalid_audio_paths,
        "invalid_audio_reasons": invalid_audio_reasons,
        "missing_audio_paths": [row["expected_audio_path"] for row in missing_rows],
        "missing_by_family": dict(sorted(missing_family_counts.items())),
        "missing_by_singer": dict(sorted(missing_singer_counts.items())),
        "intended_family_counts": dict(sorted(family_counts.items())),
        "duplicate_suggested_filenames": duplicate_suggestions,
        "ok": bool(rows) and not duplicate_suggestions,
        "wav_validation_enabled": validate_wav_files,
        "wav_duration_bounds_seconds": {"min": min_wav_seconds, "max": max_wav_seconds},
        "ready_for_review_csv": bool(rows)
        and not duplicate_suggestions
        and missing_audio == 0
        and not invalid_audio_paths,
    }


def _duplicate_suggestions(plan_rows: list[dict[str, str]]) -> list[str]:
    counts: dict[str, int] = {}
    for row in plan_rows:
        value = (row.get("suggested_filename") or "").strip()
        if not value:
            continue
        key = Path(value).as_posix().lstrip("./")
        counts[key] = counts.get(key, 0) + 1
    return sorted(value for value, count in counts.items() if count > 1)


def _invalid_audio_reasons(
    paths: list[str],
    *,
    enabled: bool,
    min_seconds: float,
    max_seconds: float,
) -> dict[str, str]:
    if not enabled:
        return {}
    invalid: dict[str, str] = {}
    for path in paths:
        try:
            with wave.open(path, "rb") as audio:
                channels = audio.getnchannels()
                frame_rate = audio.getframerate()
                frames = audio.getnframes()
                if channels <= 0 or frame_rate <= 0 or frames <= 0:
                    invalid[path] = "empty_or_invalid_wav"
                    continue
                duration = frames / frame_rate
                if duration < min_seconds:
                    invalid[path] = f"too_short:{duration:.3f}s"
                elif max_seconds > 0 and duration > max_seconds:
                    invalid[path] = f"too_long:{duration:.3f}s"
        except (EOFError, wave.Error, OSError):
            invalid[path] = "unreadable_wav"
    return invalid


def main() -> None:
    args = parse_args()
    report = materialize_collection(
        read_plan(args.plan),
        root=args.root,
        checklist_path=args.checklist,
        missing_csv_path=args.missing_csv,
        validate_wav_files=args.validate_wav_files,
        min_wav_seconds=args.min_wav_seconds,
        max_wav_seconds=args.max_wav_seconds,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.report_json:
        output_path = Path(args.report_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    if args.strict and not report["ok"]:
        raise SystemExit(1)
    if args.require_audio_files and not report["ready_for_review_csv"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
