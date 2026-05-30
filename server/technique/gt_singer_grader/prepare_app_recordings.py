"""Prepare an app-recording review CSV from collected WAV files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .constants import TECHNIQUE_KEYS


REVIEW_FIELDS = (
    "audio_path",
    "recording_id",
    "singer_id",
    "song_id",
    "intended_family",
    *TECHNIQUE_KEYS,
    "families",
    "techniques",
    "split_group",
    "label_source",
    "reviewer_id",
    "notes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a review-label CSV starter from app WAV recordings")
    parser.add_argument("--audio-dir", required=True, help="Directory containing collected app WAV files")
    parser.add_argument("--output", required=True, help="CSV path to write")
    parser.add_argument("--report-json", default=None, help="Optional JSON report path")
    parser.add_argument(
        "--collection-plan",
        default=None,
        help="Optional CSV from plan_app_collection used to prefill planned singer_id, intended_family, and notes.",
    )
    parser.add_argument(
        "--strict-collection-plan",
        action="store_true",
        help="Fail if collected WAVs do not exactly match the collection plan.",
    )
    parser.add_argument(
        "--relative-to",
        default=None,
        help="Write audio_path values relative to this directory. Defaults to the current working directory.",
    )
    parser.add_argument("--recording-prefix", default="app", help="Prefix for generated recording_id values")
    parser.add_argument("--label-source", default="coach_review")
    parser.add_argument("--reviewer-id", default="")
    parser.add_argument("--song-id", default="")
    parser.add_argument("--intended-family", default="")
    parser.add_argument("--allow-empty", action="store_true", help="Write a header-only CSV when no WAV files are found")
    parser.add_argument(
        "--singer-id-from-parent",
        action="store_true",
        help="Use each WAV file's parent directory name as singer_id and split_group.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output CSV")
    return parser.parse_args()


def discover_wavs(audio_dir: str | Path) -> list[Path]:
    root = Path(audio_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"audio directory not found: {root}")
    return sorted(path for path in root.rglob("*.wav") if path.is_file())


def _path_for_csv(path: Path, relative_to: str | Path | None) -> str:
    base = Path(relative_to) if relative_to else Path.cwd()
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path)


def _normalize_manifest_path(path: str) -> str:
    return Path(path.strip()).as_posix().lstrip("./")


def _collection_plan_match_key(path: str) -> str:
    return _normalize_manifest_path(path).lower()


def read_collection_plan(path: str | Path | None) -> list[dict[str, str]]:
    if not path:
        return []
    plan_path = Path(path)
    if not plan_path.is_file():
        raise FileNotFoundError(f"collection plan not found: {plan_path}")
    with plan_path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _match_collection_plan_row(audio_path: str, plan_rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not plan_rows:
        return None
    audio_key = _collection_plan_match_key(audio_path)
    for row in plan_rows:
        suggested = _collection_plan_match_key(row.get("suggested_filename") or "")
        if suggested and (audio_key == suggested or audio_key.endswith(f"/{suggested}")):
            return row
    return None


def collection_plan_match_report(
    wav_paths: list[Path],
    *,
    relative_to: str | Path | None,
    collection_plan_rows: list[dict[str, str]],
) -> dict[str, Any]:
    audio_paths = [_path_for_csv(path, relative_to) for path in wav_paths]
    matched_plan_keys: set[str] = set()
    matched_audio_paths: list[str] = []
    unplanned_audio_paths: list[str] = []

    for audio_path in audio_paths:
        plan_row = _match_collection_plan_row(audio_path, collection_plan_rows)
        if plan_row is None:
            unplanned_audio_paths.append(audio_path)
            continue
        suggested = _collection_plan_match_key(plan_row.get("suggested_filename") or "")
        if suggested:
            matched_plan_keys.add(suggested)
        matched_audio_paths.append(audio_path)

    missing_collection_plan_suggestions = [
        row.get("suggested_filename", "").strip()
        for row in collection_plan_rows
        if row.get("suggested_filename", "").strip()
        and _collection_plan_match_key(row.get("suggested_filename") or "") not in matched_plan_keys
    ]
    return {
        "collection_plan_rows": len(collection_plan_rows),
        "collection_plan_matches": len(matched_audio_paths),
        "collection_plan_fully_matched": bool(collection_plan_rows)
        and not unplanned_audio_paths
        and not missing_collection_plan_suggestions,
        "matched_audio_paths": matched_audio_paths,
        "unplanned_audio_paths": unplanned_audio_paths,
        "missing_collection_plan_suggestions": missing_collection_plan_suggestions,
    }


def _collection_plan_notes(row: dict[str, str]) -> str:
    parts = [
        value.strip()
        for value in (
            row.get("plan_id") or "",
            row.get("review_goal") or "",
            row.get("minimum_review_strength") or "",
            row.get("notes") or "",
        )
        if value and value.strip()
    ]
    return " | ".join(parts)


def _recording_id_from_audio_path(audio_path: str, *, recording_prefix: str) -> str:
    stem_path = Path(_normalize_manifest_path(audio_path)).with_suffix("").as_posix()
    safe = "".join(character if character.isalnum() or character in ("-", "_") else "_" for character in stem_path)
    safe = "_".join(part for part in safe.split("_") if part)
    return f"{recording_prefix}:{safe or 'recording'}"


def build_review_rows(
    wav_paths: list[Path],
    *,
    relative_to: str | Path | None,
    recording_prefix: str,
    label_source: str,
    reviewer_id: str,
    song_id: str,
    intended_family: str,
    singer_id_from_parent: bool,
    collection_plan_rows: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for wav_path in wav_paths:
        audio_path = _path_for_csv(wav_path, relative_to)
        plan_row = _match_collection_plan_row(audio_path, collection_plan_rows or [])
        plan_singer_id = (plan_row or {}).get("singer_id", "").strip()
        singer_id = plan_singer_id or (wav_path.parent.name if singer_id_from_parent else "")
        row: dict[str, str] = {
            "audio_path": audio_path,
            "recording_id": _recording_id_from_audio_path(audio_path, recording_prefix=recording_prefix),
            "singer_id": singer_id,
            "song_id": song_id,
            "intended_family": (plan_row or {}).get("intended_family", "").strip() or intended_family,
            "families": "",
            "techniques": "",
            "split_group": singer_id,
            "label_source": label_source,
            "reviewer_id": reviewer_id,
            "notes": _collection_plan_notes(plan_row or {}),
        }
        for technique in TECHNIQUE_KEYS:
            row[technique] = ""
        rows.append(row)
    return rows


def write_review_csv(output: str | Path, rows: list[dict[str, Any]], *, force: bool = False) -> None:
    output_path = Path(output)
    if output_path.exists() and not force:
        raise FileExistsError(f"output already exists: {output_path}; pass --force to overwrite")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REVIEW_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in REVIEW_FIELDS})


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    wavs = discover_wavs(args.audio_dir)
    if not wavs and not getattr(args, "allow_empty", False):
        raise ValueError(f"no WAV files found under {args.audio_dir}; pass --allow-empty to write only headers")
    collection_plan_rows = read_collection_plan(getattr(args, "collection_plan", None))
    rows = build_review_rows(
        wavs,
        relative_to=args.relative_to,
        recording_prefix=args.recording_prefix,
        label_source=args.label_source,
        reviewer_id=args.reviewer_id,
        song_id=args.song_id,
        intended_family=args.intended_family,
        singer_id_from_parent=args.singer_id_from_parent,
        collection_plan_rows=collection_plan_rows,
    )
    plan_match_report = collection_plan_match_report(
        wavs,
        relative_to=args.relative_to,
        collection_plan_rows=collection_plan_rows,
    )
    if collection_plan_rows and getattr(args, "strict_collection_plan", False):
        missing = plan_match_report["missing_collection_plan_suggestions"]
        unplanned = plan_match_report["unplanned_audio_paths"]
        if missing or unplanned:
            raise SystemExit(
                "collection plan mismatch: "
                f"missing planned={len(missing)}, unplanned wavs={len(unplanned)}"
            )
    write_review_csv(args.output, rows, force=args.force)
    return {
        "audio_dir": str(Path(args.audio_dir)),
        "collection_plan": str(Path(args.collection_plan)) if getattr(args, "collection_plan", None) else "",
        "output": str(Path(args.output)),
        "records": len(rows),
        "review_fields": list(REVIEW_FIELDS),
        **plan_match_report,
    }


def main() -> None:
    args = parse_args()
    report = build_report(args)
    if args.report_json:
        output_path = Path(args.report_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {report['records']} review row(s) to {report['output']}")
    if report["collection_plan"]:
        print(
            "Collection plan matches: "
            f"{report['collection_plan_matches']}/{report['collection_plan_rows']} "
            f"(missing planned={len(report['missing_collection_plan_suggestions'])}, "
            f"unplanned wavs={len(report['unplanned_audio_paths'])})"
        )


if __name__ == "__main__":
    main()
