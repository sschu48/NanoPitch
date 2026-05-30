"""Export singer-facing app-recording collection sheets from a checklist."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .run_metadata import file_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export app-recording collection sheets from collection_checklist.csv")
    parser.add_argument("--checklist", required=True, help="Checklist CSV from materialize_app_collection")
    parser.add_argument("--output-dir", required=True, help="Directory for generated per-singer sheets")
    parser.add_argument("--summary-json", default=None, help="Optional JSON summary path")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if the checklist is empty or has duplicate expected audio paths.",
    )
    return parser.parse_args()


def read_checklist(path: str | Path) -> list[dict[str, str]]:
    checklist = Path(path)
    if not checklist.is_file():
        raise FileNotFoundError(f"collection checklist not found: {checklist}")
    with checklist.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def export_packet(
    rows: list[dict[str, str]],
    *,
    output_dir: str | Path,
    source_checklist: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    removed_sheet_paths = _remove_previous_sheets(output)

    by_singer: dict[str, list[dict[str, str]]] = {}
    family_counts: dict[str, int] = {}
    expected_paths: dict[str, int] = {}
    for row in rows:
        singer_id = (row.get("singer_id") or "unknown_singer").strip() or "unknown_singer"
        by_singer.setdefault(singer_id, []).append(row)
        family = (row.get("intended_family") or "unknown").strip() or "unknown"
        family_counts[family] = family_counts.get(family, 0) + 1
        expected = (row.get("expected_audio_path") or "").strip()
        if expected:
            expected_paths[expected] = expected_paths.get(expected, 0) + 1

    sheet_paths: list[str] = []
    for singer_id in sorted(by_singer):
        sheet_path = output / f"{_safe_filename(singer_id)}.md"
        sheet_path.write_text(_sheet_text(singer_id, by_singer[singer_id]), encoding="utf-8")
        sheet_paths.append(str(sheet_path))

    duplicate_audio_paths = sorted(path for path, count in expected_paths.items() if count > 1)
    existing_audio_files = sum(1 for row in rows if (row.get("exists") or "").strip().lower() == "yes")
    missing_audio_files = len(rows) - existing_audio_files
    report = {
        "output_dir": str(output),
        "planned_records": len(rows),
        "planned_singers": len(by_singer),
        "existing_audio_files": existing_audio_files,
        "missing_audio_files": missing_audio_files,
        "intended_family_counts": dict(sorted(family_counts.items())),
        "duplicate_audio_paths": duplicate_audio_paths,
        "index_path": str(output / "index.md"),
        "source_checklist": file_metadata(source_checklist) if source_checklist else None,
        "sheets": sheet_paths,
        "removed_sheet_paths": removed_sheet_paths,
        "ok": bool(rows) and not duplicate_audio_paths,
    }
    (output / "index.md").write_text(_index_text(report, by_singer), encoding="utf-8")
    return report


def _remove_previous_sheets(output_dir: Path) -> list[str]:
    removed: list[str] = []
    for path in sorted(output_dir.glob("*.md")):
        if path.name == "index.md":
            continue
        path.unlink()
        removed.append(str(path))
    return removed


def _index_text(report: dict[str, Any], by_singer: dict[str, list[dict[str, str]]]) -> str:
    lines = [
        "# App Recording Collection Packet",
        "",
        f"- Planned clips: {report['planned_records']}",
        f"- Planned singer groups: {report['planned_singers']}",
        f"- Existing audio files: {report['existing_audio_files']}",
        f"- Missing audio files: {report['missing_audio_files']}",
        "",
        "## Family Targets",
        "",
        "| Intended family | Planned clips |",
        "| --- | ---: |",
    ]
    family_counts = report.get("intended_family_counts") or {}
    for family, count in family_counts.items():
        lines.append(f"| {_cell(str(family))} | {count} |")
    lines.extend(
        [
            "",
            "## Singer Sheets",
            "",
            "| Singer group | Planned clips | Sheet |",
            "| --- | ---: | --- |",
        ]
    )
    for singer_id in sorted(by_singer):
        sheet = f"{_safe_filename(singer_id)}.md"
        lines.append(f"| {_cell(singer_id)} | {len(by_singer[singer_id])} | [{sheet}]({sheet}) |")
    lines.append("")
    return "\n".join(lines)


def _sheet_text(singer_id: str, rows: list[dict[str, str]]) -> str:
    lines = [
        f"# App Recording Collection: {singer_id}",
        "",
        "Record each take as a 5-10 second WAV at the exact expected path.",
        "Reviewer labels are assigned later; the intended family is only the prompt target.",
        "",
        "| Take | Intended family | Expected file | Prompt |",
        "| --- | --- | --- | --- |",
    ]
    for index, row in enumerate(rows, start=1):
        family = _cell(row.get("intended_family") or "")
        expected = _cell(row.get("expected_audio_path") or "")
        goal = _cell(row.get("review_goal") or row.get("notes") or "")
        lines.append(f"| {index} | {family} | `{expected}` | {goal} |")
    lines.append("")
    return "\n".join(lines)


def _cell(value: str) -> str:
    return value.replace("|", "/").strip()


def _safe_filename(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in ("-", "_") else "_" for character in value)
    return safe or "unknown_singer"


def main() -> None:
    args = parse_args()
    report = export_packet(
        read_checklist(args.checklist),
        output_dir=args.output_dir,
        source_checklist=args.checklist,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.summary_json:
        summary = Path(args.summary_json)
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(text + "\n", encoding="utf-8")
    if args.strict and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
