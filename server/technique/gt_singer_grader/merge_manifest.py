"""Merge normalized technique manifests with duplicate checks."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .manifest import normalize_label_list, read_jsonl, validate_record, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge normalized technique manifests")
    parser.add_argument("--input", action="append", required=True, help="Input JSONL manifest. May be repeated.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Allow duplicate recording_id values instead of failing.",
    )
    return parser.parse_args()


def merge_manifest_records(
    manifest_records: list[tuple[str, list[dict[str, Any]]]],
    *,
    allow_duplicates: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_ids: dict[str, str] = {}
    duplicates: list[dict[str, str]] = []

    for source, records in manifest_records:
        for record in records:
            recording_id = str(record.get("recording_id") or "")
            if recording_id and recording_id in seen_ids:
                duplicates.append(
                    {
                        "recording_id": recording_id,
                        "first_source": seen_ids[recording_id],
                        "duplicate_source": source,
                    }
                )
                if not allow_duplicates:
                    continue
            elif recording_id:
                seen_ids[recording_id] = source
            merged.append(record)

    if duplicates and not allow_duplicates:
        preview = "\n".join(
            f"  - {item['recording_id']} ({item['first_source']} and {item['duplicate_source']})"
            for item in duplicates[:10]
        )
        extra = "" if len(duplicates) <= 10 else f"\n  ... and {len(duplicates) - 10} more"
        raise ValueError(f"duplicate recording_id values found:\n{preview}{extra}")

    summary = {
        "input_manifests": [
            {
                "path": source,
                "records": len(records),
                "family_counts": _family_counts(records),
            }
            for source, records in manifest_records
        ],
        "merged_records": len(merged),
        "duplicate_records": len(duplicates),
        "allow_duplicates": allow_duplicates,
        "family_counts": _family_counts(merged),
    }
    return merged, summary


def _family_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        labels = record.get("labels")
        family = "unlabeled"
        if isinstance(labels, dict):
            families = normalize_label_list(labels.get("families"))
            if families:
                family = families[0]
        counts[family] += 1
    return dict(sorted(counts.items()))


def validate_records(records: list[dict[str, Any]], *, source: str) -> None:
    errors: list[str] = []
    for index, record in enumerate(records, start=1):
        errors.extend(validate_record(record, line_number=index))
    if errors:
        for error in errors:
            print(f"{source}: {error}")
        raise SystemExit(1)


def main() -> None:
    args = parse_args()
    manifest_records = []
    for path in args.input:
        records = read_jsonl(path)
        validate_records(records, source=path)
        manifest_records.append((path, records))

    try:
        merged, summary = merge_manifest_records(
            manifest_records,
            allow_duplicates=args.allow_duplicates,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    write_jsonl(args.output, merged)
    if args.summary_output:
        Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_output, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
