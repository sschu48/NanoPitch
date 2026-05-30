"""Shared manifest schema for technique-model training data."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .constants import FAMILY_NAMES, TECHNIQUE_KEYS


KNOWN_FAMILIES = set(FAMILY_NAMES) | {"none", "unclear", "multiple"}
KNOWN_TECHNIQUES = set(TECHNIQUE_KEYS)
TRAINABLE_FAMILIES = set(FAMILY_NAMES)
REQUIRED_FIELDS = (
    "recording_id",
    "dataset",
    "audio_path",
    "recording_domain",
    "label_source",
    "labels",
    "split_group",
)


def normalize_label_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        parts = value.replace(";", ",").split(",")
        return [part.strip() for part in parts if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    raise TypeError(f"expected label list or comma-separated string, got {type(value).__name__}")


def record_families(record: dict[str, Any]) -> list[str]:
    labels = record.get("labels")
    if isinstance(labels, dict):
        families = normalize_label_list(labels.get("families"))
        if families:
            return families
    return []


def trainability_reason(record: dict[str, Any]) -> str:
    families = record_families(record)
    if not families:
        return "missing_family"
    if len(families) > 1:
        return "multiple_families"
    evaluation_only = sorted(family for family in families if family not in TRAINABLE_FAMILIES)
    if evaluation_only:
        return "evaluation_only_family:" + ",".join(evaluation_only)
    return "trainable"


def is_trainable_record(record: dict[str, Any]) -> bool:
    return trainability_reason(record) == "trainable"


def require_non_empty_records(records: list[Any], *, source: str, purpose: str) -> None:
    if not records:
        raise ValueError(f"{source} has no records for {purpose}")


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    family_counts: dict[str, int] = defaultdict(int)
    trainability_counts: dict[str, int] = defaultdict(int)
    dataset_counts: dict[str, int] = defaultdict(int)

    for record in records:
        dataset_counts[str(record.get("dataset") or "unknown")] += 1
        families = record_families(record)
        family = families[0] if families else "unlabeled"
        family_counts[family] += 1
        trainability_counts[trainability_reason(record)] += 1

    return {
        "records": len(records),
        "datasets": dict(sorted(dataset_counts.items())),
        "families": dict(sorted(family_counts.items())),
        "trainability": dict(sorted(trainability_counts.items())),
    }


def validate_record(record: dict[str, Any], *, line_number: int | None = None) -> list[str]:
    prefix = f"line {line_number}: " if line_number is not None else ""
    errors: list[str] = []

    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(f"{prefix}missing required field: {field}")

    labels = record.get("labels")
    if not isinstance(labels, dict):
        errors.append(f"{prefix}labels must be an object")
        return errors

    families = normalize_label_list(labels.get("families"))
    techniques = normalize_label_list(labels.get("techniques"))

    unknown_families = sorted(set(families) - KNOWN_FAMILIES)
    unknown_techniques = sorted(set(techniques) - KNOWN_TECHNIQUES)
    if unknown_families:
        errors.append(f"{prefix}unknown family label(s): {', '.join(unknown_families)}")
    if unknown_techniques:
        errors.append(f"{prefix}unknown technique label(s): {', '.join(unknown_techniques)}")
    if not families and not techniques:
        errors.append(f"{prefix}record has no family or technique labels")

    audio_path = record.get("audio_path")
    if not isinstance(audio_path, str) or not audio_path:
        errors.append(f"{prefix}audio_path must be a non-empty string")

    split_group = record.get("split_group")
    if not isinstance(split_group, str) or not split_group:
        errors.append(f"{prefix}split_group must be a non-empty string")

    return errors


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number}: expected JSON object")
            records.append(record)
    return records


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def validate_file(path: str | Path) -> list[str]:
    errors: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON: {exc}")
                continue
            if not isinstance(record, dict):
                errors.append(f"line {line_number}: expected JSON object")
                continue
            errors.extend(validate_record(record, line_number=line_number))
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a technique-model JSONL manifest")
    parser.add_argument("manifest")
    parser.add_argument(
        "--require-trainable",
        action="store_true",
        help="Exit non-zero if any record cannot be used by the current supervised trainer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors = validate_file(args.manifest)
    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)
    print(f"Manifest OK: {args.manifest}")
    records = read_jsonl(args.manifest)
    summary = summarize_records(records)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_trainable:
        non_trainable = {
            reason: count
            for reason, count in summary["trainability"].items()
            if reason != "trainable"
        }
        if non_trainable:
            print(
                "Manifest contains records that are not trainable by the current supervised trainer: "
                + json.dumps(non_trainable, sort_keys=True)
            )
            raise SystemExit(1)


if __name__ == "__main__":
    main()
