"""Split normalized technique manifests without leaking split groups."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .manifest import normalize_label_list, read_jsonl, validate_record, write_jsonl
from .split_health import split_coverage_errors, split_family_compatibility_errors, validate_val_ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a normalized technique manifest")
    parser.add_argument("--input", required=True, help="Input JSONL manifest")
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--val-output", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--group-field",
        default="split_group",
        help="Top-level field used to prevent train/val leakage.",
    )
    parser.add_argument("--summary-output", default=None)
    parser.add_argument(
        "--strict-non-empty",
        action="store_true",
        help="Exit non-zero if the split produces an empty train or validation output.",
    )
    parser.add_argument(
        "--strict-family-coverage",
        action="store_true",
        help="Exit non-zero if train/validation family coverage is not mutually compatible.",
    )
    return parser.parse_args()


def primary_family(record: dict[str, Any]) -> str:
    labels = record.get("labels")
    if isinstance(labels, dict):
        families = normalize_label_list(labels.get("families"))
        if families:
            return families[0]
    return "unlabeled"


def group_value(record: dict[str, Any], group_field: str) -> str:
    value = record.get(group_field)
    if isinstance(value, str) and value:
        return value
    return str(record.get("recording_id") or record.get("audio_path") or "")


def split_manifest_records(
    records: list[dict[str, Any]],
    *,
    val_ratio: float,
    seed: int,
    group_field: str = "split_group",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    validate_val_ratio(val_ratio)

    family_groups: dict[str, set[str]] = defaultdict(set)
    group_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        group = group_value(record, group_field)
        family_groups[primary_family(record)].add(group)
        group_records[group].append(record)

    rng = random.Random(seed)
    val_groups: set[str] = set()
    for groups in family_groups.values():
        unique_groups = sorted(groups)
        rng.shuffle(unique_groups)
        if len(unique_groups) <= 1 or val_ratio <= 0.0:
            continue
        n_val = max(1, int(round(len(unique_groups) * val_ratio)))
        n_val = min(n_val, len(unique_groups) - 1)
        val_groups.update(unique_groups[:n_val])

    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []
    for group, grouped_records in group_records.items():
        if group in val_groups:
            val_records.extend(grouped_records)
        else:
            train_records.extend(grouped_records)

    summary = {
        "input_records": len(records),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "group_field": group_field,
        "train_groups": len(set(group_value(record, group_field) for record in train_records)),
        "val_groups": len(set(group_value(record, group_field) for record in val_records)),
        "val_ratio": val_ratio,
        "seed": seed,
        "family_counts": {
            "train": _family_counts(train_records),
            "val": _family_counts(val_records),
        },
    }
    summary["warnings"] = split_warnings(summary)
    summary["coverage_errors"] = split_family_coverage_errors(train_records, val_records)
    return train_records, val_records, summary


def split_warnings(summary: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if summary["input_records"] == 0:
        warnings.append("input manifest is empty")
    if summary["train_records"] == 0:
        warnings.append("training split is empty")
    if summary["val_records"] == 0:
        warnings.append("validation split is empty")
    return warnings


def _family_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[primary_family(record)] += 1
    return dict(sorted(counts.items()))


def split_family_coverage_errors(
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    errors.extend(split_coverage_errors(train_records, source="train split", purpose="training"))
    errors.extend(split_coverage_errors(val_records, source="validation split", purpose="validation"))
    errors.extend(split_family_compatibility_errors(train_records, val_records, source="split manifest"))
    return errors


def validate_records(records: list[dict[str, Any]]) -> None:
    errors: list[str] = []
    for index, record in enumerate(records, start=1):
        errors.extend(validate_record(record, line_number=index))
    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    validate_records(records)
    train_records, val_records, summary = split_manifest_records(
        records,
        val_ratio=args.val_ratio,
        seed=args.seed,
        group_field=args.group_field,
    )
    if args.strict_non_empty and summary["warnings"]:
        for warning in summary["warnings"]:
            print(f"warning: {warning}")
        raise SystemExit(1)
    if args.strict_family_coverage and summary["coverage_errors"]:
        for error in summary["coverage_errors"]:
            print(f"coverage_error: {error}")
        raise SystemExit(1)
    write_jsonl(args.train_output, train_records)
    write_jsonl(args.val_output, val_records)
    if args.summary_output:
        Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_output, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
