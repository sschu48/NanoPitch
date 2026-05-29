"""Audit app-recording validation manifest coverage."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .constants import FAMILY_NAMES
from .manifest import read_jsonl, record_families, trainability_reason, validate_record
from .run_metadata import file_metadata


DEFAULT_TARGET_FAMILIES = tuple(family for family in FAMILY_NAMES if family != "control")
NEGATIVE_FAMILIES = {"control", "none", "unclear"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit app-recording validation coverage")
    parser.add_argument("--manifest", required=True, help="App validation JSONL manifest")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--min-per-family", type=int, default=20)
    parser.add_argument("--min-negative", type=int, default=20)
    parser.add_argument("--min-groups", type=int, default=3)
    parser.add_argument(
        "--target-family",
        action="append",
        default=[],
        help="Target family to require. Defaults to every non-control technique family.",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when validation coverage is not ready")
    return parser.parse_args()


def first_family(record: dict[str, Any]) -> str:
    families = record_families(record)
    if not families:
        return "unlabeled"
    if len(families) > 1:
        return "multiple"
    return families[0]


def audit_records(
    records: list[dict[str, Any]],
    *,
    target_families: list[str] | tuple[str, ...] = DEFAULT_TARGET_FAMILIES,
    min_per_family: int = 20,
    min_negative: int = 20,
    min_groups: int = 3,
) -> dict[str, Any]:
    family_counts: dict[str, int] = defaultdict(int)
    group_counts: dict[str, int] = defaultdict(int)
    trainability_counts: dict[str, int] = defaultdict(int)
    validation_errors: list[str] = []

    for index, record in enumerate(records, start=1):
        validation_errors.extend(validate_record(record, line_number=index))
        family = first_family(record)
        family_counts[family] += 1
        group_counts[str(record.get("split_group") or "missing")] += 1
        trainability_counts[trainability_reason(record)] += 1

    target_family_counts = {family: family_counts.get(family, 0) for family in target_families}
    missing_target_families = {
        family: min_per_family - count
        for family, count in target_family_counts.items()
        if count < min_per_family
    }
    negative_count = sum(family_counts.get(family, 0) for family in NEGATIVE_FAMILIES)
    negative_shortfall = max(0, min_negative - negative_count)
    group_shortfall = max(0, min_groups - len(group_counts))

    warnings: list[str] = []
    if validation_errors:
        warnings.append("manifest has schema validation errors")
    if missing_target_families:
        warnings.append("target family coverage is below threshold")
    if negative_shortfall:
        warnings.append("negative control/none/unclear coverage is below threshold")
    if group_shortfall:
        warnings.append("split_group diversity is below threshold")

    return {
        "ready_for_mvp_validation": not warnings,
        "records": len(records),
        "families": dict(sorted(family_counts.items())),
        "target_families": target_family_counts,
        "missing_target_families": missing_target_families,
        "negative_families": sorted(NEGATIVE_FAMILIES),
        "negative_records": negative_count,
        "negative_shortfall": negative_shortfall,
        "split_groups": len(group_counts),
        "group_shortfall": group_shortfall,
        "trainability": dict(sorted(trainability_counts.items())),
        "thresholds": {
            "min_per_family": min_per_family,
            "min_negative": min_negative,
            "min_groups": min_groups,
        },
        "validation_errors": validation_errors,
        "warnings": warnings,
    }


def main() -> None:
    args = parse_args()
    target_families = args.target_family or list(DEFAULT_TARGET_FAMILIES)
    records = read_jsonl(args.manifest)
    report = audit_records(
        records,
        target_families=target_families,
        min_per_family=args.min_per_family,
        min_negative=args.min_negative,
        min_groups=args.min_groups,
    )
    report["manifest"] = file_metadata(args.manifest)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    if args.strict and not report["ready_for_mvp_validation"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
