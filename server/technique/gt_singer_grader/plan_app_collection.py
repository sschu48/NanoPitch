"""Plan the next app-recording collection batch from validation targets."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from .app_label_coverage import build_report as build_label_coverage_report
from .audit_app_validation import DEFAULT_TARGET_FAMILIES


PLAN_FIELDS = (
    "plan_id",
    "singer_id",
    "intended_family",
    "suggested_filename",
    "review_goal",
    "minimum_review_strength",
    "notes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan app-recording clips needed for technique validation")
    parser.add_argument(
        "--csv",
        default=None,
        help="Existing app-recording review CSV. If omitted or missing, planning starts from zero coverage.",
    )
    parser.add_argument("--output-json", default=None, help="Optional JSON report path")
    parser.add_argument("--output-csv", default=None, help="Optional collection-plan CSV path")
    parser.add_argument("--min-per-family", type=int, default=20)
    parser.add_argument("--min-negative", type=int, default=20)
    parser.add_argument("--min-groups", type=int, default=3)
    parser.add_argument(
        "--target-family",
        action="append",
        default=[],
        help="Target family to require. Defaults to every non-control technique family.",
    )
    parser.add_argument("--singer-prefix", default="planned_singer")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument(
        "--clips-per-singer",
        type=int,
        default=7,
        help="Maximum planned takes per synthetic singer group before starting the next singer_id.",
    )
    return parser.parse_args()


def empty_coverage_report(
    *,
    target_families: list[str] | tuple[str, ...],
    min_per_family: int,
    min_negative: int,
    min_groups: int,
    csv_path: str | None,
) -> dict[str, Any]:
    return {
        "ready_for_collection_target": False,
        "source_csv": str(csv_path or ""),
        "records": 0,
        "families": {},
        "target_families": {family: 0 for family in target_families},
        "missing_target_families": {family: min_per_family for family in target_families},
        "negative_records": 0,
        "negative_shortfall": min_negative,
        "split_groups": 0,
        "group_shortfall": min_groups,
        "thresholds": {
            "min_per_family": min_per_family,
            "min_negative": min_negative,
            "min_groups": min_groups,
        },
        "warnings": ["review CSV not found"],
    }


def coverage_report_from_csv(
    csv_path: str | None,
    *,
    target_families: list[str] | tuple[str, ...],
    min_per_family: int,
    min_negative: int,
    min_groups: int,
) -> dict[str, Any]:
    if not csv_path or not Path(csv_path).is_file():
        return empty_coverage_report(
            target_families=target_families,
            min_per_family=min_per_family,
            min_negative=min_negative,
            min_groups=min_groups,
            csv_path=csv_path,
        )
    return build_label_coverage_report(
        csv_path,
        target_families=target_families,
        min_per_family=min_per_family,
        min_negative=min_negative,
        min_groups=min_groups,
    )


def build_collection_plan(
    *,
    csv_path: str | None = None,
    target_families: list[str] | tuple[str, ...] = DEFAULT_TARGET_FAMILIES,
    min_per_family: int = 20,
    min_negative: int = 20,
    min_groups: int = 3,
    singer_prefix: str = "planned_singer",
    start_index: int = 1,
    clips_per_singer: int = 7,
) -> dict[str, Any]:
    if clips_per_singer <= 0:
        raise ValueError("clips_per_singer must be greater than zero")
    report = coverage_report_from_csv(
        csv_path,
        target_families=target_families,
        min_per_family=min_per_family,
        min_negative=min_negative,
        min_groups=min_groups,
    )
    missing_target_families = {
        family: int(count)
        for family, count in (report.get("missing_target_families") or {}).items()
        if int(count) > 0
    }
    negative_shortfall = int(report.get("negative_shortfall") or 0)
    planned_rows = _planned_rows(
        missing_target_families=missing_target_families,
        negative_shortfall=negative_shortfall,
        existing_groups=int(report.get("split_groups") or 0),
        min_groups=min_groups,
        singer_prefix=singer_prefix,
        start_index=start_index,
        clips_per_singer=clips_per_singer,
    )
    return {
        "ready_for_collection_target": not planned_rows and report.get("ready_for_collection_target") is True,
        "source_csv": report.get("source_csv") or str(csv_path or ""),
        "thresholds": {
            "min_per_family": min_per_family,
            "min_negative": min_negative,
            "min_groups": min_groups,
            "clips_per_singer": clips_per_singer,
        },
        "current": {
            "records": report.get("records", 0),
            "target_families": report.get("target_families") or {},
            "negative_records": report.get("negative_records", 0),
            "split_groups": report.get("split_groups", 0),
            "warnings": report.get("warnings") or [],
        },
        "needed": {
            "target_families": missing_target_families,
            "negative": negative_shortfall,
            "additional_groups": max(0, min_groups - int(report.get("split_groups") or 0)),
        },
        "planned_records": len(planned_rows),
        "planned_groups": len({row["singer_id"] for row in planned_rows}),
        "plan_rows": planned_rows,
    }


def _planned_rows(
    *,
    missing_target_families: dict[str, int],
    negative_shortfall: int,
    existing_groups: int,
    min_groups: int,
    singer_prefix: str,
    start_index: int,
    clips_per_singer: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    planned_tasks = _round_robin_collection_tasks(missing_target_families, negative_shortfall)
    required_new_groups = max(0, min_groups - existing_groups)
    while len(planned_tasks) < required_new_groups:
        planned_tasks.append(
            (
                "control",
                "collect additional singer group for held-out validation diversity",
                "absent",
                "use a distinct singer_id/split_group",
            )
        )

    for offset, (intended_family, review_goal, strength, notes) in enumerate(planned_tasks):
        plan_number = start_index + offset
        minimum_group_count = math.ceil(len(planned_tasks) / clips_per_singer)
        group_count = max(required_new_groups, minimum_group_count, 1)
        if group_count == minimum_group_count:
            singer_offset = offset // clips_per_singer
        else:
            singer_offset = offset % group_count
        singer_number = start_index + singer_offset
        singer_id = f"{singer_prefix}_{singer_number:03d}"
        rows.append(
            {
                "plan_id": f"app_collection:{plan_number:04d}",
                "singer_id": singer_id,
                "intended_family": intended_family,
                "suggested_filename": f"raw/{singer_id}/{intended_family}_{plan_number:04d}.wav",
                "review_goal": review_goal,
                "minimum_review_strength": strength,
                "notes": notes,
            }
        )

    return rows


def _round_robin_collection_tasks(
    missing_target_families: dict[str, int], negative_shortfall: int
) -> list[tuple[str, str, str, str]]:
    remaining = {family: count for family, count in sorted(missing_target_families.items()) if count > 0}
    if negative_shortfall > 0:
        remaining["control"] = negative_shortfall
    family_order = [family for family in sorted(missing_target_families) if remaining.get(family, 0) > 0]
    if remaining.get("control", 0) > 0:
        family_order.append("control")

    tasks: list[tuple[str, str, str, str]] = []
    while any(count > 0 for count in remaining.values()):
        for family in family_order:
            if remaining.get(family, 0) <= 0:
                continue
            remaining[family] -= 1
            if family == "control":
                tasks.append(
                    (
                        "control",
                        "collect ordinary singing with no target technique",
                        "absent",
                        "5-10 seconds; reviewer should mark all technique columns absent when appropriate",
                    )
                )
            else:
                tasks.append(
                    (
                        family,
                        f"collect clear {family} technique",
                        "present",
                        "5-10 seconds; reviewer should mark the matching technique present or strong",
                    )
                )
    return tasks


def write_plan_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PLAN_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in PLAN_FIELDS})


def main() -> None:
    args = parse_args()
    target_families = args.target_family or list(DEFAULT_TARGET_FAMILIES)
    report = build_collection_plan(
        csv_path=args.csv,
        target_families=target_families,
        min_per_family=args.min_per_family,
        min_negative=args.min_negative,
        min_groups=args.min_groups,
        singer_prefix=args.singer_prefix,
        start_index=args.start_index,
        clips_per_singer=args.clips_per_singer,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    if args.output_csv:
        write_plan_csv(args.output_csv, report["plan_rows"])


if __name__ == "__main__":
    main()
