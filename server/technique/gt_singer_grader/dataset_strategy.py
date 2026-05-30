"""Audit dataset registry entries against the technique-model strategy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = (
    "id",
    "name",
    "role",
    "priority",
    "recording_domain",
    "label_fit",
    "main_gap",
    "source_url",
    "acquisition",
)
SUPPORTED_MANIFEST_BUILDERS = {"gtsinger", "vocalset", "app_recordings"}
DIRECT_SUPERVISED_ROLES = {
    "primary_supervised_technique_source",
    "supplemental_supervised_technique_source",
    "target_domain_supervised_source",
}
TARGET_DOMAIN_DATASETS = {"app_recordings"}
RECOMMENDATION_ORDER = {
    "use_for_primary_baseline": 1,
    "use_after_baseline_as_supplemental_training": 2,
    "collect_label_and_use_for_app_validation": 3,
    "use_for_robustness_or_eval_after_license_review_not_supervised_training": 4,
    "hold_for_license_taxonomy_label_review": 5,
    "defer_to_quality_axis_not_technique_detector": 6,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit technique-model dataset registry strategy")
    parser.add_argument("--registry", default="./gt_singer_grader/dataset_registry.json")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if the registry has audit errors")
    return parser.parse_args()


def read_registry(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"expected registry JSON array: {path}")
    entries: list[dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"registry entry {index} must be an object")
        entries.append(item)
    return entries


def entry_status(entry: dict[str, Any]) -> dict[str, Any]:
    dataset_id = str(entry.get("id") or "")
    role = str(entry.get("role") or "")
    acquisition = str(entry.get("acquisition") or "")
    supported_builder = dataset_id in SUPPORTED_MANIFEST_BUILDERS
    direct_supervised = role in DIRECT_SUPERVISED_ROLES
    target_domain = dataset_id in TARGET_DOMAIN_DATASETS
    needs_manual_review = (
        "manual" in acquisition
        or role not in DIRECT_SUPERVISED_ROLES
        or "not a direct" in str(entry.get("main_gap") or "").lower()
        or "no technique labels" in str(entry.get("main_gap") or "").lower()
        or "do not map" in str(entry.get("main_gap") or "").lower()
    )

    if target_domain:
        recommendation = "collect_label_and_use_for_app_validation"
    elif supported_builder and direct_supervised and not needs_manual_review:
        recommendation = "use_for_primary_baseline"
    elif supported_builder and direct_supervised:
        recommendation = "use_after_baseline_as_supplemental_training"
    elif role == "domain_adaptation_and_eval_source":
        recommendation = "use_for_robustness_or_eval_after_license_review_not_supervised_training"
    elif role == "quality_axis_reference":
        recommendation = "defer_to_quality_axis_not_technique_detector"
    else:
        recommendation = "hold_for_license_taxonomy_label_review"

    return {
        "id": dataset_id,
        "name": entry.get("name"),
        "priority": entry.get("priority"),
        "role": role,
        "supported_manifest_builder": supported_builder,
        "direct_supervised_technique_source": direct_supervised,
        "target_domain_source": target_domain,
        "needs_manual_review_before_training": needs_manual_review and not target_domain,
        "recommendation": recommendation,
    }


def audit_registry(entries: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    statuses: list[dict[str, Any]] = []

    for index, entry in enumerate(entries, start=1):
        missing = [field for field in REQUIRED_FIELDS if field not in entry]
        if missing:
            errors.append(f"entry {index} missing required field(s): {', '.join(missing)}")
        dataset_id = str(entry.get("id") or "")
        if not dataset_id:
            errors.append(f"entry {index} has empty id")
        elif dataset_id in seen_ids:
            errors.append(f"duplicate dataset id: {dataset_id}")
        seen_ids.add(dataset_id)
        priority = entry.get("priority")
        if not isinstance(priority, int) or priority < 1:
            errors.append(f"{dataset_id or f'entry {index}'} has invalid priority: {priority}")
        statuses.append(entry_status(entry))

    by_recommendation: dict[str, list[str]] = {}
    for status in statuses:
        by_recommendation.setdefault(str(status["recommendation"]), []).append(str(status["id"]))

    recommended_order = [
        status["id"]
        for status in sorted(
            statuses,
            key=lambda item: (
                RECOMMENDATION_ORDER.get(str(item.get("recommendation")), 999),
                int(item["priority"]) if isinstance(item.get("priority"), int) else 999,
                str(item["id"]),
            ),
        )
    ]
    return {
        "ok": not errors,
        "errors": errors,
        "datasets": statuses,
        "by_recommendation": {key: sorted(value) for key, value in sorted(by_recommendation.items())},
        "recommended_order": recommended_order,
        "strategy_notes": [
            "GT Singer remains the first baseline source.",
            "VocalSet is supplemental after GT Singer baseline metrics exist.",
            "App recordings are required target-domain validation before product packaging.",
            "Mobile/quality datasets without technique labels must not be treated as supervised technique labels.",
        ],
    }


def main() -> None:
    args = parse_args()
    report = audit_registry(read_registry(args.registry))
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    if args.strict and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
