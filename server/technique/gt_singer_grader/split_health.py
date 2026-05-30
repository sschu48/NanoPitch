"""Readiness checks for train/validation split coverage."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .manifest import normalize_label_list

NON_TECHNIQUE_FAMILIES = {"control", "multiple", "none", "unclear", "unlabeled"}


def validate_val_ratio(val_ratio: float) -> None:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be >= 0 and < 1")


def record_family(record: Any) -> str | None:
    if isinstance(record, dict):
        if record.get("role") in {"control", "speech"}:
            return "control"
        family = record.get("family")
        if isinstance(family, str) and family:
            return family
        labels = record.get("labels")
        if isinstance(labels, dict):
            families = normalize_label_list(labels.get("families"))
            if len(families) == 1:
                return families[0]
            if len(families) > 1:
                return "multiple"
        return None

    role = getattr(record, "role", None)
    if role in {"control", "speech"}:
        return "control"
    family = getattr(record, "family", None)
    return family if isinstance(family, str) and family else None


def family_counts(records: list[Any]) -> dict[str, int]:
    counts = Counter()
    for record in records:
        family = record_family(record)
        counts[family or "unlabeled"] += 1
    return dict(sorted(counts.items()))


def trainable_technique_families(records: list[Any]) -> set[str]:
    counts = family_counts(records)
    return {family for family, count in counts.items() if count > 0 and family not in NON_TECHNIQUE_FAMILIES}


def split_coverage_errors(
    records: list[Any],
    *,
    source: str,
    purpose: str,
    min_families: int = 2,
    require_non_control: bool = True,
) -> list[str]:
    counts = family_counts(records)
    labeled_families = sorted(family for family, count in counts.items() if family != "unlabeled" and count > 0)
    errors: list[str] = []
    if len(labeled_families) < min_families:
        errors.append(
            f"{source} has only {len(labeled_families)} labeled clip family/families for {purpose}; "
            f"need at least {min_families}. counts={counts}"
        )
    if require_non_control and not any(family not in {"control", "unlabeled"} for family in labeled_families):
        errors.append(f"{source} has no non-control technique family for {purpose}; counts={counts}")
    return errors


def split_family_compatibility_errors(
    train_records: list[Any],
    val_records: list[Any],
    *,
    source: str,
) -> list[str]:
    train_families = trainable_technique_families(train_records)
    val_families = trainable_technique_families(val_records)
    errors: list[str] = []

    missing_from_train = sorted(val_families - train_families)
    if missing_from_train:
        errors.append(
            f"{source} validation split contains trainable technique family/families with no training examples: "
            f"{', '.join(missing_from_train)}"
        )

    missing_from_val = sorted(train_families - val_families)
    if missing_from_val:
        errors.append(
            f"{source} training split contains trainable technique family/families with no validation examples: "
            f"{', '.join(missing_from_val)}"
        )

    return errors


def require_split_coverage(
    records: list[Any],
    *,
    source: str,
    purpose: str,
    min_families: int = 2,
    require_non_control: bool = True,
) -> None:
    errors = split_coverage_errors(
        records,
        source=source,
        purpose=purpose,
        min_families=min_families,
        require_non_control=require_non_control,
    )
    if errors:
        raise ValueError("\n".join(errors))


def require_split_family_compatibility(
    train_records: list[Any],
    val_records: list[Any],
    *,
    source: str,
) -> None:
    errors = split_family_compatibility_errors(train_records, val_records, source=source)
    if errors:
        raise ValueError("\n".join(errors))
