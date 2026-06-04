"""Deterministically sample normalized technique manifests."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .constants import FAMILY_NAMES
from .manifest import normalize_label_list, read_jsonl, validate_record, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a normalized technique manifest")
    parser.add_argument("--input", required=True, help="Input JSONL manifest")
    parser.add_argument("--output", required=True, help="Sampled output JSONL manifest")
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-per-family", type=int, default=None)
    return parser.parse_args()


def _require_positive_int(value: int | None, *, name: str) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive when provided")


def validate_records(records: list[dict[str, Any]]) -> None:
    errors: list[str] = []
    for index, record in enumerate(records, start=1):
        if isinstance(record.get("labels"), dict):
            errors.extend(validate_record(record, line_number=index))
            continue
        try:
            record_audio_path(record)
            record_family(record)
        except ValueError as exc:
            errors.append(f"line {index}: {exc}")
        split_group = record.get("split_group")
        if not isinstance(split_group, str) or not split_group:
            errors.append(f"line {index}: split_group must be a non-empty string")
    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)


def record_family(record: dict[str, Any]) -> str:
    labels = record.get("labels")
    if isinstance(labels, dict):
        families = normalize_label_list(labels.get("families"))
        if families:
            return families[0]
    if record.get("role") in {"control", "speech"}:
        return "control"
    family = record.get("family")
    if isinstance(family, str) and family:
        return family
    raise ValueError(f"manifest record has no family label: {record.get('recording_id') or record.get('stem')}")


def record_audio_path(record: dict[str, Any]) -> str:
    audio_path = record.get("audio_path") or record.get("wav_path")
    if not isinstance(audio_path, str) or not audio_path:
        raise ValueError(f"manifest record has no audio path: {record.get('recording_id') or record.get('stem')}")
    return audio_path


def summarize_sampled_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    family_counts: dict[str, int] = defaultdict(int)
    dataset_counts: dict[str, int] = defaultdict(int)
    trainability_counts: dict[str, int] = defaultdict(int)
    trainable_families = set(FAMILY_NAMES)

    for record in records:
        family = record_family(record)
        family_counts[family] += 1
        dataset_counts[str(record.get("dataset") or "unknown")] += 1
        if family in trainable_families:
            trainability_counts["trainable"] += 1
        else:
            trainability_counts[f"evaluation_only_family:{family}"] += 1

    return {
        "records": len(records),
        "datasets": dict(sorted(dataset_counts.items())),
        "families": dict(sorted(family_counts.items())),
        "trainability": dict(sorted(trainability_counts.items())),
    }


def sample_records(
    records: list[dict[str, Any]],
    *,
    seed: int = 1337,
    max_records: int | None = None,
    max_per_family: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return a deterministic, family-balanced sample of manifest records."""
    _require_positive_int(max_records, name="max_records")
    _require_positive_int(max_per_family, name="max_per_family")

    rng = random.Random(seed)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_family[record_family(record)].append(record)

    capped_by_family: dict[str, list[dict[str, Any]]] = {}
    for family, family_records in sorted(by_family.items()):
        shuffled = list(family_records)
        rng.shuffle(shuffled)
        if max_per_family is not None:
            shuffled = shuffled[:max_per_family]
        capped_by_family[family] = shuffled

    if max_records is None:
        sampled = [record for family in sorted(capped_by_family) for record in capped_by_family[family]]
    else:
        sampled = []
        cursors = {family: 0 for family in capped_by_family}
        families = sorted(capped_by_family)
        while len(sampled) < max_records:
            added = False
            for family in families:
                cursor = cursors[family]
                family_records = capped_by_family[family]
                if cursor >= len(family_records):
                    continue
                sampled.append(family_records[cursor])
                cursors[family] = cursor + 1
                added = True
                if len(sampled) >= max_records:
                    break
            if not added:
                break

    sampled = sorted(sampled, key=lambda record: str(record.get("recording_id") or record.get("audio_path") or ""))
    summary = {
        "input_records": len(records),
        "sampled_records": len(sampled),
        "seed": seed,
        "max_records": max_records,
        "max_per_family": max_per_family,
        "input_summary": summarize_sampled_records(records),
        "sampled_summary": summarize_sampled_records(sampled),
    }
    return sampled, summary


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    validate_records(records)
    try:
        sampled, summary = sample_records(
            records,
            seed=args.seed,
            max_records=args.max_records,
            max_per_family=args.max_per_family,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    write_jsonl(args.output, sampled)
    if args.summary_output:
        output_path = Path(args.summary_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
