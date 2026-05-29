"""Plan a technique training run without importing PyTorch."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .constants import FAMILY_NAMES, GROUP_NAME_TO_FAMILY, TECHNIQUE_FOLDER_TO_FAMILY
from .manifest import read_jsonl, require_non_empty_records, summarize_records, trainability_reason
from .split_health import (
    family_counts,
    require_split_coverage,
    split_coverage_errors,
    split_family_compatibility_errors,
    trainable_technique_families,
    validate_val_ratio,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan a technique training run before launching PyTorch")
    parser.add_argument("--dataset-root", default=None, help="Path to the downloaded GT Singer English tree")
    parser.add_argument("--train-manifest", default=None, help="Explicit training JSONL manifest")
    parser.add_argument("--val-manifest", default=None, help="Explicit validation JSONL manifest")
    parser.add_argument(
        "--extra-train-manifest",
        action="append",
        default=[],
        help="Additional weak/supplemental training manifest. May be passed more than once.",
    )
    parser.add_argument(
        "--require-train-dataset",
        action="append",
        default=[],
        help="Require at least one training record from this dataset. May be passed more than once.",
    )
    parser.add_argument(
        "--require-val-dataset",
        action="append",
        default=[],
        help="Require at least one validation record from this dataset. May be passed more than once.",
    )
    parser.add_argument("--language", default="English")
    parser.add_argument("--split-group", choices=("song", "speaker"), default="song")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--include-speech", action="store_true")
    parser.add_argument("--output-json", default=None, help="Optional path to write the plan report")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when the plan is not trainable")
    return parser.parse_args()


def _resolve_family(technique_dir_name: str, group_name: str) -> str | None:
    if group_name in {"Control_Group", "Paired_Speech_Group"}:
        return "control"
    if group_name in GROUP_NAME_TO_FAMILY:
        return GROUP_NAME_TO_FAMILY[group_name]
    return TECHNIQUE_FOLDER_TO_FAMILY.get(technique_dir_name)


def scan_gt_singer_light(root: str, language: str = "English", include_speech: bool = False) -> list[dict[str, Any]]:
    """Scan GT Singer paths and labels without importing the training dataset."""
    root_path = Path(root)
    language_root = root_path / language if (root_path / language).exists() else root_path
    if not language_root.exists():
        raise FileNotFoundError(f"dataset root not found: {language_root}")

    records: list[dict[str, Any]] = []
    for speaker_dir in sorted(path for path in language_root.iterdir() if path.is_dir()):
        for technique_dir in sorted(path for path in speaker_dir.iterdir() if path.is_dir()):
            if technique_dir.name not in TECHNIQUE_FOLDER_TO_FAMILY:
                continue
            for song_dir in sorted(path for path in technique_dir.iterdir() if path.is_dir()):
                for group_dir in sorted(path for path in song_dir.iterdir() if path.is_dir()):
                    group_name = group_dir.name
                    if group_name == "Paired_Speech_Group" and not include_speech:
                        continue
                    family = _resolve_family(technique_dir.name, group_name)
                    if family is None:
                        continue
                    if group_name == "Control_Group":
                        role = "control"
                    elif group_name == "Paired_Speech_Group":
                        role = "speech"
                    else:
                        role = "emphasis"
                    for wav_path in sorted(group_dir.glob("*.wav")):
                        json_path = wav_path.with_suffix(".json")
                        if not json_path.exists():
                            continue
                        records.append(
                            {
                                "speaker": speaker_dir.name,
                                "technique_folder": technique_dir.name,
                                "family": family,
                                "role": role,
                                "song": song_dir.name,
                                "stem": wav_path.stem,
                                "wav_path": str(wav_path),
                                "json_path": str(json_path),
                                "split_group": f"{speaker_dir.name}|{technique_dir.name}|{song_dir.name}",
                            }
                        )
    if not records:
        raise RuntimeError(f"no GT Singer records found under: {language_root}")
    return records


def split_gt_singer_records(
    records: Iterable[dict[str, Any]],
    *,
    val_ratio: float,
    seed: int,
    group_by: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if group_by not in {"song", "speaker"}:
        raise ValueError(f"unknown split group: {group_by}")
    validate_val_ratio(val_ratio)

    record_list = list(records)
    unique_groups = sorted(
        {str(record["speaker"] if group_by == "speaker" else record["split_group"]) for record in record_list}
    )
    rng = random.Random(seed)
    rng.shuffle(unique_groups)
    if len(unique_groups) <= 1 or val_ratio <= 0.0:
        return record_list, []

    n_val = max(1, int(round(len(unique_groups) * val_ratio)))
    n_val = min(n_val, len(unique_groups) - 1)

    val_groups: set[str] = set()
    target_families = trainable_technique_families(record_list)

    def key(record: dict[str, Any]) -> str:
        return str(record["speaker"] if group_by == "speaker" else record["split_group"])

    def can_add(candidate: str) -> bool:
        next_val_groups = val_groups | {candidate}
        next_train = [record for record in record_list if key(record) not in next_val_groups]
        next_val = [record for record in record_list if key(record) in next_val_groups]
        if split_coverage_errors(next_train, source="train split", purpose="training"):
            return False
        if split_coverage_errors(next_val, source="validation split", purpose="validation"):
            return False
        return target_families <= trainable_technique_families(next_train)

    def add_group(candidate: str) -> None:
        val_groups.add(candidate)

    for family in sorted(target_families):
        if family in trainable_technique_families([record for record in record_list if key(record) in val_groups]):
            continue
        for candidate in unique_groups:
            if candidate in val_groups:
                continue
            candidate_records = [record for record in record_list if key(record) == candidate]
            if family not in trainable_technique_families(candidate_records):
                continue
            if can_add(candidate):
                add_group(candidate)
                break

    target_val_groups = max(n_val, len(val_groups))
    for candidate in unique_groups:
        if len(val_groups) >= target_val_groups:
            break
        if candidate in val_groups:
            continue
        if can_add(candidate):
            add_group(candidate)

    return [record for record in record_list if key(record) not in val_groups], [
        record for record in record_list if key(record) in val_groups
    ]


def summarize_gt_singer_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(f"{record['family']}:{record['role']}" for record in records)
    speakers = {str(record["speaker"]) for record in records}
    split_groups = {str(record["split_group"]) for record in records}
    return {
        "records": len(records),
        "families": dict(sorted(family_counts(records).items())),
        "family_roles": dict(sorted(counts.items())),
        "speakers": len(speakers),
        "split_groups": len(split_groups),
    }


def _require_trainable_manifest(records: list[dict[str, Any]], *, source: str) -> list[str]:
    errors: list[str] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record.get("labels"), dict):
            family = record.get("family")
            if record.get("role") in {"control", "speech"} or family in FAMILY_NAMES:
                continue
            errors.append(f"line {index}: manifest record has no trainable family label")
            continue
        reason = trainability_reason(record)
        if reason != "trainable":
            record_id = record.get("recording_id") or record.get("stem") or f"line {index}"
            errors.append(f"{record_id}: {reason}")
    return [f"{source} contains records that cannot be used for supervised training"] + errors if errors else []


def _coverage_errors(records: list[Any], *, source: str, purpose: str) -> list[str]:
    try:
        require_non_empty_records(records, source=source, purpose=purpose)
        require_split_coverage(records, source=source, purpose=purpose)
    except ValueError as exc:
        return str(exc).splitlines()
    return []


def _manifest_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if records and isinstance(records[0].get("labels"), dict):
        return summarize_records(records)
    return {
        "records": len(records),
        "families": dict(sorted(family_counts(records).items())),
        "trainability": {"trainable": len(records)},
    }


def _list_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _record_dataset(record: dict[str, Any]) -> str:
    dataset = record.get("dataset")
    if dataset:
        return str(dataset)
    if record.get("wav_path") and record.get("json_path"):
        return "gtsinger"
    return ""


def _dataset_presence_errors(
    records: list[dict[str, Any]],
    required_datasets: Iterable[str],
    *,
    source: str,
    purpose: str,
) -> list[str]:
    present = {dataset for record in records if (dataset := _record_dataset(record))}
    errors: list[str] = []
    for required_dataset in sorted(set(str(dataset) for dataset in required_datasets)):
        if required_dataset not in present:
            errors.append(
                f"{source} missing required {purpose} dataset {required_dataset!r} "
                f"(present={sorted(present)})"
            )
    return errors


def plan_inputs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dataset_root": args.dataset_root,
        "train_manifest": args.train_manifest,
        "val_manifest": args.val_manifest,
        "extra_train_manifest": _list_value(args.extra_train_manifest),
        "require_train_dataset": _list_value(getattr(args, "require_train_dataset", [])),
        "require_val_dataset": _list_value(getattr(args, "require_val_dataset", [])),
        "language": args.language,
        "split_group": args.split_group,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "include_speech": bool(args.include_speech),
    }


def plan_match_errors(plan: dict[str, Any], args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    if plan.get("ok") is not True:
        errors.append("training plan is not ok")
    inputs = plan.get("inputs")
    if not isinstance(inputs, dict):
        errors.append("training plan has no input contract")
        return errors

    expected = plan_inputs(args)
    for key, expected_value in expected.items():
        actual_value = inputs.get(key)
        if key in {"require_train_dataset", "require_val_dataset"} and actual_value is None and expected_value == []:
            continue
        if actual_value != expected_value:
            errors.append(f"training plan input mismatch for {key}: expected {expected_value!r}, got {actual_value!r}")
    return errors


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    errors: list[str] = []
    extra_summaries: dict[str, Any] = {}
    val_ratio_valid = True
    try:
        validate_val_ratio(args.val_ratio)
    except ValueError as exc:
        errors.append(str(exc))
        val_ratio_valid = False

    if args.train_manifest or args.val_manifest:
        if not args.train_manifest or not args.val_manifest:
            errors.append("--train-manifest and --val-manifest must be provided together")
            train_records: list[dict[str, Any]] = []
            val_records: list[dict[str, Any]] = []
        else:
            train_records = read_jsonl(args.train_manifest)
            val_records = read_jsonl(args.val_manifest)
            errors.extend(_require_trainable_manifest(train_records, source=args.train_manifest))
            errors.extend(_require_trainable_manifest(val_records, source=args.val_manifest))
        source = "manifest"
    else:
        if not args.dataset_root:
            errors.append("provide --dataset-root or both --train-manifest and --val-manifest")
            train_records = []
            val_records = []
        elif not val_ratio_valid:
            train_records = []
            val_records = []
        else:
            records = scan_gt_singer_light(args.dataset_root, language=args.language, include_speech=args.include_speech)
            train_records, val_records = split_gt_singer_records(
                records,
                val_ratio=args.val_ratio,
                seed=args.seed,
                group_by=args.split_group,
            )
        source = "gtsinger"

    errors.extend(_coverage_errors(train_records, source="train split", purpose="training"))
    errors.extend(_coverage_errors(val_records, source="validation split", purpose="validation"))
    errors.extend(
        _dataset_presence_errors(
            train_records,
            getattr(args, "require_train_dataset", []),
            source="train split",
            purpose="training",
        )
    )
    errors.extend(
        _dataset_presence_errors(
            val_records,
            getattr(args, "require_val_dataset", []),
            source="validation split",
            purpose="validation",
        )
    )
    if train_records and val_records:
        errors.extend(split_family_compatibility_errors(train_records, val_records, source=source))

    for manifest_path in args.extra_train_manifest:
        extra_records = read_jsonl(manifest_path)
        try:
            require_non_empty_records(extra_records, source=manifest_path, purpose="extra training")
        except ValueError as exc:
            errors.append(str(exc))
        errors.extend(_require_trainable_manifest(extra_records, source=manifest_path))
        extra_summaries[manifest_path] = _manifest_summary(extra_records)

    summary_fn = summarize_gt_singer_records if source == "gtsinger" else _manifest_summary
    report = {
        "ok": not errors,
        "errors": errors,
        "inputs": plan_inputs(args),
        "source": source,
        "split": {
            "group_by": args.split_group,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "train_examples": len(train_records),
            "val_examples": len(val_records),
        },
        "train_summary": summary_fn(train_records),
        "val_summary": summary_fn(val_records),
        "extra_train_summaries": extra_summaries,
    }
    return report


def main() -> None:
    args = parse_args()
    try:
        report = build_plan(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        report = {"ok": False, "errors": [str(exc)]}
    output = json.dumps(report, indent=2, sort_keys=True)
    print(output)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
    if args.strict and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
