"""Filter normalized technique manifests by training readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .manifest import (
    read_jsonl,
    summarize_records,
    trainability_reason,
    validate_record,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Separate trainable and evaluation-only manifest records")
    parser.add_argument("--input", required=True, help="Input JSONL manifest")
    parser.add_argument("--trainable-output", required=True)
    parser.add_argument("--eval-only-output", required=True)
    parser.add_argument("--summary-output", default=None)
    return parser.parse_args()


def split_trainable_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    trainable_records: list[dict[str, Any]] = []
    eval_only_records: list[dict[str, Any]] = []

    for record in records:
        reason = trainability_reason(record)
        if reason == "trainable":
            trainable_records.append(record)
        else:
            eval_only_records.append(record)

    input_summary = summarize_records(records)
    summary = {
        "input_records": len(records),
        "trainable_records": len(trainable_records),
        "eval_only_records": len(eval_only_records),
        "reason_counts": input_summary["trainability"],
        "family_counts": {
            "trainable": summarize_records(trainable_records)["families"],
            "eval_only": summarize_records(eval_only_records)["families"],
        },
        "summaries": {
            "input": input_summary,
            "trainable": summarize_records(trainable_records),
            "eval_only": summarize_records(eval_only_records),
        },
    }
    return trainable_records, eval_only_records, summary


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
    trainable_records, eval_only_records, summary = split_trainable_records(records)
    write_jsonl(args.trainable_output, trainable_records)
    write_jsonl(args.eval_only_output, eval_only_records)
    if args.summary_output:
        Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_output, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
