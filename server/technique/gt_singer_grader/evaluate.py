"""Evaluate a technique checkpoint against a saved manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .constants import FAMILY_NAMES, PRIMARY_FAMILY_TO_TECHNIQUES, TECHNIQUE_KEYS
from .manifest import normalize_label_list, require_non_empty_records, validate_record
from .run_metadata import collect_run_metadata, file_metadata


GOLD_LABELS = list(FAMILY_NAMES) + ["none", "unclear", "multiple"]
PREDICTION_LABELS = GOLD_LABELS + ["not_enough_voice"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a technique checkpoint on a manifest")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True, help="JSONL manifest from train.py or build_manifest.py")
    parser.add_argument(
        "--run-config",
        default=None,
        help="Optional training run_config.json to fingerprint with this evaluation.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--confidence-thresholds", default="0.25,0.30,0.35,0.40,0.50,0.60")
    parser.add_argument("--technique-thresholds", default="0.20,0.25,0.30,0.35,0.40,0.50")
    parser.add_argument(
        "--max-control-fpr",
        type=float,
        default=0.25,
        help="Preferred maximum control false-positive rate when selecting an operating point.",
    )
    parser.add_argument(
        "--max-non-technique-fpr",
        type=float,
        default=0.25,
        help="Preferred maximum false-positive rate across gold control/none/unclear rows.",
    )
    parser.add_argument("--calibration-bins", type=int, default=10)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number}: expected JSON object")
            records.append(value)
    return records


def require_json_finite(value: Any, *, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            require_json_finite(item, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            require_json_finite(item, path=f"{path}[{index}]")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON value at {path}: {value}")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    require_json_finite(payload)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")


def parse_thresholds(value: str) -> list[float]:
    thresholds = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not thresholds:
        raise ValueError("at least one threshold is required")
    for threshold in thresholds:
        if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
            raise ValueError(f"threshold must be between 0.0 and 1.0: {threshold}")
    return thresholds


def require_probability(value: float, *, name: str) -> float:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0: {value}")
    return value


def record_audio_path(record: dict[str, Any]) -> str:
    value = record.get("wav_path") or record.get("audio_path")
    if not isinstance(value, str) or not value:
        raise ValueError(f"record missing audio path: {record.get('recording_id') or record.get('stem')}")
    return value


def validate_eval_records(records: list[dict[str, Any]], *, source: str) -> None:
    require_non_empty_records(records, source=source, purpose="evaluation")
    errors: list[str] = []
    for index, record in enumerate(records, start=1):
        if isinstance(record.get("labels"), dict):
            errors.extend(validate_record(record, line_number=index))
            continue
        try:
            record_audio_path(record)
            gold_family(record)
        except ValueError as exc:
            errors.append(f"line {index}: {exc}")
    if errors:
        preview = "\n".join(f"  - {error}" for error in errors[:10])
        extra = "" if len(errors) <= 10 else f"\n  ... and {len(errors) - 10} more"
        raise ValueError(f"{source} is not a valid evaluation manifest:\n{preview}{extra}")


def gold_family(record: dict[str, Any]) -> str:
    if record.get("role") in {"control", "speech"}:
        return "control"
    family = record.get("family")
    if isinstance(family, str) and family:
        return family
    labels = record.get("labels")
    if isinstance(labels, dict):
        families = normalize_label_list(labels.get("families"))
        if len(families) > 1:
            return "multiple"
        if families:
            return str(families[0])
    raise ValueError(f"record missing family label: {record.get('recording_id') or record.get('stem')}")


def gold_techniques_for_family(family: str) -> set[str]:
    return set(PRIMARY_FAMILY_TO_TECHNIQUES.get(family, ()))


def gold_techniques(record: dict[str, Any], family: str | None = None) -> set[str]:
    labels = record.get("labels")
    if isinstance(labels, dict):
        techniques = {
            str(technique)
            for technique in normalize_label_list(labels.get("techniques"))
            if str(technique) in TECHNIQUE_KEYS
        }
        if techniques:
            return techniques
    return gold_techniques_for_family(family or gold_family(record))


def record_song_id(record: dict[str, Any], *, fallback: str) -> str:
    for key in ("song_id", "song", "track_id", "recording_id", "stem"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return fallback


def predicted_family_with_thresholds(
    row: dict[str, Any],
    *,
    confidence_threshold: float,
    technique_threshold: float,
) -> str:
    detected_family = str(row.get("detected_family") or row.get("predicted_family") or "")
    if row["voiced_ratio"] < 0.15:
        return "not_enough_voice"
    if row["detected_confidence"] < confidence_threshold:
        return "unclear"
    if row["primary_technique_score"] < technique_threshold and detected_family != "control":
        return "none"
    return detected_family


def compute_confusion(rows: list[dict[str, Any]], label_key: str = "predicted_family") -> dict[str, dict[str, int]]:
    matrix = {gold: {pred: 0 for pred in PREDICTION_LABELS} for gold in GOLD_LABELS}
    for row in rows:
        gold = str(row["gold_family"])
        pred = str(row[label_key])
        if gold not in matrix:
            continue
        if pred not in matrix[gold]:
            matrix[gold][pred] = 0
        matrix[gold][pred] += 1
    return matrix


def top_k_accuracy(rows: list[dict[str, Any]], k: int) -> float | None:
    if not rows:
        return None
    hits = 0
    for row in rows:
        ranked = row["ranked_families"][:k]
        hits += int(row["gold_family"] in ranked)
    return hits / len(rows)


def prediction_accuracy(rows: list[dict[str, Any]], label_key: str = "predicted_family") -> float | None:
    if not rows:
        return None
    return sum(1 for row in rows if row["gold_family"] == row[label_key]) / len(rows)


def macro_f1(rows: list[dict[str, Any]], label_key: str = "predicted_family") -> float | None:
    scores: list[float] = []
    for label in GOLD_LABELS:
        tp = sum(1 for row in rows if row["gold_family"] == label and row[label_key] == label)
        fp = sum(1 for row in rows if row["gold_family"] != label and row[label_key] == label)
        fn = sum(1 for row in rows if row["gold_family"] == label and row[label_key] != label)
        denom = 2 * tp + fp + fn
        if denom:
            scores.append((2 * tp) / denom)
    return sum(scores) / len(scores) if scores else None


def technique_macro_f1(rows: list[dict[str, Any]], technique_threshold: float = 0.30) -> float | None:
    technique_names = list(TECHNIQUE_KEYS)
    scores: list[float] = []
    for technique in technique_names:
        tp = fp = fn = 0
        for row in rows:
            gold = technique in set(row.get("gold_techniques") or gold_techniques_for_family(str(row["gold_family"])))
            pred = float(row["technique_scores"].get(technique, 0.0)) >= technique_threshold
            tp += int(gold and pred)
            fp += int(not gold and pred)
            fn += int(gold and not pred)
        denom = 2 * tp + fp + fn
        if denom:
            scores.append((2 * tp) / denom)
    return sum(scores) / len(scores) if scores else None


def safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def f1_from_counts(tp: int, fp: int, fn: int) -> float | None:
    denom = 2 * tp + fp + fn
    return (2 * tp) / denom if denom else None


def average_precision(scores: list[tuple[float, bool]]) -> float | None:
    positives = sum(1 for _score, gold in scores if gold)
    if not positives:
        return None
    ranked = sorted(scores, key=lambda item: item[0], reverse=True)
    hits = 0
    precision_sum = 0.0
    for rank, (_score, gold) in enumerate(ranked, start=1):
        if gold:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / positives


def technique_detection_report(
    rows: list[dict[str, Any]],
    technique_thresholds: list[float],
) -> dict[str, Any]:
    techniques: dict[str, Any] = {}
    macro_by_threshold: list[dict[str, Any]] = []

    for technique in TECHNIQUE_KEYS:
        score_gold_pairs = [
            (
                float(row.get("technique_scores", {}).get(technique, 0.0)),
                technique in set(row.get("gold_techniques") or ()),
            )
            for row in rows
        ]
        support = sum(1 for _score, gold in score_gold_pairs if gold)
        negative_support = len(score_gold_pairs) - support
        threshold_rows: list[dict[str, Any]] = []
        for threshold in technique_thresholds:
            tp = fp = tn = fn = 0
            for score, gold in score_gold_pairs:
                pred = score >= threshold
                tp += int(gold and pred)
                fp += int(not gold and pred)
                tn += int(not gold and not pred)
                fn += int(gold and not pred)
            threshold_rows.append(
                {
                    "threshold": threshold,
                    "true_positive": tp,
                    "false_positive": fp,
                    "true_negative": tn,
                    "false_negative": fn,
                    "precision": safe_rate(tp, tp + fp),
                    "recall": safe_rate(tp, tp + fn),
                    "f1": f1_from_counts(tp, fp, fn),
                    "false_positive_rate": safe_rate(fp, fp + tn),
                }
            )
        best = max(
            threshold_rows,
            key=lambda row: (
                -1.0 if row["f1"] is None else float(row["f1"]),
                -1.0 if row["precision"] is None else float(row["precision"]),
                -float(row["false_positive_rate"] or 0.0),
                -float(row["threshold"]),
            ),
        )
        techniques[technique] = {
            "support": support,
            "negative_support": negative_support,
            "average_precision": average_precision(score_gold_pairs),
            "best_threshold": best["threshold"],
            "best_f1": best["f1"],
            "best_precision": best["precision"],
            "best_recall": best["recall"],
            "best_false_positive_rate": best["false_positive_rate"],
            "thresholds": threshold_rows,
        }

    for threshold in technique_thresholds:
        threshold_metrics = [
            row
            for technique in techniques.values()
            for row in technique["thresholds"]
            if row["threshold"] == threshold and row["f1"] is not None
        ]
        macro_by_threshold.append(
            {
                "threshold": threshold,
                "macro_precision": (
                    sum(float(row["precision"]) for row in threshold_metrics if row["precision"] is not None)
                    / len([row for row in threshold_metrics if row["precision"] is not None])
                    if any(row["precision"] is not None for row in threshold_metrics)
                    else None
                ),
                "macro_recall": (
                    sum(float(row["recall"]) for row in threshold_metrics if row["recall"] is not None)
                    / len([row for row in threshold_metrics if row["recall"] is not None])
                    if any(row["recall"] is not None for row in threshold_metrics)
                    else None
                ),
                "macro_f1": (
                    sum(float(row["f1"]) for row in threshold_metrics if row["f1"] is not None)
                    / len([row for row in threshold_metrics if row["f1"] is not None])
                    if any(row["f1"] is not None for row in threshold_metrics)
                    else None
                ),
                "macro_false_positive_rate": (
                    sum(float(row["false_positive_rate"]) for row in threshold_metrics if row["false_positive_rate"] is not None)
                    / len([row for row in threshold_metrics if row["false_positive_rate"] is not None])
                    if any(row["false_positive_rate"] is not None for row in threshold_metrics)
                    else None
                ),
            }
        )

    return {
        "technique_thresholds": technique_thresholds,
        "techniques": techniques,
        "macro_by_threshold": macro_by_threshold,
    }


def song_detection_report(
    rows: list[dict[str, Any]],
    *,
    technique_threshold: float = 0.30,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("song_id") or row["recording_id"]), []).append(row)

    songs: list[dict[str, Any]] = []
    for song_id, song_rows in sorted(grouped.items()):
        gold = sorted({technique for row in song_rows for technique in row.get("gold_techniques", [])})
        scores = {
            technique: max(float(row.get("technique_scores", {}).get(technique, 0.0)) for row in song_rows)
            for technique in TECHNIQUE_KEYS
        }
        predicted = sorted(technique for technique, score in scores.items() if score >= technique_threshold)
        songs.append(
            {
                "song_id": song_id,
                "clips": len(song_rows),
                "gold_techniques": gold,
                "predicted_techniques": predicted,
                "technique_scores": scores,
            }
        )

    technique_rows: dict[str, dict[str, Any]] = {}
    for technique in TECHNIQUE_KEYS:
        tp = fp = tn = fn = 0
        for song in songs:
            gold = technique in song["gold_techniques"]
            pred = technique in song["predicted_techniques"]
            tp += int(gold and pred)
            fp += int(not gold and pred)
            tn += int(not gold and not pred)
            fn += int(gold and not pred)
        technique_rows[technique] = {
            "support": tp + fn,
            "negative_support": fp + tn,
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
            "precision": safe_rate(tp, tp + fp),
            "recall": safe_rate(tp, tp + fn),
            "f1": f1_from_counts(tp, fp, fn),
            "false_positive_rate": safe_rate(fp, fp + tn),
        }

    valid_f1 = [float(row["f1"]) for row in technique_rows.values() if row["f1"] is not None]
    return {
        "technique_threshold": technique_threshold,
        "songs": songs,
        "techniques": technique_rows,
        "macro_f1": sum(valid_f1) / len(valid_f1) if valid_f1 else None,
    }


def false_positive_rate(
    rows: list[dict[str, Any]],
    label_key: str = "predicted_family",
    negative_gold_families: set[str] | None = None,
) -> float | None:
    negative_families = negative_gold_families or {"control"}
    control_rows = [row for row in rows if row["gold_family"] in negative_families]
    if not control_rows:
        return None
    false_positives = [
        row for row in control_rows if row[label_key] not in {"control", "none", "unclear", "not_enough_voice"}
    ]
    return len(false_positives) / len(control_rows)


def confidence_calibration(rows: list[dict[str, Any]], *, bins: int = 10) -> dict[str, Any]:
    if bins <= 0:
        raise ValueError("calibration bins must be positive")
    if not rows:
        return {
            "bins": [],
            "expected_calibration_error": None,
            "maximum_calibration_error": None,
        }

    total = len(rows)
    output_bins: list[dict[str, Any]] = []
    expected_calibration_error = 0.0
    maximum_calibration_error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        if index == bins - 1:
            bucket = [row for row in rows if lower <= row["detected_confidence"] <= upper]
        else:
            bucket = [row for row in rows if lower <= row["detected_confidence"] < upper]

        if bucket:
            accuracy = prediction_accuracy(bucket)
            avg_confidence = sum(float(row["detected_confidence"]) for row in bucket) / len(bucket)
            gap = abs(float(accuracy) - avg_confidence) if accuracy is not None else None
            if gap is not None:
                expected_calibration_error += (len(bucket) / total) * gap
                maximum_calibration_error = max(maximum_calibration_error, gap)
        else:
            accuracy = None
            avg_confidence = None
            gap = None

        output_bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(bucket),
                "accuracy": accuracy,
                "avg_confidence": avg_confidence,
                "gap": gap,
            }
        )

    return {
        "bins": output_bins,
        "expected_calibration_error": expected_calibration_error,
        "maximum_calibration_error": maximum_calibration_error,
    }


def threshold_sweep(
    rows: list[dict[str, Any]],
    confidence_thresholds: list[float],
    technique_thresholds: list[float],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for confidence_threshold in confidence_thresholds:
        for technique_threshold in technique_thresholds:
            sweep_rows = []
            for row in rows:
                next_row = dict(row)
                next_row["sweep_prediction"] = predicted_family_with_thresholds(
                    row,
                    confidence_threshold=confidence_threshold,
                    technique_threshold=technique_threshold,
                )
                sweep_rows.append(next_row)
            results.append(
                {
                    "confidence_threshold": confidence_threshold,
                    "technique_threshold": technique_threshold,
                    "prediction_accuracy": prediction_accuracy(sweep_rows, label_key="sweep_prediction"),
                    "raw_top1_family_accuracy": top_k_accuracy(sweep_rows, 1),
                    "macro_f1": macro_f1(sweep_rows, label_key="sweep_prediction"),
                    "technique_macro_f1": technique_macro_f1(sweep_rows, technique_threshold=technique_threshold),
                    "control_false_positive_rate": false_positive_rate(
                        sweep_rows,
                        label_key="sweep_prediction",
                    ),
                    "non_technique_false_positive_rate": false_positive_rate(
                        sweep_rows,
                        label_key="sweep_prediction",
                        negative_gold_families={"control", "none", "unclear"},
                    ),
                    "prediction_counts": dict(Counter(str(row["sweep_prediction"]) for row in sweep_rows)),
                }
            )
    return results


def select_operating_point(
    sweep: list[dict[str, Any]],
    *,
    max_control_fpr: float = 0.25,
    max_non_technique_fpr: float = 0.25,
) -> dict[str, Any] | None:
    if not sweep:
        return None

    def metric_value(row: dict[str, Any], key: str, default: float) -> float:
        value = row.get(key)
        return default if value is None else float(value)

    def sort_key(row: dict[str, Any]) -> tuple[int, int, float, float, float, float, float, float]:
        control_fpr = metric_value(row, "control_false_positive_rate", 1.0)
        non_technique_fpr = metric_value(row, "non_technique_false_positive_rate", 1.0)
        passes_fpr = int(control_fpr <= max_control_fpr)
        passes_non_technique_fpr = int(non_technique_fpr <= max_non_technique_fpr)
        return (
            passes_fpr,
            passes_non_technique_fpr,
            metric_value(row, "macro_f1", -1.0),
            metric_value(row, "prediction_accuracy", -1.0),
            metric_value(row, "technique_macro_f1", -1.0),
            -control_fpr,
            -non_technique_fpr,
            -metric_value(row, "confidence_threshold", 0.0),
        )

    selected = dict(max(sweep, key=sort_key))
    selected["selection_criteria"] = {
        "max_control_false_positive_rate": max_control_fpr,
        "max_non_technique_false_positive_rate": max_non_technique_fpr,
        "primary_metric": "macro_f1",
        "tie_breakers": [
            "prediction_accuracy",
            "technique_macro_f1",
            "lower_control_false_positive_rate",
            "lower_non_technique_false_positive_rate",
            "lower_confidence_threshold",
        ],
    }
    selected["passes_control_false_positive_gate"] = (
        selected.get("control_false_positive_rate") is not None
        and float(selected["control_false_positive_rate"]) <= max_control_fpr
    )
    selected["passes_non_technique_false_positive_gate"] = (
        selected.get("non_technique_false_positive_rate") is not None
        and float(selected["non_technique_false_positive_rate"]) <= max_non_technique_fpr
    )
    return selected


def write_predictions_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "recording_id",
        "audio_path",
        "gold_family",
        "gold_techniques",
        "predicted_family",
        "detected_confidence",
        "family_margin",
        "detection_status",
        "primary_technique",
        "primary_technique_score",
        "voiced_ratio",
        "top2_families",
        "technique_scores_json",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "recording_id": row["recording_id"],
                    "audio_path": row["audio_path"],
                    "gold_family": row["gold_family"],
                    "gold_techniques": ",".join(row.get("gold_techniques") or []),
                    "predicted_family": row["predicted_family"],
                    "detected_confidence": row["detected_confidence"],
                    "family_margin": row["family_margin"],
                    "detection_status": row["detection_status"],
                    "primary_technique": row["primary_technique"],
                    "primary_technique_score": row["primary_technique_score"],
                    "voiced_ratio": row["voiced_ratio"],
                    "top2_families": ",".join(row["ranked_families"][:2]),
                    "technique_scores_json": json.dumps(row["technique_scores"], sort_keys=True),
                }
            )


def write_technique_detection_csv(path: str | Path, report: dict[str, Any]) -> None:
    fieldnames = [
        "technique",
        "threshold",
        "support",
        "negative_support",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "average_precision",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for technique, payload in report.get("techniques", {}).items():
            for row in payload.get("thresholds", []):
                writer.writerow(
                    {
                        "technique": technique,
                        "threshold": row["threshold"],
                        "support": payload["support"],
                        "negative_support": payload["negative_support"],
                        "precision": row["precision"],
                        "recall": row["recall"],
                        "f1": row["f1"],
                        "false_positive_rate": row["false_positive_rate"],
                        "average_precision": payload["average_precision"],
                        "true_positive": row["true_positive"],
                        "false_positive": row["false_positive"],
                        "true_negative": row["true_negative"],
                        "false_negative": row["false_negative"],
                    }
                )


def write_song_detection_csv(path: str | Path, report: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "song_id",
                "clips",
                "gold_techniques",
                "predicted_techniques",
                "technique_scores_json",
            ],
        )
        writer.writeheader()
        for song in report.get("songs", []):
            writer.writerow(
                {
                    "song_id": song["song_id"],
                    "clips": song["clips"],
                    "gold_techniques": ",".join(song["gold_techniques"]),
                    "predicted_techniques": ",".join(song["predicted_techniques"]),
                    "technique_scores_json": json.dumps(song["technique_scores"], sort_keys=True),
                }
            )


def write_confusion_csv(path: str | Path, matrix: dict[str, dict[str, int]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gold_family", *PREDICTION_LABELS])
        for family in GOLD_LABELS:
            writer.writerow([family, *[matrix.get(family, {}).get(label, 0) for label in PREDICTION_LABELS]])


def write_calibration_csv(path: str | Path, calibration: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["lower", "upper", "count", "accuracy", "avg_confidence", "gap"],
        )
        writer.writeheader()
        for row in calibration.get("bins", []):
            writer.writerow(row)


def build_evaluation_config(args: argparse.Namespace, *, examples: int) -> dict[str, Any]:
    max_control_fpr = require_probability(args.max_control_fpr, name="max_control_fpr")
    max_non_technique_fpr = require_probability(args.max_non_technique_fpr, name="max_non_technique_fpr")
    config = {
        "checkpoint": file_metadata(args.checkpoint),
        "manifest": file_metadata(args.manifest),
        "output_dir": str(Path(args.output_dir)),
        "device": args.device,
        "max_examples": args.max_examples,
        "examples": examples,
        "thresholds": {
            "confidence_thresholds": parse_thresholds(args.confidence_thresholds),
            "technique_thresholds": parse_thresholds(args.technique_thresholds),
            "max_control_false_positive_rate": max_control_fpr,
            "max_non_technique_false_positive_rate": max_non_technique_fpr,
            "calibration_bins": args.calibration_bins,
        },
        "environment": collect_run_metadata(Path(__file__).resolve().parents[3]),
    }
    if getattr(args, "run_config", None):
        config["run_config"] = file_metadata(args.run_config)
    return config


def main() -> None:
    from .infer import load_predictor, predict_summary

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(args.manifest)
    if args.max_examples is not None:
        records = records[: args.max_examples]
    try:
        validate_eval_records(records, source=args.manifest)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    predictor = load_predictor(args.checkpoint, device_name=args.device)

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        audio_path = record_audio_path(record)
        family = gold_family(record)
        summary = predict_summary(predictor, audio_path)
        family_probabilities = dict(summary.get("family_probabilities") or {})
        ranked_families = [
            name for name, _score in sorted(family_probabilities.items(), key=lambda item: item[1], reverse=True)
        ]
        row = {
            "recording_id": record.get("recording_id") or record.get("stem") or f"record-{index}",
            "song_id": record_song_id(record, fallback=f"record-{index}"),
            "audio_path": audio_path,
            "gold_family": family,
            "gold_techniques": sorted(gold_techniques(record, family)),
            "predicted_family": summary.get("detected_family"),
            "detected_confidence": float(summary.get("detected_confidence", 0.0)),
            "family_margin": float(summary.get("family_margin", 0.0)),
            "detection_status": summary.get("detection_status"),
            "primary_technique": summary.get("primary_technique"),
            "primary_technique_score": float(summary.get("primary_technique_score", 0.0)),
            "voiced_ratio": float(summary.get("voiced_ratio", 0.0)),
            "ranked_families": ranked_families,
            "technique_scores": dict(summary.get("technique_scores") or {}),
        }
        rows.append(row)
        print(f"[{index}/{len(records)}] {row['recording_id']} gold={family} pred={row['predicted_family']}")

    confidence_thresholds = parse_thresholds(args.confidence_thresholds)
    technique_thresholds = parse_thresholds(args.technique_thresholds)
    max_control_fpr = require_probability(args.max_control_fpr, name="max_control_fpr")
    max_non_technique_fpr = require_probability(args.max_non_technique_fpr, name="max_non_technique_fpr")
    metrics = {
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "examples": len(rows),
        "prediction_accuracy": prediction_accuracy(rows),
        "top1_accuracy": top_k_accuracy(rows, 1),
        "top2_accuracy": top_k_accuracy(rows, 2),
        "clip_macro_f1": macro_f1(rows),
        "technique_macro_f1_at_0_30": technique_macro_f1(rows, technique_threshold=0.30),
        "control_false_positive_rate": false_positive_rate(rows),
        "non_technique_false_positive_rate": false_positive_rate(
            rows,
            negative_gold_families={"control", "none", "unclear"},
        ),
        "gold_counts": dict(Counter(str(row["gold_family"]) for row in rows)),
        "prediction_counts": dict(Counter(str(row["predicted_family"]) for row in rows)),
        "detection_status_counts": dict(Counter(str(row["detection_status"]) for row in rows)),
    }
    sweep = threshold_sweep(rows, confidence_thresholds, technique_thresholds)
    operating_point = select_operating_point(
        sweep,
        max_control_fpr=max_control_fpr,
        max_non_technique_fpr=max_non_technique_fpr,
    )
    matrix = compute_confusion(rows)
    calibration = confidence_calibration(rows, bins=args.calibration_bins)
    technique_report = technique_detection_report(rows, technique_thresholds)
    selected_technique_threshold = (
        float(operating_point["technique_threshold"])
        if isinstance(operating_point, dict) and operating_point.get("technique_threshold") is not None
        else 0.30
    )
    song_report = song_detection_report(rows, technique_threshold=selected_technique_threshold)
    metrics["expected_calibration_error"] = calibration["expected_calibration_error"]
    metrics["maximum_calibration_error"] = calibration["maximum_calibration_error"]
    metrics["selected_operating_point"] = operating_point
    metrics["song_detection_macro_f1"] = song_report["macro_f1"]
    evaluation_config = build_evaluation_config(args, examples=len(rows))

    write_json(output_dir / "evaluation_config.json", evaluation_config)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "threshold_sweep.json", {"results": sweep})
    write_json(output_dir / "operating_point.json", operating_point or {})
    write_json(output_dir / "calibration.json", calibration)
    write_json(output_dir / "technique_detection.json", technique_report)
    write_json(output_dir / "song_detection.json", song_report)
    write_predictions_csv(output_dir / "predictions.csv", rows)
    write_confusion_csv(output_dir / "confusion_matrix.csv", matrix)
    write_calibration_csv(output_dir / "calibration.csv", calibration)
    write_technique_detection_csv(output_dir / "technique_detection.csv", technique_report)
    write_song_detection_csv(output_dir / "song_detection.csv", song_report)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
