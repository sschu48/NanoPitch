"""Compare technique-model evaluation directories."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .evaluation_artifacts import eval_artifact_hashes, validate_eval_artifacts
from .verify_evaluation import verify_evaluation_dir


DEFAULT_GATES = {
    "top2_accuracy": 0.60,
    "clip_macro_f1": 0.35,
    "control_false_positive_rate": 0.25,
    "non_technique_false_positive_rate": 0.25,
    "expected_calibration_error": 0.20,
}
DEFAULT_DELTA_GATES = {
    "top2_accuracy": 0.0,
    "clip_macro_f1": 0.0,
    "control_false_positive_rate": 0.0,
    "non_technique_false_positive_rate": 0.0,
    "expected_calibration_error": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare technique evaluation outputs")
    parser.add_argument("--baseline", required=True, help="Baseline evaluation directory")
    parser.add_argument("--candidate", action="append", default=[], required=True, help="Candidate evaluation directory")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--min-top2", type=float, default=DEFAULT_GATES["top2_accuracy"])
    parser.add_argument("--min-macro-f1", type=float, default=DEFAULT_GATES["clip_macro_f1"])
    parser.add_argument("--max-control-fpr", type=float, default=DEFAULT_GATES["control_false_positive_rate"])
    parser.add_argument(
        "--max-non-technique-fpr",
        type=float,
        default=DEFAULT_GATES["non_technique_false_positive_rate"],
        help="Maximum false-positive rate across gold control/none/unclear rows.",
    )
    parser.add_argument("--max-ece", type=float, default=DEFAULT_GATES["expected_calibration_error"])
    parser.add_argument(
        "--min-top2-delta",
        type=float,
        default=DEFAULT_DELTA_GATES["top2_accuracy"],
        help="Minimum allowed candidate-baseline top2 delta.",
    )
    parser.add_argument(
        "--min-macro-f1-delta",
        type=float,
        default=DEFAULT_DELTA_GATES["clip_macro_f1"],
        help="Minimum allowed candidate-baseline macro F1 delta.",
    )
    parser.add_argument(
        "--max-control-fpr-delta",
        type=float,
        default=DEFAULT_DELTA_GATES["control_false_positive_rate"],
        help="Maximum allowed candidate-baseline control false-positive-rate increase.",
    )
    parser.add_argument(
        "--max-non-technique-fpr-delta",
        type=float,
        default=DEFAULT_DELTA_GATES["non_technique_false_positive_rate"],
        help="Maximum allowed candidate-baseline non-technique false-positive-rate increase.",
    )
    parser.add_argument(
        "--max-ece-delta",
        type=float,
        default=DEFAULT_DELTA_GATES["expected_calibration_error"],
        help="Maximum allowed candidate-baseline expected calibration error increase.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require_probability(value: float, *, name: str) -> float:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0: {value}")
    return value


def require_probability_delta(value: float, *, name: str) -> float:
    if not math.isfinite(value) or value < -1.0 or value > 1.0:
        raise ValueError(f"{name} must be between -1.0 and 1.0: {value}")
    return value


def load_eval_dir(path: str) -> dict[str, Any]:
    root = Path(path)
    missing_artifacts = validate_eval_artifacts(root)
    if missing_artifacts:
        raise FileNotFoundError(
            f"{root} missing required evaluation artifact(s): " + ", ".join(missing_artifacts)
        )
    artifact_verification = verify_evaluation_dir(root)
    if not artifact_verification["ok"]:
        raise ValueError(
            f"{root} failed evaluation provenance verification: "
            + ", ".join(artifact_verification["failed_checks"])
        )
    metrics_path = root / "metrics.json"
    metrics = read_json(metrics_path)
    threshold_sweep = read_json(root / "threshold_sweep.json") if (root / "threshold_sweep.json").exists() else {}
    operating_point = read_json(root / "operating_point.json") if (root / "operating_point.json").exists() else {}
    calibration = read_json(root / "calibration.json") if (root / "calibration.json").exists() else {}
    evaluation_config = read_json(root / "evaluation_config.json")
    return {
        "path": str(root),
        "metrics": metrics,
        "threshold_sweep": threshold_sweep,
        "operating_point": operating_point,
        "calibration": calibration,
        "evaluation_config": evaluation_config,
        "evaluation_artifact_sha256": eval_artifact_hashes(root),
        "artifact_verification": {
            "ok": artifact_verification["ok"],
            "failed_checks": artifact_verification["failed_checks"],
        },
    }


def _file_fingerprint(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    return value if isinstance(value, dict) else {}


def _comparison_contract_errors(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    baseline_config = baseline.get("evaluation_config") if isinstance(baseline.get("evaluation_config"), dict) else {}
    candidate_config = candidate.get("evaluation_config") if isinstance(candidate.get("evaluation_config"), dict) else {}
    errors: list[str] = []

    baseline_manifest = _file_fingerprint(baseline_config, "manifest")
    candidate_manifest = _file_fingerprint(candidate_config, "manifest")
    for key in ("sha256", "bytes"):
        if baseline_manifest.get(key) != candidate_manifest.get(key):
            errors.append(
                f"candidate {candidate['path']} was evaluated on a different manifest {key}: "
                f"baseline={baseline_manifest.get(key)!r}, candidate={candidate_manifest.get(key)!r}"
            )

    baseline_thresholds = baseline_config.get("thresholds") if isinstance(baseline_config.get("thresholds"), dict) else {}
    candidate_thresholds = (
        candidate_config.get("thresholds") if isinstance(candidate_config.get("thresholds"), dict) else {}
    )
    if baseline_thresholds != candidate_thresholds:
        errors.append(
            f"candidate {candidate['path']} was evaluated with different threshold settings: "
            f"baseline={baseline_thresholds!r}, candidate={candidate_thresholds!r}"
        )
    return errors


def metric(metrics: dict[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    if not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"metric {name} must be finite: {value}")
    return numeric


def delta(candidate: dict[str, Any], baseline: dict[str, Any], name: str) -> float | None:
    candidate_value = metric(candidate, name)
    baseline_value = metric(baseline, name)
    if candidate_value is None or baseline_value is None:
        return None
    return candidate_value - baseline_value


def gate_status(metrics: dict[str, Any], gates: dict[str, float]) -> dict[str, dict[str, Any]]:
    checks = {
        "top2_accuracy": {
            "value": metric(metrics, "top2_accuracy"),
            "threshold": gates["top2_accuracy"],
            "direction": ">=",
        },
        "clip_macro_f1": {
            "value": metric(metrics, "clip_macro_f1"),
            "threshold": gates["clip_macro_f1"],
            "direction": ">=",
        },
        "control_false_positive_rate": {
            "value": metric(metrics, "control_false_positive_rate"),
            "threshold": gates["control_false_positive_rate"],
            "direction": "<=",
        },
        "non_technique_false_positive_rate": {
            "value": metric(metrics, "non_technique_false_positive_rate"),
            "threshold": gates["non_technique_false_positive_rate"],
            "direction": "<=",
        },
        "expected_calibration_error": {
            "value": metric(metrics, "expected_calibration_error"),
            "threshold": gates["expected_calibration_error"],
            "direction": "<=",
        },
    }
    for check in checks.values():
        value = check["value"]
        if value is None:
            check["pass"] = None
        elif check["direction"] == ">=":
            check["pass"] = value >= check["threshold"]
        else:
            check["pass"] = value <= check["threshold"]
    return checks


def delta_gate_status(
    candidate_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    delta_gates: dict[str, float],
) -> dict[str, dict[str, Any]]:
    checks = {
        "top2_accuracy_delta": {
            "value": delta(candidate_metrics, baseline_metrics, "top2_accuracy"),
            "threshold": delta_gates["top2_accuracy"],
            "direction": ">=",
        },
        "clip_macro_f1_delta": {
            "value": delta(candidate_metrics, baseline_metrics, "clip_macro_f1"),
            "threshold": delta_gates["clip_macro_f1"],
            "direction": ">=",
        },
        "control_false_positive_rate_delta": {
            "value": delta(candidate_metrics, baseline_metrics, "control_false_positive_rate"),
            "threshold": delta_gates["control_false_positive_rate"],
            "direction": "<=",
        },
        "non_technique_false_positive_rate_delta": {
            "value": delta(candidate_metrics, baseline_metrics, "non_technique_false_positive_rate"),
            "threshold": delta_gates["non_technique_false_positive_rate"],
            "direction": "<=",
        },
        "expected_calibration_error_delta": {
            "value": delta(candidate_metrics, baseline_metrics, "expected_calibration_error"),
            "threshold": delta_gates["expected_calibration_error"],
            "direction": "<=",
        },
    }
    for check in checks.values():
        value = check["value"]
        if value is None:
            check["pass"] = None
        elif check["direction"] == ">=":
            check["pass"] = value >= check["threshold"]
        else:
            check["pass"] = value <= check["threshold"]
    return checks


def promotion_status(*gate_groups: dict[str, dict[str, Any]]) -> dict[str, Any]:
    combined = {name: check for gates in gate_groups for name, check in gates.items()}
    failed = sorted(name for name, check in combined.items() if check.get("pass") is False)
    unknown = sorted(name for name, check in combined.items() if check.get("pass") is None)
    return {
        "eligible": not failed and not unknown,
        "failed_gates": failed,
        "unknown_gates": unknown,
    }


def summarize_candidate(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    gates: dict[str, float],
    delta_gates: dict[str, float],
) -> dict[str, Any]:
    candidate_metrics = candidate["metrics"]
    baseline_metrics = baseline["metrics"]
    names = (
        "prediction_accuracy",
        "top1_accuracy",
        "top2_accuracy",
        "clip_macro_f1",
        "technique_macro_f1_at_0_30",
        "control_false_positive_rate",
        "non_technique_false_positive_rate",
        "expected_calibration_error",
        "maximum_calibration_error",
    )
    candidate_gates = gate_status(candidate_metrics, gates)
    candidate_delta_gates = delta_gate_status(candidate_metrics, baseline_metrics, delta_gates)
    return {
        "path": candidate["path"],
        "metrics": {name: metric(candidate_metrics, name) for name in names},
        "operating_point": summarize_operating_point(candidate.get("operating_point") or candidate_metrics),
        "evaluation_artifact_sha256": candidate.get("evaluation_artifact_sha256") or {},
        "delta_vs_baseline": {name: delta(candidate_metrics, baseline_metrics, name) for name in names},
        "gates": candidate_gates,
        "regression_gates": candidate_delta_gates,
        "promotion": promotion_status(candidate_gates, candidate_delta_gates),
    }


def summarize_operating_point(source: dict[str, Any]) -> dict[str, Any]:
    operating_point = source.get("selected_operating_point") if "selected_operating_point" in source else source
    if not isinstance(operating_point, dict) or not operating_point:
        return {}
    keys = (
        "confidence_threshold",
        "technique_threshold",
        "macro_f1",
        "prediction_accuracy",
        "technique_macro_f1",
        "control_false_positive_rate",
        "non_technique_false_positive_rate",
        "passes_control_false_positive_gate",
        "passes_non_technique_false_positive_gate",
    )
    return {key: operating_point.get(key) for key in keys if key in operating_point}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    if not args.candidate:
        raise ValueError("at least one --candidate evaluation directory is required")
    gates = {
        "top2_accuracy": require_probability(args.min_top2, name="min_top2"),
        "clip_macro_f1": require_probability(args.min_macro_f1, name="min_macro_f1"),
        "control_false_positive_rate": require_probability(args.max_control_fpr, name="max_control_fpr"),
        "non_technique_false_positive_rate": require_probability(
            args.max_non_technique_fpr,
            name="max_non_technique_fpr",
        ),
        "expected_calibration_error": require_probability(args.max_ece, name="max_ece"),
    }
    delta_gates = {
        "top2_accuracy": require_probability_delta(
            getattr(args, "min_top2_delta", DEFAULT_DELTA_GATES["top2_accuracy"]),
            name="min_top2_delta",
        ),
        "clip_macro_f1": require_probability_delta(
            getattr(args, "min_macro_f1_delta", DEFAULT_DELTA_GATES["clip_macro_f1"]),
            name="min_macro_f1_delta",
        ),
        "control_false_positive_rate": require_probability_delta(
            getattr(args, "max_control_fpr_delta", DEFAULT_DELTA_GATES["control_false_positive_rate"]),
            name="max_control_fpr_delta",
        ),
        "non_technique_false_positive_rate": require_probability_delta(
            getattr(args, "max_non_technique_fpr_delta", DEFAULT_DELTA_GATES["non_technique_false_positive_rate"]),
            name="max_non_technique_fpr_delta",
        ),
        "expected_calibration_error": require_probability_delta(
            getattr(args, "max_ece_delta", DEFAULT_DELTA_GATES["expected_calibration_error"]),
            name="max_ece_delta",
        ),
    }
    baseline = load_eval_dir(args.baseline)
    candidates = [load_eval_dir(candidate_path) for candidate_path in args.candidate]
    contract_errors = [
        error
        for candidate in candidates
        for error in _comparison_contract_errors(baseline, candidate)
    ]
    if contract_errors:
        raise ValueError("evaluation directories are not comparable:\n" + "\n".join(contract_errors))
    return {
        "gates": gates,
        "regression_gates": delta_gates,
        "baseline": summarize_candidate(baseline, baseline, gates, delta_gates),
        "candidates": [summarize_candidate(candidate, baseline, gates, delta_gates) for candidate in candidates],
    }


def main() -> None:
    args = parse_args()
    report = build_report(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
