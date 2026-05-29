"""Check local readiness for technique model training and evaluation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from .app_label_coverage import build_report as build_app_label_coverage_report
from .build_manifest import build_app_recordings_manifest
from .manifest import summarize_records, validate_record
from .package_candidate import read_json
from .verify_package import verify_metadata


TECHNIQUE_ROOT = Path(__file__).resolve().parents[1]


def resolve_local_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj

    normalized = path_obj.as_posix()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized == "gt_singer_grader" or normalized.startswith("gt_singer_grader/"):
        return TECHNIQUE_ROOT / normalized
    return path_obj


def is_package_relative_path(path: str | Path) -> bool:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return False
    normalized = path_obj.as_posix()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized == "gt_singer_grader" or normalized.startswith("gt_singer_grader/")


def resolve_audio_root(audio_root: str | Path, label_path: str | Path | None = None) -> Path:
    root = Path(audio_root)
    if root.as_posix() != ".":
        return resolve_local_path(root)

    if label_path is not None:
        try:
            resolve_local_path(label_path).relative_to(TECHNIQUE_ROOT)
        except ValueError:
            return root
        return TECHNIQUE_ROOT
    return root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check technique-model training readiness")
    parser.add_argument("--gtsinger-root", default="./gt_singer_grader/data/GTSinger")
    parser.add_argument("--vocalset-root", default="./gt_singer_grader/data/VocalSet")
    parser.add_argument("--app-audio-dir", default="./gt_singer_grader/data/app_recordings/raw")
    parser.add_argument("--app-audio-root", default=".")
    parser.add_argument("--app-labels", default="./gt_singer_grader/data/app_recordings/review_labels.csv")
    parser.add_argument("--app-collection-plan", default="./gt_singer_grader/data/app_recordings/collection_plan.csv")
    parser.add_argument(
        "--app-collection-plan-json",
        default="./gt_singer_grader/data/app_recordings/collection_plan.json",
    )
    parser.add_argument(
        "--app-collection-checklist",
        default="./gt_singer_grader/data/app_recordings/collection_checklist.csv",
    )
    parser.add_argument("--app-collection-missing", default="./gt_singer_grader/data/app_recordings/collection_missing.csv")
    parser.add_argument("--app-collection-root", default="./gt_singer_grader/data/app_recordings")
    parser.add_argument(
        "--app-collection-materialize-report",
        default="./gt_singer_grader/data/app_recordings/collection_materialize_report.json",
    )
    parser.add_argument(
        "--app-collection-packet-dir",
        default="./gt_singer_grader/data/app_recordings/collection_packet",
    )
    parser.add_argument(
        "--app-collection-packet-summary",
        default="./gt_singer_grader/data/app_recordings/collection_packet_summary.json",
    )
    parser.add_argument("--checkpoint", default="./gt_singer_grader/models/technique_demo_best.pth")
    parser.add_argument("--metadata", default="./gt_singer_grader/models/technique_demo_metadata.json")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when required checks fail")
    return parser.parse_args()


def check_python() -> dict[str, Any]:
    version = sys.version_info
    ok = version >= (3, 9)
    return {
        "name": "python",
        "ok": ok,
        "required": True,
        "detail": f"{version.major}.{version.minor}.{version.micro}",
    }


def check_import(module_name: str, *, required: bool) -> dict[str, Any]:
    found = importlib.util.find_spec(module_name) is not None
    return {
        "name": f"python_module:{module_name}",
        "ok": found,
        "required": required,
        "detail": "installed" if found else "missing",
    }


def check_path(path: str, *, name: str, required: bool, kind: str = "any") -> dict[str, Any]:
    path_obj = resolve_local_path(path)
    if kind == "dir":
        ok = path_obj.is_dir()
    elif kind == "file":
        ok = path_obj.is_file()
    else:
        ok = path_obj.exists()
    return {
        "name": name,
        "ok": ok,
        "required": required,
        "detail": str(path_obj),
    }


def count_files(root: str, pattern: str) -> int:
    path = resolve_local_path(root)
    if not path.exists():
        return 0
    return sum(1 for _item in path.rglob(pattern))


def dataset_check(root: str, *, name: str, required: bool, needs_json: bool = False) -> dict[str, Any]:
    root_path = resolve_local_path(root)
    wav_count = count_files(root, "*.wav")
    json_count = count_files(root, "*.json") if needs_json else None
    ok = wav_count > 0 and (json_count is None or json_count > 0)
    detail = {"path": str(root_path), "wav_count": wav_count}
    if json_count is not None:
        detail["json_count"] = json_count
    return {
        "name": name,
        "ok": ok,
        "required": required,
        "detail": detail,
    }


def check_app_labels(path: str, *, audio_root: str | Path = ".") -> dict[str, Any]:
    label_path = resolve_local_path(path)
    if not label_path.is_file():
        return {
            "name": "app_recording_labels",
            "ok": False,
            "required": False,
            "detail": str(label_path),
        }

    try:
        records = build_app_recordings_manifest(str(label_path), "app_recordings", "app_user")
    except SystemExit as exc:
        return {
            "name": "app_recording_labels",
            "ok": False,
            "required": False,
            "detail": {
                "path": str(label_path),
                "error": str(exc),
            },
        }

    errors: list[str] = []
    for index, record in enumerate(records, start=1):
        errors.extend(validate_record(record, line_number=index))

    if not records:
        errors.append("CSV has no labeled records")
    coverage = build_app_label_coverage_report(
        str(label_path),
        min_per_family=20,
        min_negative=20,
        min_groups=3,
        require_audio_files=True,
        audio_root=resolve_audio_root(audio_root, label_path),
    )

    return {
        "name": "app_recording_labels",
        "ok": not errors,
        "required": False,
        "detail": {
            "path": str(label_path),
            "summary": summarize_records(records),
            "coverage": {
                "ready_for_collection_target": coverage["ready_for_collection_target"],
                "missing_target_families": coverage["missing_target_families"],
                "negative_shortfall": coverage["negative_shortfall"],
                "group_shortfall": coverage["group_shortfall"],
                "unlabeled_records": coverage["unlabeled_records"],
                "missing_audio_file_count": coverage.get("missing_audio_file_count"),
                "intended_family_mismatch_count": coverage.get("intended_family_mismatch_count"),
                "missing_reviewer_id_count": coverage.get("missing_reviewer_id_count"),
                "warnings": coverage["warnings"],
            },
            "errors": errors,
        },
    }


def check_packaged_metadata(path: str) -> dict[str, Any]:
    metadata_path = resolve_local_path(path)
    if not metadata_path.is_file():
        return {
            "name": "packaged_metadata",
            "ok": False,
            "required": False,
            "detail": str(metadata_path),
        }
    try:
        verification = verify_metadata(read_json(metadata_path))
    except Exception as exc:
        return {
            "name": "packaged_metadata",
            "ok": False,
            "required": False,
            "detail": {
                "path": str(metadata_path),
                "error": str(exc),
            },
        }
    return {
        "name": "packaged_metadata",
        "ok": verification["ok"],
        "required": False,
        "detail": {
            "path": str(metadata_path),
            "failed_checks": verification["failed_checks"],
            "checks": verification["checks"],
        },
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    gtsinger_check = dataset_check(args.gtsinger_root, name="dataset:gtsinger", required=True, needs_json=True)
    checks = [
        check_python(),
        check_import("torch", required=True),
        check_import("numpy", required=True),
        check_import("tqdm", required=True),
        check_import("huggingface_hub", required=not gtsinger_check["ok"]),
        check_import("tensorboard", required=False),
        gtsinger_check,
        dataset_check(args.vocalset_root, name="dataset:vocalset", required=False, needs_json=False),
        check_app_labels(args.app_labels, audio_root=getattr(args, "app_audio_root", ".")),
        check_path(args.checkpoint, name="packaged_checkpoint", required=False, kind="file"),
        check_packaged_metadata(args.metadata),
    ]
    required_failures = [check for check in checks if check["required"] and not check["ok"]]
    next_steps, optional_next_steps = remediation_steps(checks, args)
    return {
        "ok": not required_failures,
        "required_failures": [check["name"] for check in required_failures],
        "required_next_steps": next_steps,
        "optional_next_steps": optional_next_steps,
        "next_steps": next_steps,
        "checks": checks,
    }


def remediation_steps(checks: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[str], list[str]]:
    by_name = {str(check["name"]): check for check in checks}
    steps: list[str] = []
    optional_steps: list[str] = []
    app_labels_path = getattr(args, "app_labels", "./gt_singer_grader/data/app_recordings/review_labels.csv")
    collection_base = Path(app_labels_path).parent
    use_label_local_defaults = not is_package_relative_path(app_labels_path)
    app_audio_dir = getattr(
        args,
        "app_audio_dir",
        str(collection_base / "raw") if use_label_local_defaults else "./gt_singer_grader/data/app_recordings/raw",
    )
    app_audio_root = getattr(args, "app_audio_root", ".")
    app_prepare_report = str(Path(app_labels_path).with_name("prepare_report.json"))
    app_collection_plan = getattr(
        args,
        "app_collection_plan",
        str(collection_base / "collection_plan.csv")
        if use_label_local_defaults
        else "./gt_singer_grader/data/app_recordings/collection_plan.csv",
    )
    app_collection_plan_json = getattr(
        args,
        "app_collection_plan_json",
        str(collection_base / "collection_plan.json")
        if use_label_local_defaults
        else "./gt_singer_grader/data/app_recordings/collection_plan.json",
    )
    app_collection_root = getattr(
        args,
        "app_collection_root",
        str(collection_base) if use_label_local_defaults else "./gt_singer_grader/data/app_recordings",
    )
    app_collection_checklist = getattr(
        args,
        "app_collection_checklist",
        str(collection_base / "collection_checklist.csv")
        if use_label_local_defaults
        else "./gt_singer_grader/data/app_recordings/collection_checklist.csv",
    )
    app_collection_missing = getattr(
        args,
        "app_collection_missing",
        str(collection_base / "collection_missing.csv")
        if use_label_local_defaults
        else "./gt_singer_grader/data/app_recordings/collection_missing.csv",
    )
    app_collection_materialize_report = getattr(
        args,
        "app_collection_materialize_report",
        str(collection_base / "collection_materialize_report.json")
        if use_label_local_defaults
        else "./gt_singer_grader/data/app_recordings/collection_materialize_report.json",
    )
    app_collection_packet_dir = getattr(
        args,
        "app_collection_packet_dir",
        str(collection_base / "collection_packet")
        if use_label_local_defaults
        else "./gt_singer_grader/data/app_recordings/collection_packet",
    )
    app_collection_packet_summary = getattr(
        args,
        "app_collection_packet_summary",
        str(collection_base / "collection_packet_summary.json")
        if use_label_local_defaults
        else "./gt_singer_grader/data/app_recordings/collection_packet_summary.json",
    )

    missing_training_modules = [
        name
        for name in ("torch", "numpy", "tqdm", "huggingface_hub")
        if not by_name.get(f"python_module:{name}", {}).get("ok")
    ]
    if missing_training_modules:
        steps.append("Install training dependencies: python3 -m pip install -r gt_singer_grader/requirements-training.txt")

    gtsinger = by_name.get("dataset:gtsinger", {})
    if not gtsinger.get("ok"):
        steps.append(
            "Download GT Singer: python3 -m gt_singer_grader.download_dataset "
            f"--output-dir {args.gtsinger_root} --language English"
        )

    app_labels = by_name.get("app_recording_labels", {})
    if not app_labels.get("ok") and isinstance(app_labels.get("detail"), str):
        plan_exists = resolve_local_path(app_collection_plan).is_file()
        if not plan_exists:
            optional_steps.append(
                "Generate the app recording collection plan from release coverage targets: "
                "python3 -m gt_singer_grader.plan_app_collection "
                f"--csv {app_labels_path} --output-json {app_collection_plan_json} "
                f"--output-csv {app_collection_plan} --clips-per-singer 7"
            )
        elif (
            not resolve_local_path(app_collection_checklist).is_file()
            or not resolve_local_path(app_collection_missing).is_file()
            or not resolve_local_path(app_collection_materialize_report).is_file()
        ):
            optional_steps.append(
                "Materialize the app collection plan into recording folders and collection CSVs: "
                "python3 -m gt_singer_grader.materialize_app_collection "
                f"--plan {app_collection_plan} --root {app_collection_root} "
                f"--checklist {app_collection_checklist} "
                f"--missing-csv {app_collection_missing} "
                f"--report-json {app_collection_materialize_report} --strict"
            )
        elif not resolve_local_path(app_collection_packet_dir).is_dir() or not resolve_local_path(
            app_collection_packet_summary
        ).is_file():
            optional_steps.append(
                "Export per-singer app recording sheets before collection handoff: "
                "python3 -m gt_singer_grader.export_app_collection_packet "
                f"--checklist {app_collection_checklist} --output-dir {app_collection_packet_dir} "
                f"--summary-json {app_collection_packet_summary} --strict"
            )
        optional_steps.append(
            "Collect and label app recordings at "
            f"{app_labels_path}, using gt_singer_grader/app_recordings_review_template.csv. "
            "Use the generated collection checklist and per-singer sheets first. "
            "After WAVs are collected, validate readability and 5-10 second duration with: "
            "python3 -m gt_singer_grader.materialize_app_collection "
            f"--plan {app_collection_plan} --root {app_collection_root} "
            f"--checklist {app_collection_checklist} --missing-csv {app_collection_missing} "
            f"--report-json {app_collection_materialize_report} "
            "--strict --require-audio-files --validate-wav-files --min-wav-seconds 5 --max-wav-seconds 10. "
            "Then prepare the starter CSV with: "
            "python3 -m gt_singer_grader.prepare_app_recordings "
            f"--audio-dir {app_audio_dir} --output {app_labels_path} "
            f"--report-json {app_prepare_report} "
            f"--relative-to . --collection-plan {app_collection_plan} --singer-id-from-parent"
        )
    elif not app_labels.get("ok"):
        optional_steps.append(
            "Fix app recording label CSV errors, then rebuild: python3 -m gt_singer_grader.build_manifest "
            f"app-recordings --csv {app_labels_path} --output ./gt_singer_grader/manifests/app_recordings.jsonl"
        )
    elif isinstance(app_labels.get("detail"), dict):
        coverage = app_labels["detail"].get("coverage")
        if isinstance(coverage, dict) and not coverage.get("ready_for_collection_target"):
            optional_steps.append(
                "App recording labels parse, but coverage is short; refresh the collection plan with: "
                "python3 -m gt_singer_grader.plan_app_collection "
                f"--csv {app_labels_path} --output-json {app_collection_plan_json} "
                f"--output-csv {app_collection_plan} --clips-per-singer 7. "
                "Then inspect collection gaps with: "
                "python3 -m gt_singer_grader.app_label_coverage "
                f"--csv {app_labels_path} --audio-root {app_audio_root} --require-audio-files --strict"
            )

    packaged_metadata = by_name.get("packaged_metadata", {})
    if packaged_metadata and not packaged_metadata.get("ok") and not isinstance(packaged_metadata.get("detail"), str):
        optional_steps.append(
            "Packaged checkpoint metadata is not release-verifiable; repackage after a promoted app-adapted "
            "comparison and app validation audit, then run python3 -m gt_singer_grader.verify_package --strict"
        )

    return steps, optional_steps


def main() -> None:
    args = parse_args()
    report = build_report(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.strict and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
