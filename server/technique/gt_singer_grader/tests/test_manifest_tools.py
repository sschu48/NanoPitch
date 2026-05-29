from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
import wave
from argparse import Namespace
from pathlib import Path
from unittest import mock

from api import load_package_metadata
from gt_singer_grader.app_label_coverage import build_report as build_app_label_coverage_report
from gt_singer_grader.audit_app_validation import audit_records as audit_app_validation_records
from gt_singer_grader.build_manifest import build_app_recordings_manifest, build_vocalset_manifest
from gt_singer_grader.compare_runs import build_report as build_compare_report
from gt_singer_grader.constants import FAMILY_NAMES, TECHNIQUE_KEYS
from gt_singer_grader.dataset_strategy import audit_registry
from gt_singer_grader.download_dataset import allow_patterns, ignore_patterns, load_snapshot_download
from gt_singer_grader.evaluate import (
    build_evaluation_config,
    confidence_calibration,
    false_positive_rate,
    gold_family,
    parse_thresholds,
    prediction_accuracy,
    predicted_family_with_thresholds,
    require_probability,
    select_operating_point,
    threshold_sweep,
    top_k_accuracy,
    validate_eval_records,
    write_json,
)
from gt_singer_grader.experiment_status import (
    app_collection_materialize_status,
    app_collection_packet_status,
    app_collection_plan_status,
    app_prepare_report_status,
    build_report as build_experiment_status_report,
    command_list as experiment_command_list,
    current_stage as experiment_current_stage,
    file_status as experiment_file_status,
    next_actions as experiment_next_actions,
)
from gt_singer_grader.export_app_collection_packet import export_packet
from gt_singer_grader.evaluation_artifacts import eval_artifact_hashes
from gt_singer_grader.filter_manifest import split_trainable_records
from gt_singer_grader.manifest import (
    normalize_label_list,
    read_jsonl,
    require_non_empty_records,
    summarize_records,
    trainability_reason,
    validate_record,
    write_jsonl,
)
from gt_singer_grader.materialize_app_collection import materialize_collection, read_plan as read_materialized_plan
from gt_singer_grader.merge_manifest import merge_manifest_records
from gt_singer_grader.package_candidate import REQUIRED_EVALUATION_ARTIFACTS, candidate_kind, package_candidate, sha256_file
from gt_singer_grader.plan_app_collection import build_collection_plan, write_plan_csv
from gt_singer_grader.plan_training import build_plan as build_training_plan
from gt_singer_grader.plan_training import plan_match_errors
from gt_singer_grader.preflight import build_report as build_preflight_report
from gt_singer_grader.prepare_app_recordings import (
    build_report as build_prepare_app_recordings_report,
    build_review_rows,
    collection_plan_match_report,
    discover_wavs,
    read_collection_plan,
    write_review_csv,
)
from gt_singer_grader.run_metadata import collect_run_metadata, file_metadata
from gt_singer_grader.sample_manifest import sample_records
from gt_singer_grader.split_health import family_counts as split_family_counts
from gt_singer_grader.split_health import split_coverage_errors
from gt_singer_grader.split_health import split_family_compatibility_errors
from gt_singer_grader.split_manifest import split_manifest_records
from gt_singer_grader.verify_evaluation import verify_evaluation_dir
from gt_singer_grader.verify_package import verify_metadata
from gt_singer_grader.verify_run import verify_run_config


def write_eval_artifacts(eval_dir: Path, *, manifest_path: Path | None = None) -> dict[str, Path]:
    eval_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = eval_dir.parent / f"{eval_dir.name}_checkpoint.pth"
    manifest = manifest_path or eval_dir.parent / "eval_manifest.jsonl"
    checkpoint.write_bytes(b"checkpoint")
    if not manifest.exists():
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{}\n", encoding="utf-8")
    for name in REQUIRED_EVALUATION_ARTIFACTS:
        path = eval_dir / name
        if name == "evaluation_config.json":
            path.write_text(
                json.dumps(
                    {
                        "checkpoint": file_metadata(checkpoint),
                        "manifest": file_metadata(manifest),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif name.endswith(".json"):
            path.write_text("{}\n", encoding="utf-8")
        else:
            path.write_text("header\n", encoding="utf-8")
    return {"checkpoint": checkpoint, "manifest": manifest}


def write_tiny_wav(path: Path, *, seconds: float = 0.001) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_rate = 16_000
    frames = max(1, int(frame_rate * seconds))
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(frame_rate)
        audio.writeframes(b"\x00\x00" * frames)


def write_app_domain_comparison(
    comparison: Path,
    candidate_eval_dir: Path,
    *,
    app_manifest: Path,
    candidate_metrics: dict[str, object] | None = None,
    promotion: dict[str, object] | None = None,
) -> Path:
    baseline_eval_dir = comparison.parent / "gtsinger_song_aug_v1" / "eval_app"
    write_eval_artifacts(baseline_eval_dir, manifest_path=app_manifest)
    comparison.write_text(
        json.dumps(
            {
                "gates": {"top2_accuracy": 0.6},
                "regression_gates": {"top2_accuracy": 0.0},
                "baseline": {
                    "path": str(baseline_eval_dir),
                    "evaluation_artifact_sha256": eval_artifact_hashes(baseline_eval_dir),
                },
                "candidates": [
                    {
                        "path": str(candidate_eval_dir),
                        "evaluation_artifact_sha256": eval_artifact_hashes(candidate_eval_dir),
                        "metrics": candidate_metrics or {"top2_accuracy": 0.7},
                        "operating_point": {"confidence_threshold": 0.35},
                        "delta_vs_baseline": {"top2_accuracy": 0.1},
                        "promotion": promotion
                        or {
                            "eligible": True,
                            "failed_gates": [],
                            "unknown_gates": [],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return baseline_eval_dir


def write_app_validation_audit(
    audit_path: Path,
    *,
    manifest_path: Path | None = None,
    ready: bool = True,
    records: int = 42,
    warnings: list[str] | None = None,
) -> Path:
    if manifest_path is None:
        manifest_path = audit_path.parent / "app_recordings_eval.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        manifest_path.write_text("{}\n", encoding="utf-8")
    audit_path.write_text(
        json.dumps(
            {
                "ready_for_mvp_validation": ready,
                "records": records,
                "warnings": warnings or [],
                "manifest": file_metadata(manifest_path),
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


class ManifestToolsTest(unittest.TestCase):
    def test_api_package_metadata_marks_only_app_adapted_packages_release_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = root / "metadata.json"
            payload = {
                "model_contract": {"candidate_kind": "vocalset"},
                "promotion": {"eligible": True, "failed_gates": [], "unknown_gates": []},
                "app_validation_audit": {"ready_for_mvp_validation": True},
                "app_domain_comparison": {"ok": True, "failed_checks": []},
            }
            metadata.write_text(json.dumps(payload) + "\n", encoding="utf-8")

            public_report = load_package_metadata(metadata)
            payload["model_contract"]["candidate_kind"] = "app_adapted"
            metadata.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            app_report = load_package_metadata(metadata)
            payload.pop("app_domain_comparison")
            metadata.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            missing_app_domain_report = load_package_metadata(metadata)

        self.assertFalse(public_report["release_ready"])
        self.assertEqual(public_report["candidate_kind"], "vocalset")
        self.assertTrue(public_report["promotion_eligible"])
        self.assertTrue(public_report["app_validation_ready"])
        self.assertTrue(public_report["app_domain_comparison_ready"])
        self.assertTrue(app_report["release_ready"])
        self.assertEqual(app_report["candidate_kind"], "app_adapted")
        self.assertTrue(app_report["app_domain_comparison_ready"])
        self.assertFalse(missing_app_domain_report["release_ready"])
        self.assertFalse(missing_app_domain_report["app_domain_comparison_ready"])

    def test_api_package_metadata_missing_file_is_not_release_ready(self) -> None:
        report = load_package_metadata(Path("/tmp/nanopitch_missing_package_metadata.json"))

        self.assertFalse(report["exists"])
        self.assertFalse(report["release_ready"])
        self.assertIsNone(report["candidate_kind"])

    def test_collect_run_metadata_includes_python_platform_and_git_keys(self) -> None:
        metadata = collect_run_metadata(Path(__file__).resolve().parents[3])

        self.assertIn("version", metadata["python"])
        self.assertIn("system", metadata["platform"])
        self.assertIn("dirty", metadata["git"])
        self.assertIsInstance(metadata["git"]["dirty"], bool)

    def test_file_metadata_includes_size_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.jsonl"
            path.write_text("record\n", encoding="utf-8")

            metadata = file_metadata(path)

        self.assertTrue(metadata["exists"])
        self.assertEqual(metadata["bytes"], 7)
        self.assertEqual(len(metadata["sha256"]), 64)

    def test_experiment_file_status_reports_stale_source_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "review_labels.csv"
            output = root / "app_recordings.jsonl"
            source.write_text("audio_path\nraw/a.wav\n", encoding="utf-8")
            output.write_text("{}\n", encoding="utf-8")
            old_time = 1_700_000_000
            new_time = old_time + 10
            os.utime(output, (old_time, old_time))
            os.utime(source, (new_time, new_time))

            status = experiment_file_status(output, [source])

        self.assertTrue(status["exists"])
        self.assertFalse(status["current_for_sources"])
        self.assertEqual(status["stale_source_files"], [str(source)])

    def test_experiment_file_status_reports_missing_source_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "app_recordings.jsonl"
            missing_source = root / "review_labels.csv"
            output.write_text("{}\n", encoding="utf-8")

            status = experiment_file_status(output, [missing_source])

        self.assertTrue(status["exists"])
        self.assertFalse(status["current_for_sources"])
        self.assertEqual(status["missing_source_files"], [str(missing_source)])

    def test_verify_run_config_accepts_artifact_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_manifest = root / "train_manifest.jsonl"
            val_manifest = root / "val_manifest.jsonl"
            train_manifest.write_text("{}\n", encoding="utf-8")
            val_manifest.write_text("{}\n", encoding="utf-8")
            config = {
                "artifacts": {
                    "train_manifest": file_metadata(train_manifest),
                    "val_manifest": file_metadata(val_manifest),
                }
            }

            report = verify_run_config(config)

        self.assertTrue(report["ok"])
        self.assertEqual(report["failed_checks"], [])

    def test_verify_run_config_reports_tampered_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_manifest = root / "train_manifest.jsonl"
            train_manifest.write_text("{}\n", encoding="utf-8")
            config = {"artifacts": {"train_manifest": file_metadata(train_manifest)}}
            train_manifest.write_text("{\"changed\": true}\n", encoding="utf-8")

            report = verify_run_config(config)

        self.assertFalse(report["ok"])
        self.assertIn("artifacts.train_manifest:sha256", report["failed_checks"])
        self.assertIn("artifacts.train_manifest:bytes", report["failed_checks"])

    def test_verify_run_config_reports_duplicate_metric_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_manifest = root / "train_manifest.jsonl"
            train_manifest.write_text("{}\n", encoding="utf-8")
            metrics_history = root / "metrics_history.jsonl"
            metrics_history.write_text(
                json.dumps({"epoch": 1}) + "\n" + json.dumps({"epoch": 1}) + "\n",
                encoding="utf-8",
            )
            config = {"artifacts": {"train_manifest": file_metadata(train_manifest)}}

            report = verify_run_config(config)

        self.assertFalse(report["ok"])
        self.assertIn("metrics_history:unique_epochs", report["failed_checks"])
        self.assertIn("metrics_history:contiguous_epochs", report["failed_checks"])

    def test_verify_run_config_reports_tampered_training_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            training_plan = root / "training_plan.json"
            training_plan.write_text('{"ok": true}\n', encoding="utf-8")
            config = {"artifacts": {"training_plan": file_metadata(training_plan)}}
            training_plan.write_text('{"ok": false}\n', encoding="utf-8")

            report = verify_run_config(config)

        self.assertFalse(report["ok"])
        self.assertIn("artifacts.training_plan:sha256", report["failed_checks"])
        self.assertIn("artifacts.training_plan:bytes", report["failed_checks"])

    def test_evaluation_config_fingerprints_checkpoint_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "best.pth"
            manifest = root / "val_manifest.jsonl"
            run_config = root / "run_config.json"
            checkpoint.write_bytes(b"checkpoint")
            manifest.write_text("{}\n", encoding="utf-8")
            run_config.write_text('{"artifacts": {}}\n', encoding="utf-8")
            checkpoint_sha256 = sha256_file(checkpoint)
            manifest_sha256 = sha256_file(manifest)
            run_config_sha256 = sha256_file(run_config)

            config = build_evaluation_config(
                Namespace(
                    checkpoint=str(checkpoint),
                    manifest=str(manifest),
                    run_config=str(run_config),
                    output_dir=str(root / "eval"),
                    device="cpu",
                    max_examples=None,
                    confidence_thresholds="0.25,0.5",
                    technique_thresholds="0.2,0.4",
                    max_control_fpr=0.25,
                    max_non_technique_fpr=0.2,
                    calibration_bins=5,
                ),
                examples=12,
            )

        self.assertEqual(config["examples"], 12)
        self.assertEqual(config["checkpoint"]["sha256"], checkpoint_sha256)
        self.assertEqual(config["manifest"]["sha256"], manifest_sha256)
        self.assertEqual(config["run_config"]["sha256"], run_config_sha256)
        self.assertEqual(config["thresholds"]["confidence_thresholds"], [0.25, 0.5])
        self.assertEqual(config["thresholds"]["max_non_technique_false_positive_rate"], 0.2)

    def test_verify_evaluation_reports_tampered_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_dir = root / "eval"
            checkpoint = root / "best.pth"
            manifest = root / "val_manifest.jsonl"
            checkpoint.write_bytes(b"checkpoint")
            manifest.write_text("{}\n", encoding="utf-8")
            write_eval_artifacts(eval_dir)
            config = build_evaluation_config(
                Namespace(
                    checkpoint=str(checkpoint),
                    manifest=str(manifest),
                    run_config=None,
                    output_dir=str(eval_dir),
                    device="cpu",
                    max_examples=None,
                    confidence_thresholds="0.25",
                    technique_thresholds="0.2",
                    max_control_fpr=0.25,
                    max_non_technique_fpr=0.25,
                    calibration_bins=5,
                ),
                examples=1,
            )
            (eval_dir / "evaluation_config.json").write_text(json.dumps(config), encoding="utf-8")
            manifest.write_text("{\"changed\": true}\n", encoding="utf-8")

            report = verify_evaluation_dir(eval_dir)

        self.assertFalse(report["ok"])
        self.assertIn("evaluation_config.manifest:sha256", report["failed_checks"])
        self.assertIn("evaluation_config.manifest:bytes", report["failed_checks"])

    def test_verify_evaluation_reports_tampered_run_config_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_dir = root / "eval"
            checkpoint = root / "best.pth"
            manifest = root / "val_manifest.jsonl"
            train_manifest = root / "train_manifest.jsonl"
            run_config = root / "run_config.json"
            checkpoint.write_bytes(b"checkpoint")
            manifest.write_text("{}\n", encoding="utf-8")
            train_manifest.write_text("{}\n", encoding="utf-8")
            run_config.write_text(
                json.dumps({"artifacts": {"train_manifest": file_metadata(train_manifest)}}) + "\n",
                encoding="utf-8",
            )
            write_eval_artifacts(eval_dir)
            config = build_evaluation_config(
                Namespace(
                    checkpoint=str(checkpoint),
                    manifest=str(manifest),
                    run_config=str(run_config),
                    output_dir=str(eval_dir),
                    device="cpu",
                    max_examples=None,
                    confidence_thresholds="0.25",
                    technique_thresholds="0.2",
                    max_control_fpr=0.25,
                    max_non_technique_fpr=0.25,
                    calibration_bins=5,
                ),
                examples=1,
            )
            (eval_dir / "evaluation_config.json").write_text(json.dumps(config), encoding="utf-8")
            train_manifest.write_text("{\"changed\": true}\n", encoding="utf-8")

            report = verify_evaluation_dir(eval_dir)

        self.assertFalse(report["ok"])
        self.assertIn("evaluation_config.run_config:artifacts", report["failed_checks"])

    def test_normalize_label_list_accepts_csv_style_values(self) -> None:
        self.assertEqual(normalize_label_list("vibrato, breathy; falsetto"), ["vibrato", "breathy", "falsetto"])
        self.assertEqual(normalize_label_list(["vibrato", ""]), ["vibrato"])
        self.assertEqual(normalize_label_list(""), [])

    def test_app_recording_csv_builds_valid_manifest_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "labels.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "audio_path",
                        "recording_id",
                        "singer_id",
                        "song_id",
                        "families",
                        "techniques",
                        "split_group",
                        "label_source",
                        "notes",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "audio_path": "data/app_recordings/take.wav",
                        "recording_id": "app:take-001",
                        "singer_id": "singer_001",
                        "song_id": "warmup",
                        "families": "vibrato",
                        "techniques": "vibrato",
                        "split_group": "",
                        "label_source": "coach_review",
                        "notes": "clear vibrato",
                    }
                )

            records = build_app_recordings_manifest(str(csv_path), "app_recordings", "app_user")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["split_group"], "singer_001")
        self.assertEqual(records[0]["labels"]["families"], ["vibrato"])
        self.assertEqual(records[0]["labels"]["techniques"], ["vibrato"])
        self.assertEqual(validate_record(records[0]), [])

    def test_app_recording_review_strengths_derive_training_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "review.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "audio_path",
                        "recording_id",
                        "singer_id",
                        "song_id",
                        "intended_family",
                        "mix",
                        "falsetto",
                        "breathy",
                        "pharyngeal",
                        "glissando",
                        "vibrato",
                        "families",
                        "techniques",
                        "split_group",
                        "label_source",
                        "reviewer_id",
                        "notes",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "audio_path": "data/app_recordings/take.wav",
                        "recording_id": "app:take-002",
                        "singer_id": "singer_002",
                        "song_id": "warmup",
                        "intended_family": "mixed_voice",
                        "mix": "strong",
                        "falsetto": "weak",
                        "breathy": "absent",
                        "pharyngeal": "absent",
                        "glissando": "absent",
                        "vibrato": "present",
                        "families": "",
                        "techniques": "",
                        "split_group": "",
                        "label_source": "coach_review",
                        "reviewer_id": "reviewer_a",
                        "notes": "mixed voice with vibrato",
                    }
                )

            records = build_app_recordings_manifest(str(csv_path), "app_recordings", "app_user")

        self.assertEqual(records[0]["labels"]["families"], ["mixed_voice", "vibrato"])
        self.assertEqual(records[0]["labels"]["techniques"], ["mix", "vibrato"])
        self.assertEqual(records[0]["labels"]["technique_strengths"]["falsetto"], "weak")
        self.assertEqual(records[0]["labels"]["intended_family"], "mixed_voice")
        self.assertEqual(records[0]["metadata"]["reviewer_id"], "reviewer_a")
        self.assertEqual(validate_record(records[0]), [])

    def test_app_recording_manifest_rejects_duplicate_review_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "review.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "audio_path",
                        "recording_id",
                        "singer_id",
                        "song_id",
                        "intended_family",
                        "mix",
                        "falsetto",
                        "breathy",
                        "pharyngeal",
                        "glissando",
                        "vibrato",
                        "families",
                        "techniques",
                        "split_group",
                        "label_source",
                        "reviewer_id",
                        "notes",
                    ],
                )
                writer.writeheader()
                base_row = {
                    "audio_path": "data/app_recordings/duplicate.wav",
                    "recording_id": "app:duplicate",
                    "singer_id": "singer_001",
                    "song_id": "warmup",
                    "intended_family": "vibrato",
                    "mix": "absent",
                    "falsetto": "absent",
                    "breathy": "absent",
                    "pharyngeal": "absent",
                    "glissando": "absent",
                    "vibrato": "present",
                    "families": "",
                    "techniques": "",
                    "split_group": "singer_001",
                    "label_source": "coach_review",
                    "reviewer_id": "reviewer_a",
                    "notes": "",
                }
                writer.writerow(base_row)
                writer.writerow({**base_row, "singer_id": "singer_002", "split_group": "singer_002"})

            with self.assertRaises(SystemExit) as exc:
                build_app_recordings_manifest(str(csv_path), "app_recordings", "app_user")

        self.assertIn("duplicate audio_path", str(exc.exception))
        self.assertIn("row 2", str(exc.exception))
        self.assertIn("row 3", str(exc.exception))

    def test_prepare_app_recordings_builds_review_csv_from_wavs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_dir = root / "raw"
            singer_dir = audio_dir / "singer_001"
            singer_dir.mkdir(parents=True)
            (singer_dir / "take_b.wav").write_bytes(b"")
            (singer_dir / "take_a.wav").write_bytes(b"")
            (singer_dir / "ignore.txt").write_text("not audio", encoding="utf-8")
            output = root / "review_labels.csv"

            wavs = discover_wavs(audio_dir)
            rows = build_review_rows(
                wavs,
                relative_to=root,
                recording_prefix="app",
                label_source="coach_review",
                reviewer_id="reviewer_a",
                song_id="warmup",
                intended_family="vibrato",
                singer_id_from_parent=True,
            )
            write_review_csv(output, rows)

            with output.open("r", encoding="utf-8", newline="") as handle:
                written = list(csv.DictReader(handle))

        self.assertEqual([path.name for path in wavs], ["take_a.wav", "take_b.wav"])
        self.assertEqual(len(written), 2)
        self.assertEqual(written[0]["audio_path"], "raw/singer_001/take_a.wav")
        self.assertEqual(written[0]["recording_id"], "app:raw_singer_001_take_a")
        self.assertEqual(written[0]["singer_id"], "singer_001")
        self.assertEqual(written[0]["split_group"], "singer_001")
        self.assertEqual(written[0]["intended_family"], "vibrato")
        self.assertEqual(written[0]["reviewer_id"], "reviewer_a")
        self.assertEqual(written[0]["vibrato"], "")

    def test_prepare_app_recordings_recording_ids_include_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            (raw_dir / "singer_001").mkdir(parents=True)
            (raw_dir / "singer_002").mkdir(parents=True)
            (raw_dir / "singer_001" / "take_01.wav").write_bytes(b"")
            (raw_dir / "singer_002" / "take_01.wav").write_bytes(b"")

            rows = build_review_rows(
                discover_wavs(raw_dir),
                relative_to=root,
                recording_prefix="app",
                label_source="coach_review",
                reviewer_id="",
                song_id="",
                intended_family="",
                singer_id_from_parent=True,
            )

        self.assertEqual(
            [row["recording_id"] for row in rows],
            ["app:raw_singer_001_take_01", "app:raw_singer_002_take_01"],
        )

    def test_prepare_app_recordings_prefills_rows_from_collection_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_dir = root / "gt_singer_grader" / "data" / "app_recordings" / "raw" / "planned_singer_001"
            audio_dir.mkdir(parents=True)
            (audio_dir / "vibrato_0001.wav").write_bytes(b"")
            plan_csv = root / "collection_plan.csv"
            write_plan_csv(
                plan_csv,
                [
                    {
                        "plan_id": "app_collection:0001",
                        "singer_id": "planned_singer_001",
                        "intended_family": "vibrato",
                        "suggested_filename": "raw/planned_singer_001/vibrato_0001.wav",
                        "review_goal": "collect clear vibrato technique",
                        "minimum_review_strength": "present",
                        "notes": "5-10 seconds",
                    }
                ],
            )

            rows = build_review_rows(
                discover_wavs(root / "gt_singer_grader" / "data" / "app_recordings" / "raw"),
                relative_to=root / "gt_singer_grader" / "data" / "app_recordings",
                recording_prefix="app",
                label_source="coach_review",
                reviewer_id="reviewer_a",
                song_id="warmup",
                intended_family="",
                singer_id_from_parent=False,
                collection_plan_rows=read_collection_plan(plan_csv),
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["audio_path"], "raw/planned_singer_001/vibrato_0001.wav")
        self.assertEqual(rows[0]["singer_id"], "planned_singer_001")
        self.assertEqual(rows[0]["split_group"], "planned_singer_001")
        self.assertEqual(rows[0]["intended_family"], "vibrato")
        self.assertIn("app_collection:0001", rows[0]["notes"])
        self.assertIn("collect clear vibrato technique", rows[0]["notes"])

    def test_prepare_app_recordings_reports_collection_plan_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "app_recordings" / "raw"
            planned_dir = raw_dir / "planned_singer_001"
            extra_dir = raw_dir / "extra_singer"
            planned_dir.mkdir(parents=True)
            extra_dir.mkdir(parents=True)
            (planned_dir / "vibrato_0001.wav").write_bytes(b"")
            (extra_dir / "extra_take.wav").write_bytes(b"")
            plan_csv = root / "collection_plan.csv"
            write_plan_csv(
                plan_csv,
                [
                    {
                        "plan_id": "app_collection:0001",
                        "singer_id": "planned_singer_001",
                        "intended_family": "vibrato",
                        "suggested_filename": "raw/planned_singer_001/vibrato_0001.wav",
                        "review_goal": "collect clear vibrato technique",
                        "minimum_review_strength": "present",
                        "notes": "",
                    },
                    {
                        "plan_id": "app_collection:0002",
                        "singer_id": "planned_singer_002",
                        "intended_family": "breathy",
                        "suggested_filename": "raw/planned_singer_002/breathy_0002.wav",
                        "review_goal": "collect clear breathy technique",
                        "minimum_review_strength": "present",
                        "notes": "",
                    },
                ],
            )

            report = collection_plan_match_report(
                discover_wavs(raw_dir),
                relative_to=root / "app_recordings",
                collection_plan_rows=read_collection_plan(plan_csv),
            )

        self.assertFalse(report["collection_plan_fully_matched"])
        self.assertEqual(report["collection_plan_matches"], 1)
        self.assertEqual(report["unplanned_audio_paths"], ["raw/extra_singer/extra_take.wav"])
        self.assertEqual(
            report["missing_collection_plan_suggestions"],
            ["raw/planned_singer_002/breathy_0002.wav"],
        )

    def test_prepare_app_recordings_strict_collection_plan_rejects_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            singer_dir = raw_dir / "planned_singer_001"
            singer_dir.mkdir(parents=True)
            (singer_dir / "vibrato_0001.wav").write_bytes(b"")
            plan_csv = root / "collection_plan.csv"
            write_plan_csv(
                plan_csv,
                [
                    {
                        "plan_id": "app_collection:0001",
                        "singer_id": "planned_singer_001",
                        "intended_family": "vibrato",
                        "suggested_filename": "raw/planned_singer_001/vibrato_0001.wav",
                        "review_goal": "collect clear vibrato technique",
                        "minimum_review_strength": "present",
                        "notes": "",
                    },
                    {
                        "plan_id": "app_collection:0002",
                        "singer_id": "planned_singer_002",
                        "intended_family": "control",
                        "suggested_filename": "raw/planned_singer_002/control_0002.wav",
                        "review_goal": "collect ordinary singing",
                        "minimum_review_strength": "absent",
                        "notes": "",
                    },
                ],
            )
            args = Namespace(
                audio_dir=str(raw_dir),
                output=str(root / "review_labels.csv"),
                report_json=None,
                collection_plan=str(plan_csv),
                strict_collection_plan=True,
                relative_to=str(root),
                recording_prefix="app",
                label_source="coach_review",
                reviewer_id="",
                song_id="",
                intended_family="",
                singer_id_from_parent=True,
                force=False,
                allow_empty=False,
            )

            with self.assertRaises(SystemExit) as exc:
                build_prepare_app_recordings_report(args)

        self.assertIn("collection plan mismatch", str(exc.exception))
        self.assertFalse(Path(args.output).exists())

    def test_experiment_status_reads_app_prepare_report_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "prepare_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "records": 2,
                        "collection_plan_rows": 3,
                        "collection_plan_matches": 1,
                        "collection_plan_fully_matched": False,
                        "missing_collection_plan_suggestions": ["raw/singer_002/breathy.wav"],
                        "unplanned_audio_paths": ["raw/extra/take.wav"],
                    }
                ),
                encoding="utf-8",
            )

            status = app_prepare_report_status(str(report_path))

        self.assertTrue(status["exists"])
        self.assertFalse(status["collection_plan_fully_matched"])
        self.assertEqual(status["collection_plan_matches"], 1)
        self.assertEqual(status["missing_collection_plan_suggestions"], ["raw/singer_002/breathy.wav"])
        self.assertEqual(status["unplanned_audio_paths"], ["raw/extra/take.wav"])

    def test_prepare_app_recordings_refuses_to_overwrite_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "review_labels.csv"
            output.write_text("existing\n", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                write_review_csv(output, [])

    def test_prepare_app_recordings_rejects_empty_audio_dir_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = Namespace(
                audio_dir=str(root / "raw"),
                output=str(root / "review_labels.csv"),
                report_json=None,
                relative_to=str(root),
                recording_prefix="app",
                label_source="coach_review",
                reviewer_id="",
                song_id="",
                intended_family="",
                collection_plan=None,
                strict_collection_plan=False,
                singer_id_from_parent=True,
                force=False,
                allow_empty=False,
            )
            Path(args.audio_dir).mkdir()

            with self.assertRaises(ValueError):
                build_prepare_app_recordings_report(args)

    def test_prepare_app_recordings_can_write_header_only_csv_when_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = Namespace(
                audio_dir=str(root / "raw"),
                output=str(root / "review_labels.csv"),
                report_json=None,
                relative_to=str(root),
                recording_prefix="app",
                label_source="coach_review",
                reviewer_id="",
                song_id="",
                intended_family="",
                collection_plan=None,
                strict_collection_plan=False,
                singer_id_from_parent=True,
                force=False,
                allow_empty=True,
            )
            Path(args.audio_dir).mkdir()

            report = build_prepare_app_recordings_report(args)

            self.assertEqual(report["records"], 0)
            self.assertTrue(Path(args.output).is_file())

    def test_app_label_coverage_reports_collection_shortfalls_from_review_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "review.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "audio_path",
                        "recording_id",
                        "singer_id",
                        "song_id",
                        "intended_family",
                        "mix",
                        "falsetto",
                        "breathy",
                        "pharyngeal",
                        "glissando",
                        "vibrato",
                        "families",
                        "techniques",
                        "split_group",
                        "label_source",
                        "reviewer_id",
                        "notes",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "audio_path": "raw/singer_001/breathy.wav",
                        "recording_id": "app:breathy-001",
                        "singer_id": "singer_001",
                        "song_id": "warmup",
                        "intended_family": "breathy",
                        "mix": "absent",
                        "falsetto": "absent",
                        "breathy": "present",
                        "pharyngeal": "absent",
                        "glissando": "absent",
                        "vibrato": "absent",
                        "families": "",
                        "techniques": "",
                        "split_group": "singer_001",
                        "label_source": "coach_review",
                        "reviewer_id": "reviewer_a",
                        "notes": "",
                    }
                )
                writer.writerow(
                    {
                        "audio_path": "raw/singer_002/control.wav",
                        "recording_id": "app:control-001",
                        "singer_id": "singer_002",
                        "song_id": "warmup",
                        "intended_family": "control",
                        "mix": "absent",
                        "falsetto": "absent",
                        "breathy": "absent",
                        "pharyngeal": "absent",
                        "glissando": "absent",
                        "vibrato": "absent",
                        "families": "",
                        "techniques": "",
                        "split_group": "singer_002",
                        "label_source": "coach_review",
                        "reviewer_id": "reviewer_a",
                        "notes": "",
                    }
                )
                writer.writerow(
                    {
                        "audio_path": "raw/singer_003/unreviewed.wav",
                        "recording_id": "app:unreviewed-001",
                        "singer_id": "singer_003",
                        "song_id": "warmup",
                        "intended_family": "vibrato",
                        "mix": "",
                        "falsetto": "",
                        "breathy": "",
                        "pharyngeal": "",
                        "glissando": "",
                        "vibrato": "",
                        "families": "",
                        "techniques": "",
                        "split_group": "singer_003",
                        "label_source": "coach_review",
                        "reviewer_id": "reviewer_a",
                        "notes": "",
                    }
                )

            report = build_app_label_coverage_report(
                str(csv_path),
                target_families=["breathy", "vibrato"],
                min_per_family=2,
                min_negative=2,
                min_groups=3,
            )

        self.assertFalse(report["ready_for_collection_target"])
        self.assertEqual(report["target_families"]["breathy"], 1)
        self.assertEqual(report["target_families"]["vibrato"], 0)
        self.assertEqual(report["missing_target_families"], {"breathy": 1, "vibrato": 2})
        self.assertEqual(report["negative_shortfall"], 1)
        self.assertEqual(report["group_shortfall"], 0)
        self.assertEqual(report["unlabeled_records"], 1)
        self.assertEqual(report["review_progress"]["records"], 3)
        self.assertEqual(report["review_progress"]["labeled_records"], 2)
        self.assertEqual(report["review_progress"]["unlabeled_records"], 1)
        self.assertEqual(report["review_progress"]["by_intended_family"]["vibrato"]["unlabeled_records"], 1)
        self.assertEqual(report["review_progress"]["by_split_group"]["singer_003"]["unlabeled_records"], 1)
        self.assertIn("review CSV has unlabeled rows", report["warnings"])

    def test_app_label_coverage_can_require_existing_audio_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_dir = root / "raw" / "singer_001"
            audio_dir.mkdir(parents=True)
            (audio_dir / "present.wav").write_bytes(b"wav")
            csv_path = root / "review.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "audio_path",
                        "recording_id",
                        "singer_id",
                        "song_id",
                        "intended_family",
                        "mix",
                        "falsetto",
                        "breathy",
                        "pharyngeal",
                        "glissando",
                        "vibrato",
                        "families",
                        "techniques",
                        "split_group",
                        "label_source",
                        "reviewer_id",
                        "notes",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "audio_path": "raw/singer_001/present.wav",
                        "recording_id": "app:present",
                        "singer_id": "singer_001",
                        "song_id": "warmup",
                        "intended_family": "vibrato",
                        "mix": "absent",
                        "falsetto": "absent",
                        "breathy": "absent",
                        "pharyngeal": "absent",
                        "glissando": "absent",
                        "vibrato": "present",
                        "families": "",
                        "techniques": "",
                        "split_group": "singer_001",
                        "label_source": "coach_review",
                        "reviewer_id": "reviewer_a",
                        "notes": "",
                    }
                )
                writer.writerow(
                    {
                        "audio_path": "raw/singer_002/missing.wav",
                        "recording_id": "app:missing",
                        "singer_id": "singer_002",
                        "song_id": "warmup",
                        "intended_family": "vibrato",
                        "mix": "absent",
                        "falsetto": "absent",
                        "breathy": "absent",
                        "pharyngeal": "absent",
                        "glissando": "absent",
                        "vibrato": "present",
                        "families": "",
                        "techniques": "",
                        "split_group": "singer_002",
                        "label_source": "coach_review",
                        "reviewer_id": "reviewer_a",
                        "notes": "",
                    }
                )

            report = build_app_label_coverage_report(
                str(csv_path),
                target_families=["vibrato"],
                min_per_family=2,
                min_negative=0,
                min_groups=2,
                require_audio_files=True,
                audio_root=root,
            )

        self.assertFalse(report["ready_for_collection_target"])
        self.assertTrue(report["audio_files_checked"])
        self.assertEqual(report["missing_audio_file_count"], 1)
        self.assertEqual(report["missing_audio_files"], ["raw/singer_002/missing.wav"])
        self.assertIn("review CSV references missing audio files", report["warnings"])

    def test_app_label_coverage_reports_duplicate_review_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "review.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "audio_path",
                        "recording_id",
                        "singer_id",
                        "song_id",
                        "intended_family",
                        "mix",
                        "falsetto",
                        "breathy",
                        "pharyngeal",
                        "glissando",
                        "vibrato",
                        "families",
                        "techniques",
                        "split_group",
                        "label_source",
                        "reviewer_id",
                        "notes",
                    ],
                )
                writer.writeheader()
                base_row = {
                    "audio_path": "raw/singer_001/vibrato.wav",
                    "recording_id": "app:vibrato-001",
                    "singer_id": "singer_001",
                    "song_id": "warmup",
                    "intended_family": "vibrato",
                    "mix": "absent",
                    "falsetto": "absent",
                    "breathy": "absent",
                    "pharyngeal": "absent",
                    "glissando": "absent",
                    "vibrato": "present",
                    "families": "",
                    "techniques": "",
                    "split_group": "singer_001",
                    "label_source": "coach_review",
                    "reviewer_id": "reviewer_a",
                    "notes": "",
                }
                writer.writerow(base_row)
                writer.writerow({**base_row, "singer_id": "singer_002", "split_group": "singer_002"})

            report = build_app_label_coverage_report(
                str(csv_path),
                target_families=["vibrato"],
                min_per_family=1,
                min_negative=0,
                min_groups=1,
            )

        self.assertFalse(report["ready_for_collection_target"])
        self.assertEqual(report["duplicate_audio_paths"], ["raw/singer_001/vibrato.wav"])
        self.assertEqual(report["duplicate_recording_ids"], ["app:vibrato-001"])
        self.assertIn("review CSV has duplicate audio_path values", report["warnings"])
        self.assertIn("review CSV has duplicate recording_id values", report["warnings"])

    def test_app_label_coverage_reports_intended_family_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "review.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "audio_path",
                        "recording_id",
                        "singer_id",
                        "song_id",
                        "intended_family",
                        "mix",
                        "falsetto",
                        "breathy",
                        "pharyngeal",
                        "glissando",
                        "vibrato",
                        "families",
                        "techniques",
                        "split_group",
                        "label_source",
                        "reviewer_id",
                        "notes",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "audio_path": "raw/singer_001/vibrato.wav",
                        "recording_id": "app:vibrato-prompt",
                        "singer_id": "singer_001",
                        "song_id": "warmup",
                        "intended_family": "vibrato",
                        "mix": "absent",
                        "falsetto": "absent",
                        "breathy": "present",
                        "pharyngeal": "absent",
                        "glissando": "absent",
                        "vibrato": "absent",
                        "families": "",
                        "techniques": "",
                        "split_group": "singer_001",
                        "label_source": "coach_review",
                        "reviewer_id": "reviewer_a",
                        "notes": "",
                    }
                )
                writer.writerow(
                    {
                        "audio_path": "raw/singer_002/control.wav",
                        "recording_id": "app:control-prompt",
                        "singer_id": "singer_002",
                        "song_id": "warmup",
                        "intended_family": "control",
                        "mix": "absent",
                        "falsetto": "absent",
                        "breathy": "absent",
                        "pharyngeal": "absent",
                        "glissando": "absent",
                        "vibrato": "present",
                        "families": "",
                        "techniques": "",
                        "split_group": "singer_002",
                        "label_source": "coach_review",
                        "reviewer_id": "reviewer_a",
                        "notes": "",
                    }
                )

            report = build_app_label_coverage_report(
                str(csv_path),
                target_families=["breathy", "vibrato"],
                min_per_family=1,
                min_negative=0,
                min_groups=2,
            )

        self.assertFalse(report["ready_for_collection_target"])
        self.assertEqual(report["intended_family_mismatch_count"], 2)
        self.assertEqual(
            [item["recording_id"] for item in report["intended_family_mismatches"]],
            ["app:vibrato-prompt", "app:control-prompt"],
        )
        self.assertIn("review CSV has intended_family/reviewer-label mismatches", report["warnings"])

    def test_app_label_coverage_reports_labeled_rows_without_reviewer_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "review.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "audio_path",
                        "recording_id",
                        "singer_id",
                        "song_id",
                        "intended_family",
                        "mix",
                        "falsetto",
                        "breathy",
                        "pharyngeal",
                        "glissando",
                        "vibrato",
                        "families",
                        "techniques",
                        "split_group",
                        "label_source",
                        "reviewer_id",
                        "notes",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "audio_path": "raw/singer_001/vibrato.wav",
                        "recording_id": "app:vibrato-001",
                        "singer_id": "singer_001",
                        "song_id": "warmup",
                        "intended_family": "vibrato",
                        "mix": "absent",
                        "falsetto": "absent",
                        "breathy": "absent",
                        "pharyngeal": "absent",
                        "glissando": "absent",
                        "vibrato": "present",
                        "families": "",
                        "techniques": "",
                        "split_group": "singer_001",
                        "label_source": "coach_review",
                        "reviewer_id": "",
                        "notes": "",
                    }
                )
                writer.writerow(
                    {
                        "audio_path": "raw/singer_002/unreviewed.wav",
                        "recording_id": "app:unreviewed-001",
                        "singer_id": "singer_002",
                        "song_id": "warmup",
                        "intended_family": "breathy",
                        "mix": "",
                        "falsetto": "",
                        "breathy": "",
                        "pharyngeal": "",
                        "glissando": "",
                        "vibrato": "",
                        "families": "",
                        "techniques": "",
                        "split_group": "singer_002",
                        "label_source": "coach_review",
                        "reviewer_id": "",
                        "notes": "",
                    }
                )

            report = build_app_label_coverage_report(
                str(csv_path),
                target_families=["vibrato"],
                min_per_family=1,
                min_negative=0,
                min_groups=1,
            )

        self.assertFalse(report["ready_for_collection_target"])
        self.assertEqual(report["missing_reviewer_id_count"], 1)
        self.assertEqual(report["missing_reviewer_ids"][0]["recording_id"], "app:vibrato-001")
        self.assertEqual(report["review_progress"]["missing_reviewer_id_records"], 1)
        self.assertEqual(report["review_progress"]["by_intended_family"]["vibrato"]["missing_reviewer_id_records"], 1)
        self.assertIn("review CSV has labeled rows without reviewer_id", report["warnings"])

    def test_app_collection_plan_starts_from_zero_when_review_csv_is_missing(self) -> None:
        report = build_collection_plan(
            csv_path="/tmp/nanopitch_missing_review_labels.csv",
            target_families=["breathy", "vibrato"],
            min_per_family=2,
            min_negative=1,
            min_groups=4,
            singer_prefix="singer",
        )

        self.assertFalse(report["ready_for_collection_target"])
        self.assertEqual(report["needed"]["target_families"], {"breathy": 2, "vibrato": 2})
        self.assertEqual(report["needed"]["negative"], 1)
        self.assertEqual(report["planned_records"], 5)
        self.assertEqual(report["planned_groups"], 4)
        self.assertEqual(report["plan_rows"][0]["singer_id"], "singer_001")
        self.assertEqual(report["plan_rows"][0]["intended_family"], "breathy")
        self.assertEqual(
            sorted(row["intended_family"] for row in report["plan_rows"]),
            ["breathy", "breathy", "control", "vibrato", "vibrato"],
        )
        self.assertEqual(
            {row["singer_id"] for row in report["plan_rows"]},
            {"singer_001", "singer_002", "singer_003", "singer_004"},
        )

    def test_app_collection_plan_can_limit_takes_per_singer_group(self) -> None:
        report = build_collection_plan(
            csv_path="/tmp/nanopitch_missing_review_labels.csv",
            target_families=["breathy", "vibrato"],
            min_per_family=2,
            min_negative=2,
            min_groups=3,
            singer_prefix="singer",
            clips_per_singer=2,
        )

        self.assertEqual(report["planned_records"], 6)
        self.assertEqual(report["planned_groups"], 3)
        self.assertEqual(
            [row["singer_id"] for row in report["plan_rows"]],
            ["singer_001", "singer_001", "singer_002", "singer_002", "singer_003", "singer_003"],
        )
        self.assertEqual(
            [row["intended_family"] for row in report["plan_rows"]],
            ["breathy", "vibrato", "control", "breathy", "vibrato", "control"],
        )

    def test_experiment_status_summarizes_app_collection_plan_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_csv = root / "collection_plan.csv"
            plan_json = root / "collection_plan.json"
            report = build_collection_plan(
                csv_path=str(root / "missing_review_labels.csv"),
                target_families=["breathy", "vibrato"],
                min_per_family=2,
                min_negative=2,
                min_groups=2,
                clips_per_singer=3,
            )
            write_plan_csv(plan_csv, report["plan_rows"])
            plan_json.write_text(json.dumps(report) + "\n", encoding="utf-8")

            status = app_collection_plan_status(str(plan_csv), str(plan_json))

        self.assertTrue(status["exists"])
        self.assertTrue(status["json_exists"])
        self.assertEqual(status["planned_records"], 6)
        self.assertEqual(status["planned_groups"], 2)
        self.assertEqual(status["intended_family_counts"], {"breathy": 2, "control": 2, "vibrato": 2})
        self.assertEqual(status["thresholds"]["clips_per_singer"], 3)

    def test_materialize_app_collection_creates_directories_and_checklist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_csv = root / "collection_plan.csv"
            app_root = root / "app_recordings"
            report = build_collection_plan(
                csv_path=str(root / "missing_review_labels.csv"),
                target_families=["vibrato"],
                min_per_family=1,
                min_negative=1,
                min_groups=2,
                singer_prefix="singer",
            )
            write_plan_csv(plan_csv, report["plan_rows"])

            materialized = materialize_collection(
                read_materialized_plan(plan_csv),
                root=app_root,
                checklist_path=app_root / "collection_checklist.csv",
                missing_csv_path=app_root / "collection_missing.csv",
            )
            with (app_root / "collection_checklist.csv").open("r", encoding="utf-8") as handle:
                checklist_rows = list(csv.DictReader(handle))
            with (app_root / "collection_missing.csv").open("r", encoding="utf-8") as handle:
                missing_rows = list(csv.DictReader(handle))

            self.assertTrue((app_root / "raw" / "singer_001").is_dir())
            self.assertTrue((app_root / "raw" / "singer_002").is_dir())
            self.assertEqual(materialized["missing_csv"], str(app_root / "collection_missing.csv"))
            self.assertEqual(materialized["planned_records"], 2)
            self.assertEqual(materialized["planned_groups"], 2)
            self.assertEqual(materialized["created_directories"], 2)
            self.assertEqual(materialized["missing_audio_files"], 2)
            self.assertEqual(len(materialized["missing_audio_paths"]), 2)
            self.assertEqual(materialized["existing_audio_paths"], [])
            self.assertEqual(materialized["missing_by_family"], {"control": 1, "vibrato": 1})
            self.assertEqual(materialized["missing_by_singer"], {"singer_001": 1, "singer_002": 1})
            self.assertFalse(materialized["ready_for_review_csv"])
            self.assertEqual(materialized["intended_family_counts"], {"control": 1, "vibrato": 1})
            self.assertEqual(checklist_rows[0]["exists"], "no")
            self.assertEqual(len(missing_rows), 2)
            self.assertTrue(missing_rows[0]["expected_audio_path"].endswith("raw/singer_001/vibrato_0001.wav"))
            self.assertTrue(checklist_rows[0]["expected_audio_path"].endswith("raw/singer_001/vibrato_0001.wav"))

    def test_materialize_app_collection_reports_ready_after_all_audio_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_csv = root / "collection_plan.csv"
            app_root = root / "app_recordings"
            report = build_collection_plan(
                csv_path=str(root / "missing_review_labels.csv"),
                target_families=["vibrato"],
                min_per_family=1,
                min_negative=1,
                min_groups=2,
                singer_prefix="singer",
            )
            write_plan_csv(plan_csv, report["plan_rows"])
            for row in report["plan_rows"]:
                audio_path = app_root / row["suggested_filename"]
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                audio_path.write_bytes(b"wav")

            materialized = materialize_collection(
                read_materialized_plan(plan_csv),
                root=app_root,
                checklist_path=app_root / "collection_checklist.csv",
                missing_csv_path=app_root / "collection_missing.csv",
            )
            with (app_root / "collection_missing.csv").open("r", encoding="utf-8") as handle:
                missing_rows = list(csv.DictReader(handle))

        self.assertEqual(materialized["existing_audio_files"], 2)
        self.assertEqual(materialized["missing_audio_files"], 0)
        self.assertEqual(len(materialized["existing_audio_paths"]), 2)
        self.assertEqual(materialized["missing_audio_paths"], [])
        self.assertEqual(materialized["missing_by_family"], {})
        self.assertEqual(materialized["missing_by_singer"], {})
        self.assertEqual(missing_rows, [])
        self.assertTrue(materialized["ready_for_review_csv"])

    def test_materialize_app_collection_can_validate_existing_wav_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_root = root / "app_recordings"
            valid_path = app_root / "raw" / "singer_001" / "vibrato.wav"
            invalid_path = app_root / "raw" / "singer_001" / "control.wav"
            short_path = app_root / "raw" / "singer_001" / "breathy.wav"
            write_tiny_wav(valid_path, seconds=6.0)
            write_tiny_wav(short_path, seconds=1.0)
            invalid_path.parent.mkdir(parents=True, exist_ok=True)
            invalid_path.write_bytes(b"not a wav")

            materialized = materialize_collection(
                [
                    {
                        "plan_id": "app_collection:0001",
                        "singer_id": "singer_001",
                        "intended_family": "vibrato",
                        "suggested_filename": "raw/singer_001/vibrato.wav",
                    },
                    {
                        "plan_id": "app_collection:0002",
                        "singer_id": "singer_001",
                        "intended_family": "control",
                        "suggested_filename": "raw/singer_001/control.wav",
                    },
                    {
                        "plan_id": "app_collection:0003",
                        "singer_id": "singer_001",
                        "intended_family": "breathy",
                        "suggested_filename": "raw/singer_001/breathy.wav",
                    },
                ],
                root=app_root,
                validate_wav_files=True,
            )

        self.assertEqual(materialized["existing_audio_files"], 3)
        self.assertEqual(materialized["valid_audio_files"], 1)
        self.assertEqual(materialized["invalid_audio_files"], 2)
        self.assertEqual(materialized["invalid_audio_paths"], [str(short_path), str(invalid_path)])
        self.assertEqual(materialized["invalid_audio_reasons"][str(invalid_path)], "unreadable_wav")
        self.assertTrue(materialized["invalid_audio_reasons"][str(short_path)].startswith("too_short:"))
        self.assertEqual(materialized["wav_duration_bounds_seconds"], {"min": 5.0, "max": 10.0})
        self.assertTrue(materialized["wav_validation_enabled"])
        self.assertFalse(materialized["ready_for_review_csv"])

    def test_experiment_status_summarizes_app_collection_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_csv = root / "collection_plan.csv"
            app_root = root / "app_recordings"
            report = build_collection_plan(
                csv_path=str(root / "missing_review_labels.csv"),
                target_families=["vibrato"],
                min_per_family=1,
                min_negative=1,
                min_groups=2,
                singer_prefix="singer",
            )
            write_plan_csv(plan_csv, report["plan_rows"])
            materialized = materialize_collection(
                read_materialized_plan(plan_csv),
                root=app_root,
                checklist_path=app_root / "collection_checklist.csv",
            )
            materialize_report = app_root / "collection_materialize_report.json"
            materialize_report.write_text(json.dumps(materialized) + "\n", encoding="utf-8")

            status = app_collection_materialize_status(
                str(app_root / "collection_checklist.csv"),
                str(materialize_report),
                str(app_root / "collection_missing.csv"),
            )

        self.assertTrue(status["checklist_exists"])
        self.assertTrue(status["report_exists"])
        self.assertTrue(status["ok"])
        self.assertFalse(status["ready_for_review_csv"])
        self.assertEqual(status["planned_records"], 2)
        self.assertEqual(status["planned_groups"], 2)
        self.assertEqual(status["missing_audio_files"], 2)
        self.assertEqual(status["checklist_missing_audio_files"], 2)
        self.assertTrue(status["missing_csv_exists"])
        self.assertEqual(status["missing_csv_records"], 2)
        self.assertEqual(len(status["missing_audio_paths"]), 2)
        self.assertEqual(status["missing_csv"], str(app_root / "collection_missing.csv"))
        self.assertEqual(status["missing_by_family"], {"control": 1, "vibrato": 1})
        self.assertEqual(status["missing_by_singer"], {"singer_001": 1, "singer_002": 1})

    def test_experiment_status_summarizes_app_collection_packet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet_dir = root / "packet"
            checklist = root / "collection_checklist.csv"
            checklist.write_text("header\n", encoding="utf-8")
            report = export_packet(
                [
                    {
                        "singer_id": "singer_001",
                        "intended_family": "vibrato",
                        "expected_audio_path": "raw/singer_001/vibrato_0001.wav",
                        "exists": "no",
                    }
                ],
                output_dir=packet_dir,
                source_checklist=checklist,
            )
            summary = root / "packet_summary.json"
            summary.write_text(json.dumps(report) + "\n", encoding="utf-8")

            status = app_collection_packet_status(str(packet_dir), str(summary), str(checklist))

        self.assertTrue(status["exists"])
        self.assertTrue(status["summary_exists"])
        self.assertTrue(status["index_exists"])
        self.assertTrue(status["ok"])
        self.assertTrue(status["checklist_match"])
        self.assertEqual(status["source_checklist"]["path"], str(checklist))
        self.assertEqual(status["sheet_count"], 1)
        self.assertEqual(status["planned_records"], 1)
        self.assertEqual(status["planned_singers"], 1)

    def test_export_app_collection_packet_removes_stale_sheets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "packet"
            output_dir.mkdir()
            stale = output_dir / "stale_singer.md"
            stale.write_text("old sheet\n", encoding="utf-8")
            (output_dir / "index.md").write_text("old index\n", encoding="utf-8")
            rows = [
                {
                    "singer_id": "singer_001",
                    "intended_family": "vibrato",
                    "expected_audio_path": "raw/singer_001/vibrato.wav",
                    "exists": "no",
                    "review_goal": "collect vibrato",
                }
            ]

            report = export_packet(rows, output_dir=output_dir)
            index_exists = (output_dir / "index.md").is_file()

        self.assertFalse(stale.exists())
        self.assertEqual(report["removed_sheet_paths"], [str(stale)])
        self.assertEqual(report["sheets"], [str(output_dir / "singer_001.md")])
        self.assertTrue(index_exists)

    def test_experiment_status_reports_stale_app_collection_packet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet_dir = root / "packet"
            checklist = root / "collection_checklist.csv"
            checklist.write_text("old\n", encoding="utf-8")
            report = export_packet(
                [
                    {
                        "singer_id": "singer_001",
                        "intended_family": "vibrato",
                        "expected_audio_path": "raw/singer_001/vibrato_0001.wav",
                    }
                ],
                output_dir=packet_dir,
                source_checklist=checklist,
            )
            summary = root / "packet_summary.json"
            summary.write_text(json.dumps(report) + "\n", encoding="utf-8")
            checklist.write_text("new\n", encoding="utf-8")

            status = app_collection_packet_status(str(packet_dir), str(summary), str(checklist))

        self.assertFalse(status["ok"])
        self.assertFalse(status["checklist_match"])
        self.assertNotEqual(
            status["source_checklist"]["sha256"],
            status["current_checklist"]["sha256"],
        )

    def test_experiment_status_reports_missing_app_collection_packet_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet_dir = root / "packet"
            checklist = root / "collection_checklist.csv"
            checklist.write_text("header\n", encoding="utf-8")
            report = export_packet(
                [
                    {
                        "singer_id": "singer_001",
                        "intended_family": "vibrato",
                        "expected_audio_path": "raw/singer_001/vibrato_0001.wav",
                    }
                ],
                output_dir=packet_dir,
                source_checklist=checklist,
            )
            summary = root / "packet_summary.json"
            summary.write_text(json.dumps(report) + "\n", encoding="utf-8")
            (packet_dir / "index.md").unlink()

            status = app_collection_packet_status(str(packet_dir), str(summary), str(checklist))

        self.assertFalse(status["ok"])
        self.assertFalse(status["index_exists"])
        self.assertTrue(status["checklist_match"])

    def test_materialize_app_collection_reports_duplicate_suggested_filenames(self) -> None:
        report = materialize_collection(
            [
                {
                    "plan_id": "app_collection:0001",
                    "singer_id": "singer_001",
                    "intended_family": "vibrato",
                    "suggested_filename": "raw/singer_001/take.wav",
                },
                {
                    "plan_id": "app_collection:0002",
                    "singer_id": "singer_001",
                    "intended_family": "control",
                    "suggested_filename": "raw/singer_001/take.wav",
                },
            ],
            root="/tmp/nanopitch-materialize-duplicates",
        )

        self.assertFalse(report["ok"])
        self.assertEqual(report["duplicate_suggested_filenames"], ["raw/singer_001/take.wav"])

    def test_export_app_collection_packet_writes_per_singer_sheets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet_dir = root / "packet"

            report = export_packet(
                [
                    {
                        "singer_id": "singer_001",
                        "intended_family": "vibrato",
                        "expected_audio_path": "raw/singer_001/vibrato_0001.wav",
                        "exists": "no",
                        "review_goal": "collect clear vibrato technique",
                    },
                    {
                        "singer_id": "singer_001",
                        "intended_family": "control",
                        "expected_audio_path": "raw/singer_001/control_0002.wav",
                        "exists": "yes",
                        "review_goal": "collect ordinary singing with no target technique",
                    },
                    {
                        "singer_id": "singer_002",
                        "intended_family": "breathy",
                        "expected_audio_path": "raw/singer_002/breathy_0003.wav",
                        "exists": "no",
                        "review_goal": "collect clear breathy technique",
                    },
                ],
                output_dir=packet_dir,
                source_checklist=root / "collection_checklist.csv",
            )

            singer_001 = (packet_dir / "singer_001.md").read_text(encoding="utf-8")
            index = (packet_dir / "index.md").read_text(encoding="utf-8")

        self.assertTrue(report["ok"])
        self.assertEqual(report["planned_records"], 3)
        self.assertEqual(report["planned_singers"], 2)
        self.assertEqual(report["existing_audio_files"], 1)
        self.assertEqual(report["missing_audio_files"], 2)
        self.assertTrue(report["index_path"].endswith("index.md"))
        self.assertEqual(report["intended_family_counts"], {"breathy": 1, "control": 1, "vibrato": 1})
        self.assertIn("raw/singer_001/vibrato_0001.wav", singer_001)
        self.assertIn("collect ordinary singing", singer_001)
        self.assertIn("Planned clips: 3", index)
        self.assertIn("[singer_001.md](singer_001.md)", index)

    def test_export_app_collection_packet_reports_duplicate_audio_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = export_packet(
                [
                    {
                        "singer_id": "singer_001",
                        "intended_family": "vibrato",
                        "expected_audio_path": "raw/singer_001/take.wav",
                    },
                    {
                        "singer_id": "singer_001",
                        "intended_family": "control",
                        "expected_audio_path": "raw/singer_001/take.wav",
                    },
                ],
                output_dir=Path(tmp) / "packet",
            )

        self.assertFalse(report["ok"])
        self.assertEqual(report["duplicate_audio_paths"], ["raw/singer_001/take.wav"])

    def test_app_collection_plan_uses_existing_review_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "review.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "audio_path",
                        "recording_id",
                        "singer_id",
                        "song_id",
                        "intended_family",
                        "mix",
                        "falsetto",
                        "breathy",
                        "pharyngeal",
                        "glissando",
                        "vibrato",
                        "families",
                        "techniques",
                        "split_group",
                        "label_source",
                        "reviewer_id",
                        "notes",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "audio_path": "raw/singer_001/vibrato.wav",
                        "recording_id": "app:vibrato-001",
                        "singer_id": "singer_001",
                        "song_id": "warmup",
                        "intended_family": "vibrato",
                        "mix": "absent",
                        "falsetto": "absent",
                        "breathy": "absent",
                        "pharyngeal": "absent",
                        "glissando": "absent",
                        "vibrato": "present",
                        "families": "",
                        "techniques": "",
                        "split_group": "singer_001",
                        "label_source": "coach_review",
                        "reviewer_id": "reviewer_a",
                        "notes": "",
                    }
                )
                writer.writerow(
                    {
                        "audio_path": "raw/singer_002/control.wav",
                        "recording_id": "app:control-001",
                        "singer_id": "singer_002",
                        "song_id": "warmup",
                        "intended_family": "control",
                        "mix": "absent",
                        "falsetto": "absent",
                        "breathy": "absent",
                        "pharyngeal": "absent",
                        "glissando": "absent",
                        "vibrato": "absent",
                        "families": "",
                        "techniques": "",
                        "split_group": "singer_002",
                        "label_source": "coach_review",
                        "reviewer_id": "reviewer_a",
                        "notes": "",
                    }
                )
            output_csv = root / "plan.csv"

            report = build_collection_plan(
                csv_path=str(csv_path),
                target_families=["vibrato"],
                min_per_family=2,
                min_negative=2,
                min_groups=4,
                singer_prefix="next",
                start_index=10,
            )
            write_plan_csv(output_csv, report["plan_rows"])
            with output_csv.open("r", encoding="utf-8") as handle:
                written = list(csv.DictReader(handle))

        self.assertEqual(report["current"]["target_families"], {"vibrato": 1})
        self.assertEqual(report["needed"]["target_families"], {"vibrato": 1})
        self.assertEqual(report["needed"]["negative"], 1)
        self.assertEqual(report["planned_records"], 2)
        self.assertEqual(written[0]["singer_id"], "next_010")
        self.assertEqual(written[0]["intended_family"], "vibrato")
        self.assertEqual(written[1]["singer_id"], "next_011")
        self.assertEqual(written[1]["intended_family"], "control")

    def test_app_validation_audit_reports_ready_coverage(self) -> None:
        records = []
        for index in range(2):
            records.append(
                {
                    "recording_id": f"app:vibrato-{index}",
                    "dataset": "app_recordings",
                    "audio_path": f"vibrato-{index}.wav",
                    "recording_domain": "app_user",
                    "label_source": "coach_review",
                    "split_group": f"singer_{index}",
                    "labels": {"families": ["vibrato"], "techniques": ["vibrato"]},
                }
            )
        records.append(
            {
                "recording_id": "app:control",
                "dataset": "app_recordings",
                "audio_path": "control.wav",
                "recording_domain": "app_user",
                "label_source": "coach_review",
                "split_group": "singer_control",
                "labels": {"families": ["control"], "techniques": []},
            }
        )

        report = audit_app_validation_records(
            records,
            target_families=["vibrato"],
            min_per_family=2,
            min_negative=1,
            min_groups=3,
        )

        self.assertTrue(report["ready_for_mvp_validation"])
        self.assertEqual(report["target_families"]["vibrato"], 2)
        self.assertEqual(report["negative_records"], 1)
        self.assertEqual(report["warnings"], [])

    def test_app_validation_audit_reports_shortfalls(self) -> None:
        records = [
            {
                "recording_id": "app:vibrato",
                "dataset": "app_recordings",
                "audio_path": "vibrato.wav",
                "recording_domain": "app_user",
                "label_source": "coach_review",
                "split_group": "singer_001",
                "labels": {"families": ["vibrato"], "techniques": ["vibrato"]},
            }
        ]

        report = audit_app_validation_records(
            records,
            target_families=["vibrato", "breathy"],
            min_per_family=2,
            min_negative=1,
            min_groups=2,
        )

        self.assertFalse(report["ready_for_mvp_validation"])
        self.assertEqual(report["missing_target_families"], {"breathy": 2, "vibrato": 1})
        self.assertEqual(report["negative_shortfall"], 1)
        self.assertEqual(report["group_shortfall"], 1)

    def test_app_recording_all_absent_strengths_become_control(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "review.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "audio_path",
                        "recording_id",
                        "singer_id",
                        "song_id",
                        "intended_family",
                        "mix",
                        "falsetto",
                        "breathy",
                        "pharyngeal",
                        "glissando",
                        "vibrato",
                        "families",
                        "techniques",
                        "split_group",
                        "label_source",
                        "reviewer_id",
                        "notes",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "audio_path": "data/app_recordings/control.wav",
                        "recording_id": "app:control-001",
                        "singer_id": "singer_003",
                        "song_id": "warmup",
                        "intended_family": "",
                        "mix": "absent",
                        "falsetto": "absent",
                        "breathy": "absent",
                        "pharyngeal": "absent",
                        "glissando": "absent",
                        "vibrato": "absent",
                        "families": "",
                        "techniques": "",
                        "split_group": "",
                        "label_source": "coach_review",
                        "reviewer_id": "reviewer_a",
                        "notes": "neutral take",
                    }
                )

            records = build_app_recordings_manifest(str(csv_path), "app_recordings", "app_user")

        self.assertEqual(records[0]["labels"]["families"], ["control"])
        self.assertEqual(records[0]["labels"]["techniques"], [])
        self.assertEqual(validate_record(records[0]), [])

    def test_app_recording_rejects_explicit_techniques_that_contradict_strengths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "review.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "audio_path",
                        "recording_id",
                        "singer_id",
                        "song_id",
                        "intended_family",
                        "mix",
                        "falsetto",
                        "breathy",
                        "pharyngeal",
                        "glissando",
                        "vibrato",
                        "families",
                        "techniques",
                        "split_group",
                        "label_source",
                        "reviewer_id",
                        "notes",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "audio_path": "data/app_recordings/conflict.wav",
                        "recording_id": "app:conflict-001",
                        "singer_id": "singer_004",
                        "song_id": "warmup",
                        "intended_family": "",
                        "mix": "absent",
                        "falsetto": "absent",
                        "breathy": "present",
                        "pharyngeal": "absent",
                        "glissando": "absent",
                        "vibrato": "absent",
                        "families": "breathy",
                        "techniques": "vibrato",
                        "split_group": "",
                        "label_source": "coach_review",
                        "reviewer_id": "reviewer_a",
                        "notes": "conflicting review",
                    }
                )

            with self.assertRaises(SystemExit) as exc:
                build_app_recordings_manifest(str(csv_path), "app_recordings", "app_user")

        self.assertIn("explicit techniques do not match", str(exc.exception))

    def test_app_recording_rejects_none_family_with_present_strength(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "review.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "audio_path",
                        "recording_id",
                        "singer_id",
                        "song_id",
                        "intended_family",
                        "mix",
                        "falsetto",
                        "breathy",
                        "pharyngeal",
                        "glissando",
                        "vibrato",
                        "families",
                        "techniques",
                        "split_group",
                        "label_source",
                        "reviewer_id",
                        "notes",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "audio_path": "data/app_recordings/conflict_none.wav",
                        "recording_id": "app:conflict-none-001",
                        "singer_id": "singer_005",
                        "song_id": "warmup",
                        "intended_family": "none",
                        "mix": "absent",
                        "falsetto": "absent",
                        "breathy": "absent",
                        "pharyngeal": "absent",
                        "glissando": "absent",
                        "vibrato": "present",
                        "families": "none",
                        "techniques": "",
                        "split_group": "",
                        "label_source": "coach_review",
                        "reviewer_id": "reviewer_a",
                        "notes": "conflicting none label",
                    }
                )

            with self.assertRaises(SystemExit) as exc:
                build_app_recordings_manifest(str(csv_path), "app_recordings", "app_user")

        self.assertIn("control/none/unclear family labels", str(exc.exception))

    def test_vocalset_manifest_maps_known_technique_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "VocalSet"
            breathy = root / "by_technique" / "breathy" / "female1" / "f1_breathy_a_scales.wav"
            belt = root / "by_technique" / "belt" / "male2" / "m2_belt_i_arpeggios.wav"
            ignored = root / "by_technique" / "unknown_style" / "female1" / "f1_unknown_a.wav"
            breathy.parent.mkdir(parents=True)
            belt.parent.mkdir(parents=True)
            ignored.parent.mkdir(parents=True)
            breathy.touch()
            belt.touch()
            ignored.touch()

            records = build_vocalset_manifest(str(root), "vocalset", "studio_exercises")

        self.assertEqual(len(records), 2)
        by_source = {record["labels"]["source_technique"]: record for record in records}
        self.assertEqual(by_source["breathy"]["labels"]["families"], ["breathy"])
        self.assertEqual(by_source["breathy"]["labels"]["techniques"], ["breathy"])
        self.assertEqual(by_source["belt"]["labels"]["families"], ["mixed_voice"])
        self.assertEqual(by_source["belt"]["labels"]["techniques"], ["mix"])
        self.assertEqual(by_source["breathy"]["split_group"], "female1")
        self.assertEqual(validate_record(by_source["breathy"]), [])
        self.assertEqual(validate_record(by_source["belt"]), [])

    def test_vocalset_manifest_prefers_data_by_technique_and_skips_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "VocalSet"
            canonical = root / "data_by_technique" / "breathy" / "female1" / "f1_breathy_a_scales.wav"
            duplicate = root / "data_by_singer" / "female1" / "scales" / "breathy" / "f1_breathy_a_scales.wav"
            sidecar = root / "__MACOSX" / "data_by_technique" / "breathy" / "female1" / "._f1_breathy_a_scales.wav"
            canonical.parent.mkdir(parents=True)
            duplicate.parent.mkdir(parents=True)
            sidecar.parent.mkdir(parents=True)
            canonical.touch()
            duplicate.touch()
            sidecar.touch()

            records = build_vocalset_manifest(str(root), "vocalset", "studio_exercises")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["metadata"]["relative_path"], "data_by_technique/breathy/female1/f1_breathy_a_scales.wav")

    def test_sample_manifest_caps_each_family_deterministically(self) -> None:
        records = []
        for family in ("breathy", "control", "vibrato"):
            for index in range(4):
                records.append(
                    {
                        "recording_id": f"{family}:{index}",
                        "dataset": "fixture",
                        "audio_path": f"data/{family}_{index}.wav",
                        "recording_domain": "test",
                        "labels": {
                            "families": [family],
                            "techniques": ["breathy"] if family == "breathy" else [],
                        },
                    }
                )

        first, summary = sample_records(records, seed=7, max_per_family=2)
        second, _ = sample_records(records, seed=7, max_per_family=2)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertEqual(summary["sampled_summary"]["families"], {"breathy": 2, "control": 2, "vibrato": 2})

    def test_sample_manifest_max_records_round_robins_across_families(self) -> None:
        records = []
        for family in ("breathy", "control", "vibrato"):
            for index in range(3):
                records.append(
                    {
                        "recording_id": f"{family}:{index}",
                        "dataset": "fixture",
                        "audio_path": f"data/{family}_{index}.wav",
                        "recording_domain": "test",
                        "labels": {
                            "families": [family],
                            "techniques": ["breathy"] if family == "breathy" else [],
                        },
                    }
                )

        sampled, summary = sample_records(records, seed=11, max_records=5)

        self.assertEqual(len(sampled), 5)
        self.assertEqual(summary["sampled_summary"]["families"], {"breathy": 2, "control": 2, "vibrato": 1})

    def test_dataset_strategy_classifies_training_validation_and_review_sources(self) -> None:
        report = audit_registry(
            [
                {
                    "id": "gtsinger",
                    "name": "GT Singer",
                    "role": "primary_supervised_technique_source",
                    "priority": 1,
                    "recording_domain": "studio",
                    "label_fit": "Strong fit.",
                    "main_gap": "Studio data.",
                    "source_url": "https://example.com/gtsinger",
                    "acquisition": "python3 -m gt_singer_grader.download_dataset",
                },
                {
                    "id": "vocalset",
                    "name": "VocalSet",
                    "role": "supplemental_supervised_technique_source",
                    "priority": 2,
                    "recording_domain": "studio_exercises",
                    "label_fit": "Technique exercises.",
                    "main_gap": "Studio exercises.",
                    "source_url": "https://example.com/vocalset",
                    "acquisition": "manual_download_then_manifest",
                },
                {
                    "id": "damp_s_ag",
                    "name": "DAMP-S-AG",
                    "role": "domain_adaptation_and_eval_source",
                    "priority": 3,
                    "recording_domain": "mobile_karaoke",
                    "label_fit": "Mobile singing.",
                    "main_gap": "No technique labels.",
                    "source_url": "https://example.com/damp",
                    "acquisition": "manual_download_after_license_review",
                },
                {
                    "id": "app_recordings",
                    "name": "NanoPitch App Recordings",
                    "role": "target_domain_supervised_source",
                    "priority": 1,
                    "recording_domain": "app_user",
                    "label_fit": "Target domain.",
                    "main_gap": "Needs collection.",
                    "source_url": "internal",
                    "acquisition": "collect_with_protocol",
                },
            ]
        )

        self.assertTrue(report["ok"])
        by_id = {item["id"]: item for item in report["datasets"]}
        self.assertEqual(by_id["gtsinger"]["recommendation"], "use_for_primary_baseline")
        self.assertEqual(by_id["vocalset"]["recommendation"], "use_after_baseline_as_supplemental_training")
        self.assertEqual(
            by_id["damp_s_ag"]["recommendation"],
            "use_for_robustness_or_eval_after_license_review_not_supervised_training",
        )
        self.assertEqual(by_id["app_recordings"]["recommendation"], "collect_label_and_use_for_app_validation")
        self.assertEqual(report["recommended_order"][:3], ["gtsinger", "vocalset", "app_recordings"])

    def test_dataset_strategy_reports_registry_errors(self) -> None:
        report = audit_registry(
            [
                {
                    "id": "duplicate",
                    "name": "First",
                    "role": "quality_axis_reference",
                    "priority": 0,
                    "recording_domain": "studio",
                    "label_fit": "Quality labels.",
                    "main_gap": "Not technique.",
                    "source_url": "https://example.com/first",
                    "acquisition": "manual_download",
                },
                {
                    "id": "duplicate",
                    "name": "Second",
                    "role": "quality_axis_reference",
                    "priority": 2,
                    "recording_domain": "studio",
                    "label_fit": "Quality labels.",
                    "main_gap": "Not technique.",
                    "source_url": "https://example.com/second",
                },
            ]
        )

        self.assertFalse(report["ok"])
        self.assertTrue(any("invalid priority" in error for error in report["errors"]))
        self.assertTrue(any("duplicate dataset id" in error for error in report["errors"]))
        self.assertTrue(any("missing required field" in error for error in report["errors"]))

    def test_validation_rejects_unknown_labels(self) -> None:
        record = {
            "recording_id": "x",
            "dataset": "app_recordings",
            "audio_path": "take.wav",
            "recording_domain": "app_user",
            "label_source": "test",
            "split_group": "singer",
            "labels": {
                "families": ["yodel"],
                "techniques": ["vibrato"],
            },
        }
        errors = validate_record(record)
        self.assertTrue(any("unknown family" in error for error in errors))

    def test_write_jsonl_creates_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "manifest.jsonl"
            write_jsonl(path, [{"recording_id": "x"}])
            self.assertEqual(read_jsonl(path), [{"recording_id": "x"}])

    def test_require_non_empty_records_reports_source_and_purpose(self) -> None:
        with self.assertRaises(ValueError) as context:
            require_non_empty_records([], source="train.jsonl", purpose="training")

        message = str(context.exception)
        self.assertIn("train.jsonl", message)
        self.assertIn("training", message)

    def test_manifest_summary_counts_families_and_trainability(self) -> None:
        records = [
            {
                "recording_id": "app:vibrato",
                "dataset": "app_recordings",
                "audio_path": "vibrato.wav",
                "recording_domain": "app_user",
                "label_source": "test",
                "split_group": "singer_1",
                "labels": {
                    "families": ["vibrato"],
                    "techniques": ["vibrato"],
                },
            },
            {
                "recording_id": "app:unclear",
                "dataset": "app_recordings",
                "audio_path": "unclear.wav",
                "recording_domain": "app_user",
                "label_source": "test",
                "split_group": "singer_2",
                "labels": {
                    "families": ["unclear"],
                    "techniques": [],
                },
            },
            {
                "recording_id": "app:multi",
                "dataset": "app_recordings",
                "audio_path": "multi.wav",
                "recording_domain": "app_user",
                "label_source": "test",
                "split_group": "singer_3",
                "labels": {
                    "families": ["vibrato", "breathy"],
                    "techniques": ["vibrato", "breathy"],
                },
            },
        ]

        summary = summarize_records(records)

        self.assertEqual(summary["records"], 3)
        self.assertEqual(summary["families"]["vibrato"], 2)
        self.assertEqual(summary["trainability"]["trainable"], 1)
        self.assertEqual(summary["trainability"]["evaluation_only_family:unclear"], 1)
        self.assertEqual(summary["trainability"]["multiple_families"], 1)

    def test_trainability_reason_marks_eval_only_records(self) -> None:
        self.assertEqual(
            trainability_reason({"labels": {"families": ["vibrato"], "techniques": ["vibrato"]}}),
            "trainable",
        )
        self.assertEqual(
            trainability_reason({"labels": {"families": ["none"], "techniques": []}}),
            "evaluation_only_family:none",
        )
        self.assertEqual(
            trainability_reason({"labels": {"families": ["vibrato", "breathy"], "techniques": []}}),
            "multiple_families",
        )

    def test_split_manifest_keeps_split_groups_disjoint(self) -> None:
        records = []
        for family in ("vibrato", "breathy"):
            for group_index in range(4):
                for take_index in range(2):
                    records.append(
                        {
                            "recording_id": f"{family}:{group_index}:{take_index}",
                            "dataset": "app_recordings",
                            "audio_path": f"{family}_{group_index}_{take_index}.wav",
                            "recording_domain": "app_user",
                            "label_source": "test",
                            "split_group": f"singer_{family}_{group_index}",
                            "labels": {
                                "families": [family],
                                "techniques": [family],
                            },
                        }
                    )

        train_records, val_records, summary = split_manifest_records(
            records,
            val_ratio=0.25,
            seed=7,
            group_field="split_group",
        )

        train_groups = {record["split_group"] for record in train_records}
        val_groups = {record["split_group"] for record in val_records}
        self.assertFalse(train_groups & val_groups)
        self.assertGreater(len(train_records), 0)
        self.assertGreater(len(val_records), 0)
        self.assertEqual(summary["input_records"], len(records))
        self.assertEqual(summary["warnings"], [])
        self.assertEqual(summary["coverage_errors"], [])
        self.assertTrue(all(validate_record(record) == [] for record in train_records + val_records))

    def test_split_manifest_reports_family_coverage_errors(self) -> None:
        records = [
            {
                "recording_id": "app:vibrato-1",
                "dataset": "app_recordings",
                "audio_path": "vibrato-1.wav",
                "recording_domain": "app_user",
                "label_source": "test",
                "split_group": "singer_vibrato_1",
                "labels": {"families": ["vibrato"], "techniques": ["vibrato"]},
            },
            {
                "recording_id": "app:vibrato-2",
                "dataset": "app_recordings",
                "audio_path": "vibrato-2.wav",
                "recording_domain": "app_user",
                "label_source": "test",
                "split_group": "singer_vibrato_2",
                "labels": {"families": ["vibrato"], "techniques": ["vibrato"]},
            },
            {
                "recording_id": "app:breathy-1",
                "dataset": "app_recordings",
                "audio_path": "breathy-1.wav",
                "recording_domain": "app_user",
                "label_source": "test",
                "split_group": "singer_breathy_1",
                "labels": {"families": ["breathy"], "techniques": ["breathy"]},
            },
        ]

        _train_records, _val_records, summary = split_manifest_records(
            records,
            val_ratio=0.5,
            seed=7,
            group_field="split_group",
        )

        self.assertTrue(summary["coverage_errors"])
        self.assertTrue(any("breathy" in error for error in summary["coverage_errors"]))

    def test_split_manifest_warns_when_validation_split_is_empty(self) -> None:
        records = [
            {
                "recording_id": "vibrato:1",
                "dataset": "app_recordings",
                "audio_path": "vibrato.wav",
                "recording_domain": "app_user",
                "label_source": "test",
                "split_group": "singer_1",
                "labels": {
                    "families": ["vibrato"],
                    "techniques": ["vibrato"],
                },
            }
        ]

        train_records, val_records, summary = split_manifest_records(
            records,
            val_ratio=0.5,
            seed=7,
            group_field="split_group",
        )

        self.assertEqual(len(train_records), 1)
        self.assertEqual(len(val_records), 0)
        self.assertIn("validation split is empty", summary["warnings"])

    def test_split_manifest_rejects_invalid_validation_ratio(self) -> None:
        with self.assertRaises(ValueError) as exc:
            split_manifest_records([], val_ratio=1.0, seed=7, group_field="split_group")

        self.assertIn("val_ratio", str(exc.exception))

    def test_split_health_reports_single_family_training_split(self) -> None:
        records = [
            {
                "recording_id": "app:vibrato-1",
                "dataset": "app_recordings",
                "audio_path": "vibrato-1.wav",
                "recording_domain": "app_user",
                "label_source": "test",
                "split_group": "singer_1",
                "labels": {
                    "families": ["vibrato"],
                    "techniques": ["vibrato"],
                },
            },
            {
                "recording_id": "app:vibrato-2",
                "dataset": "app_recordings",
                "audio_path": "vibrato-2.wav",
                "recording_domain": "app_user",
                "label_source": "test",
                "split_group": "singer_2",
                "labels": {
                    "families": ["vibrato"],
                    "techniques": ["vibrato"],
                },
            },
        ]

        errors = split_coverage_errors(records, source="train.jsonl", purpose="training")

        self.assertEqual(split_family_counts(records), {"vibrato": 2})
        self.assertEqual(len(errors), 1)
        self.assertIn("only 1 labeled clip family", errors[0])

    def test_split_health_accepts_control_plus_technique_split(self) -> None:
        records = [
            {
                "recording_id": "app:vibrato",
                "dataset": "app_recordings",
                "audio_path": "vibrato.wav",
                "recording_domain": "app_user",
                "label_source": "test",
                "split_group": "singer_1",
                "labels": {
                    "families": ["vibrato"],
                    "techniques": ["vibrato"],
                },
            },
            {
                "recording_id": "app:control",
                "dataset": "app_recordings",
                "audio_path": "control.wav",
                "recording_domain": "app_user",
                "label_source": "test",
                "split_group": "singer_2",
                "labels": {
                    "families": ["control"],
                    "techniques": [],
                },
            },
        ]

        self.assertEqual(split_coverage_errors(records, source="val.jsonl", purpose="validation"), [])

    def test_split_health_reports_family_mismatch_between_train_and_validation(self) -> None:
        train_records = [
            {
                "recording_id": "app:vibrato",
                "dataset": "app_recordings",
                "audio_path": "vibrato.wav",
                "recording_domain": "app_user",
                "label_source": "test",
                "split_group": "singer_1",
                "labels": {"families": ["vibrato"], "techniques": ["vibrato"]},
            },
            {
                "recording_id": "app:control-train",
                "dataset": "app_recordings",
                "audio_path": "control_train.wav",
                "recording_domain": "app_user",
                "label_source": "test",
                "split_group": "singer_1",
                "labels": {"families": ["control"], "techniques": []},
            },
        ]
        val_records = [
            {
                "recording_id": "app:breathy",
                "dataset": "app_recordings",
                "audio_path": "breathy.wav",
                "recording_domain": "app_user",
                "label_source": "test",
                "split_group": "singer_2",
                "labels": {"families": ["breathy"], "techniques": ["breathy"]},
            },
            {
                "recording_id": "app:control-val",
                "dataset": "app_recordings",
                "audio_path": "control_val.wav",
                "recording_domain": "app_user",
                "label_source": "test",
                "split_group": "singer_2",
                "labels": {"families": ["control"], "techniques": []},
            },
        ]

        errors = split_family_compatibility_errors(train_records, val_records, source="manifest")

        self.assertEqual(len(errors), 2)
        self.assertIn("no training examples: breathy", errors[0])
        self.assertIn("no validation examples: vibrato", errors[1])

    def test_split_health_rejects_control_only_split(self) -> None:
        records = [
            {
                "recording_id": "app:control",
                "dataset": "app_recordings",
                "audio_path": "control.wav",
                "recording_domain": "app_user",
                "label_source": "test",
                "split_group": "singer_1",
                "labels": {
                    "families": ["control"],
                    "techniques": [],
                },
            }
        ]

        errors = split_coverage_errors(records, source="val.jsonl", purpose="validation")

        self.assertEqual(len(errors), 2)
        self.assertIn("only 1 labeled clip family", errors[0])
        self.assertIn("no non-control technique family", errors[1])

    def test_training_plan_reports_manifest_split_readiness_without_torch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_manifest = root / "train.jsonl"
            val_manifest = root / "val.jsonl"
            records = [
                {
                    "recording_id": "app:vibrato",
                    "dataset": "app_recordings",
                    "audio_path": "vibrato.wav",
                    "recording_domain": "app_user",
                    "label_source": "test",
                    "split_group": "singer_1",
                    "labels": {"families": ["vibrato"], "techniques": ["vibrato"]},
                },
                {
                    "recording_id": "app:control",
                    "dataset": "app_recordings",
                    "audio_path": "control.wav",
                    "recording_domain": "app_user",
                    "label_source": "test",
                    "split_group": "singer_2",
                    "labels": {"families": ["control"], "techniques": []},
                },
            ]
            write_jsonl(train_manifest, records)
            write_jsonl(val_manifest, records)

            report = build_training_plan(
                Namespace(
                    dataset_root=None,
                    train_manifest=str(train_manifest),
                    val_manifest=str(val_manifest),
                    extra_train_manifest=[],
                    language="English",
                    split_group="speaker",
                    val_ratio=0.2,
                    seed=1337,
                    include_speech=False,
                )
            )

        self.assertTrue(report["ok"])
        self.assertEqual(report["source"], "manifest")
        self.assertEqual(report["split"]["train_examples"], 2)
        self.assertEqual(report["train_summary"]["families"], {"control": 1, "vibrato": 1})
        self.assertEqual(report["inputs"]["train_manifest"], str(train_manifest))
        self.assertEqual(report["inputs"]["val_manifest"], str(val_manifest))
        self.assertEqual(report["inputs"]["split_group"], "speaker")

    def test_training_plan_requires_configured_train_and_validation_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_manifest = root / "train.jsonl"
            val_manifest = root / "val.jsonl"
            records = [
                {
                    "recording_id": "app:vibrato",
                    "dataset": "app_recordings",
                    "audio_path": "vibrato.wav",
                    "recording_domain": "app_user",
                    "label_source": "test",
                    "split_group": "singer_1",
                    "labels": {"families": ["vibrato"], "techniques": ["vibrato"]},
                },
                {
                    "recording_id": "app:control",
                    "dataset": "app_recordings",
                    "audio_path": "control.wav",
                    "recording_domain": "app_user",
                    "label_source": "test",
                    "split_group": "singer_2",
                    "labels": {"families": ["control"], "techniques": []},
                },
            ]
            write_jsonl(train_manifest, records)
            write_jsonl(val_manifest, records)

            report = build_training_plan(
                Namespace(
                    dataset_root=None,
                    train_manifest=str(train_manifest),
                    val_manifest=str(val_manifest),
                    extra_train_manifest=[],
                    require_train_dataset=["gtsinger", "app_recordings"],
                    require_val_dataset=["app_recordings"],
                    language="English",
                    split_group="speaker",
                    val_ratio=0.2,
                    seed=1337,
                    include_speech=False,
                )
            )

        self.assertFalse(report["ok"])
        self.assertTrue(any("missing required training dataset 'gtsinger'" in error for error in report["errors"]))

    def test_training_plan_accepts_required_datasets_for_app_adapted_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_manifest = root / "train.jsonl"
            val_manifest = root / "val.jsonl"
            app_records = [
                {
                    "recording_id": "app:vibrato",
                    "dataset": "app_recordings",
                    "audio_path": "vibrato.wav",
                    "recording_domain": "app_user",
                    "label_source": "test",
                    "split_group": "singer_1",
                    "labels": {"families": ["vibrato"], "techniques": ["vibrato"]},
                },
                {
                    "recording_id": "app:control",
                    "dataset": "app_recordings",
                    "audio_path": "control.wav",
                    "recording_domain": "app_user",
                    "label_source": "test",
                    "split_group": "singer_2",
                    "labels": {"families": ["control"], "techniques": []},
                },
            ]
            gtsinger_records = [
                {
                    "speaker": "singer_a",
                    "technique_folder": "Vibrato",
                    "family": "vibrato",
                    "role": "emphasis",
                    "song": "song_a",
                    "stem": "singer_a_vibrato",
                    "wav_path": "gtsinger/vibrato.wav",
                    "json_path": "gtsinger/vibrato.json",
                    "split_group": "singer_a|Vibrato|song_a",
                },
                {
                    "speaker": "singer_b",
                    "technique_folder": "Vibrato",
                    "family": "control",
                    "role": "control",
                    "song": "song_a",
                    "stem": "singer_b_control",
                    "wav_path": "gtsinger/control.wav",
                    "json_path": "gtsinger/control.json",
                    "split_group": "singer_b|Vibrato|song_a",
                },
            ]
            write_jsonl(train_manifest, gtsinger_records + app_records)
            write_jsonl(val_manifest, app_records)

            report = build_training_plan(
                Namespace(
                    dataset_root=None,
                    train_manifest=str(train_manifest),
                    val_manifest=str(val_manifest),
                    extra_train_manifest=[],
                    require_train_dataset=["gtsinger", "app_recordings"],
                    require_val_dataset=["app_recordings"],
                    language="English",
                    split_group="speaker",
                    val_ratio=0.2,
                    seed=1337,
                    include_speech=False,
                )
            )

        self.assertTrue(report["ok"])
        self.assertEqual(report["inputs"]["require_train_dataset"], ["gtsinger", "app_recordings"])
        self.assertEqual(report["inputs"]["require_val_dataset"], ["app_recordings"])

    def test_training_plan_rejects_manifest_family_mismatch_between_train_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_manifest = root / "train.jsonl"
            val_manifest = root / "val.jsonl"
            train_records = [
                {
                    "recording_id": "app:vibrato",
                    "dataset": "app_recordings",
                    "audio_path": "vibrato.wav",
                    "recording_domain": "app_user",
                    "label_source": "test",
                    "split_group": "singer_1",
                    "labels": {"families": ["vibrato"], "techniques": ["vibrato"]},
                },
                {
                    "recording_id": "app:control-train",
                    "dataset": "app_recordings",
                    "audio_path": "control_train.wav",
                    "recording_domain": "app_user",
                    "label_source": "test",
                    "split_group": "singer_1",
                    "labels": {"families": ["control"], "techniques": []},
                },
            ]
            val_records = [
                {
                    "recording_id": "app:breathy",
                    "dataset": "app_recordings",
                    "audio_path": "breathy.wav",
                    "recording_domain": "app_user",
                    "label_source": "test",
                    "split_group": "singer_2",
                    "labels": {"families": ["breathy"], "techniques": ["breathy"]},
                },
                {
                    "recording_id": "app:control-val",
                    "dataset": "app_recordings",
                    "audio_path": "control_val.wav",
                    "recording_domain": "app_user",
                    "label_source": "test",
                    "split_group": "singer_2",
                    "labels": {"families": ["control"], "techniques": []},
                },
            ]
            write_jsonl(train_manifest, train_records)
            write_jsonl(val_manifest, val_records)

            report = build_training_plan(
                Namespace(
                    dataset_root=None,
                    train_manifest=str(train_manifest),
                    val_manifest=str(val_manifest),
                    extra_train_manifest=[],
                    language="English",
                    split_group="speaker",
                    val_ratio=0.2,
                    seed=1337,
                    include_speech=False,
                )
            )

        self.assertFalse(report["ok"])
        self.assertTrue(any("no training examples: breathy" in error for error in report["errors"]))
        self.assertTrue(any("no validation examples: vibrato" in error for error in report["errors"]))

    def test_training_plan_match_errors_catch_command_drift(self) -> None:
        plan_args = Namespace(
            dataset_root="./gt_singer_grader/data/GTSinger",
            train_manifest=None,
            val_manifest=None,
            extra_train_manifest=["./gt_singer_grader/manifests/vocalset.jsonl"],
            language="English",
            split_group="speaker",
            val_ratio=0.2,
            seed=1337,
            include_speech=False,
        )
        plan = {
            "ok": True,
            "inputs": {
                "dataset_root": plan_args.dataset_root,
                "train_manifest": plan_args.train_manifest,
                "val_manifest": plan_args.val_manifest,
                "extra_train_manifest": plan_args.extra_train_manifest,
                "require_train_dataset": [],
                "require_val_dataset": [],
                "language": plan_args.language,
                "split_group": plan_args.split_group,
                "val_ratio": plan_args.val_ratio,
                "seed": plan_args.seed,
                "include_speech": plan_args.include_speech,
            },
        }
        matching_args = Namespace(**vars(plan_args))
        mismatched_args = Namespace(**{**vars(plan_args), "split_group": "song"})

        self.assertEqual(plan_match_errors(plan, matching_args), [])
        errors = plan_match_errors(plan, mismatched_args)
        self.assertEqual(len(errors), 1)
        self.assertIn("split_group", errors[0])

    def test_training_plan_rejects_eval_only_training_manifest_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_manifest = root / "train.jsonl"
            val_manifest = root / "val.jsonl"
            records = [
                {
                    "recording_id": "app:unclear",
                    "dataset": "app_recordings",
                    "audio_path": "unclear.wav",
                    "recording_domain": "app_user",
                    "label_source": "test",
                    "split_group": "singer_1",
                    "labels": {"families": ["unclear"], "techniques": []},
                }
            ]
            write_jsonl(train_manifest, records)
            write_jsonl(val_manifest, records)

            report = build_training_plan(
                Namespace(
                    dataset_root=None,
                    train_manifest=str(train_manifest),
                    val_manifest=str(val_manifest),
                    extra_train_manifest=[],
                    language="English",
                    split_group="speaker",
                    val_ratio=0.2,
                    seed=1337,
                    include_speech=False,
                )
            )

        self.assertFalse(report["ok"])
        self.assertTrue(any("evaluation_only_family:unclear" in error for error in report["errors"]))

    def test_training_plan_rejects_invalid_validation_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_manifest = root / "train.jsonl"
            val_manifest = root / "val.jsonl"
            records = [
                {
                    "recording_id": "app:vibrato",
                    "dataset": "app_recordings",
                    "audio_path": "vibrato.wav",
                    "recording_domain": "app_user",
                    "label_source": "test",
                    "split_group": "singer_1",
                    "labels": {"families": ["vibrato"], "techniques": ["vibrato"]},
                },
                {
                    "recording_id": "app:control",
                    "dataset": "app_recordings",
                    "audio_path": "control.wav",
                    "recording_domain": "app_user",
                    "label_source": "test",
                    "split_group": "singer_2",
                    "labels": {"families": ["control"], "techniques": []},
                },
            ]
            write_jsonl(train_manifest, records)
            write_jsonl(val_manifest, records)

            report = build_training_plan(
                Namespace(
                    dataset_root=None,
                    train_manifest=str(train_manifest),
                    val_manifest=str(val_manifest),
                    extra_train_manifest=[],
                    language="English",
                    split_group="speaker",
                    val_ratio=1.0,
                    seed=1337,
                    include_speech=False,
                )
            )

        self.assertFalse(report["ok"])
        self.assertIn("val_ratio must be >= 0 and < 1", report["errors"])

    def test_training_plan_scans_gtsinger_layout_without_torch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "GTSinger" / "English"
            for speaker in ("singer_1", "singer_2"):
                for group in ("Vibrato_Group", "Control_Group"):
                    group_dir = root / speaker / "Vibrato" / "song_a" / group
                    group_dir.mkdir(parents=True)
                    stem = f"{speaker}_{group}"
                    (group_dir / f"{stem}.wav").write_bytes(b"wav")
                    (group_dir / f"{stem}.json").write_text("{}\n", encoding="utf-8")

            report = build_training_plan(
                Namespace(
                    dataset_root=str(root.parent),
                    train_manifest=None,
                    val_manifest=None,
                    extra_train_manifest=[],
                    language="English",
                    split_group="speaker",
                    val_ratio=0.5,
                    seed=1337,
                    include_speech=False,
                )
            )

        self.assertTrue(report["ok"])
        self.assertEqual(report["source"], "gtsinger")
        self.assertEqual(report["split"]["train_examples"], 2)
        self.assertEqual(report["split"]["val_examples"], 2)
        self.assertEqual(report["train_summary"]["families"], {"control": 1, "vibrato": 1})

    def test_filter_manifest_separates_trainable_from_eval_only_records(self) -> None:
        records = [
            {
                "recording_id": "app:vibrato",
                "dataset": "app_recordings",
                "audio_path": "vibrato.wav",
                "recording_domain": "app_user",
                "label_source": "test",
                "split_group": "singer_1",
                "labels": {
                    "families": ["vibrato"],
                    "techniques": ["vibrato"],
                },
            },
            {
                "recording_id": "app:control",
                "dataset": "app_recordings",
                "audio_path": "control.wav",
                "recording_domain": "app_user",
                "label_source": "test",
                "split_group": "singer_2",
                "labels": {
                    "families": ["control"],
                    "techniques": [],
                },
            },
            {
                "recording_id": "app:unclear",
                "dataset": "app_recordings",
                "audio_path": "unclear.wav",
                "recording_domain": "app_user",
                "label_source": "test",
                "split_group": "singer_3",
                "labels": {
                    "families": ["unclear"],
                    "techniques": [],
                },
            },
            {
                "recording_id": "app:none",
                "dataset": "app_recordings",
                "audio_path": "none.wav",
                "recording_domain": "app_user",
                "label_source": "test",
                "split_group": "singer_4",
                "labels": {
                    "families": ["none"],
                    "techniques": [],
                },
            },
            {
                "recording_id": "app:multiple-trainable",
                "dataset": "app_recordings",
                "audio_path": "multiple_trainable.wav",
                "recording_domain": "app_user",
                "label_source": "test",
                "split_group": "singer_5",
                "labels": {
                    "families": ["vibrato", "breathy"],
                    "techniques": ["vibrato", "breathy"],
                },
            },
            {
                "recording_id": "app:mixed-eval-only",
                "dataset": "app_recordings",
                "audio_path": "mixed_eval_only.wav",
                "recording_domain": "app_user",
                "label_source": "test",
                "split_group": "singer_6",
                "labels": {
                    "families": ["vibrato", "unclear"],
                    "techniques": ["vibrato"],
                },
            },
        ]

        trainable, eval_only, summary = split_trainable_records(records)

        self.assertEqual([record["recording_id"] for record in trainable], ["app:vibrato", "app:control"])
        self.assertEqual(
            [record["recording_id"] for record in eval_only],
            ["app:unclear", "app:none", "app:multiple-trainable", "app:mixed-eval-only"],
        )
        self.assertEqual(summary["trainable_records"], 2)
        self.assertEqual(summary["eval_only_records"], 4)
        self.assertEqual(summary["reason_counts"]["evaluation_only_family:none"], 1)
        self.assertEqual(summary["reason_counts"]["evaluation_only_family:unclear"], 1)
        self.assertEqual(summary["reason_counts"]["multiple_families"], 2)
        self.assertEqual(summary["summaries"]["input"]["records"], 6)
        self.assertEqual(summary["summaries"]["trainable"]["trainability"]["trainable"], 2)
        self.assertEqual(summary["summaries"]["eval_only"]["records"], 4)

    def test_merge_manifest_combines_records_and_rejects_duplicates(self) -> None:
        trainable_record = {
            "recording_id": "app:vibrato",
            "dataset": "app_recordings",
            "audio_path": "vibrato.wav",
            "recording_domain": "app_user",
            "label_source": "test",
            "split_group": "singer_1",
            "labels": {
                "families": ["vibrato"],
                "techniques": ["vibrato"],
            },
        }
        eval_only_record = {
            "recording_id": "app:unclear",
            "dataset": "app_recordings",
            "audio_path": "unclear.wav",
            "recording_domain": "app_user",
            "label_source": "test",
            "split_group": "singer_2",
            "labels": {
                "families": ["unclear"],
                "techniques": [],
            },
        }

        merged, summary = merge_manifest_records(
            [
                ("val.jsonl", [trainable_record]),
                ("eval_only.jsonl", [eval_only_record]),
            ]
        )

        self.assertEqual([record["recording_id"] for record in merged], ["app:vibrato", "app:unclear"])
        self.assertEqual(summary["merged_records"], 2)
        self.assertEqual(summary["family_counts"]["unclear"], 1)

        with self.assertRaises(ValueError):
            merge_manifest_records(
                [
                    ("first.jsonl", [trainable_record]),
                    ("duplicate.jsonl", [trainable_record]),
                ]
            )

    def test_app_adapted_train_manifest_merge_is_trainable_for_planning(self) -> None:
        def record(recording_id: str, family: str, split_group: str, *, dataset: str) -> dict[str, object]:
            return {
                "recording_id": recording_id,
                "dataset": dataset,
                "audio_path": f"{recording_id.replace(':', '_')}.wav",
                "recording_domain": "app_user" if dataset == "app_recordings" else "studio",
                "label_source": "test",
                "split_group": split_group,
                "labels": {
                    "families": [family],
                    "techniques": [] if family == "control" else [family],
                },
            }

        baseline_train = [
            record("gtsinger:vibrato-001", "vibrato", "gtsinger_singer_001", dataset="gtsinger"),
            record("gtsinger:breathy-001", "breathy", "gtsinger_singer_002", dataset="gtsinger"),
            record("gtsinger:control-001", "control", "gtsinger_singer_003", dataset="gtsinger"),
        ]
        app_train = [
            record("app:vibrato-001", "vibrato", "app_singer_001", dataset="app_recordings"),
            record("app:breathy-001", "breathy", "app_singer_002", dataset="app_recordings"),
        ]
        app_val = [
            record("app:vibrato-002", "vibrato", "app_singer_003", dataset="app_recordings"),
            record("app:breathy-002", "breathy", "app_singer_004", dataset="app_recordings"),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_train_path = root / "baseline_train.jsonl"
            app_train_path = root / "app_train.jsonl"
            app_val_path = root / "app_val.jsonl"
            app_adapted_train_path = root / "app_adapted_train.jsonl"
            write_jsonl(baseline_train_path, baseline_train)
            write_jsonl(app_train_path, app_train)
            write_jsonl(app_val_path, app_val)

            merged, summary = merge_manifest_records(
                [
                    (str(baseline_train_path), read_jsonl(baseline_train_path)),
                    (str(app_train_path), read_jsonl(app_train_path)),
                ]
            )
            write_jsonl(app_adapted_train_path, merged)

            plan = build_training_plan(
                Namespace(
                    dataset_root=None,
                    train_manifest=str(app_adapted_train_path),
                    val_manifest=str(app_val_path),
                    extra_train_manifest=[],
                    language="English",
                    split_group="song",
                    val_ratio=0.2,
                    seed=1337,
                    include_speech=False,
                    output_json=None,
                    strict=True,
                )
            )

        self.assertEqual(summary["merged_records"], 5)
        self.assertEqual(summary["family_counts"], {"breathy": 2, "control": 1, "vibrato": 2})
        self.assertTrue(plan["ok"])
        self.assertEqual(plan["errors"], [])
        self.assertEqual(plan["source"], "manifest")
        self.assertEqual(plan["split"]["train_examples"], 5)
        self.assertEqual(plan["split"]["val_examples"], 2)

    def test_compare_runs_reports_deltas_and_gate_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline"
            candidate = root / "candidate"
            write_eval_artifacts(baseline)
            write_eval_artifacts(candidate)
            baseline_metrics = {
                "prediction_accuracy": 0.4,
                "top1_accuracy": 0.4,
                "top2_accuracy": 0.6,
                "clip_macro_f1": 0.35,
                "technique_macro_f1_at_0_30": 0.3,
                "control_false_positive_rate": 0.2,
                "non_technique_false_positive_rate": 0.2,
                "expected_calibration_error": 0.18,
                "maximum_calibration_error": 0.25,
            }
            candidate_metrics = {
                "prediction_accuracy": 0.5,
                "top1_accuracy": 0.5,
                "top2_accuracy": 0.7,
                "clip_macro_f1": 0.42,
                "technique_macro_f1_at_0_30": 0.34,
                "control_false_positive_rate": 0.18,
                "non_technique_false_positive_rate": 0.18,
                "expected_calibration_error": 0.16,
                "maximum_calibration_error": 0.2,
            }
            candidate_operating_point = {
                "confidence_threshold": 0.35,
                "technique_threshold": 0.3,
                "macro_f1": 0.46,
                "prediction_accuracy": 0.55,
                "technique_macro_f1": 0.36,
                "control_false_positive_rate": 0.18,
                "non_technique_false_positive_rate": 0.18,
                "passes_control_false_positive_gate": True,
                "passes_non_technique_false_positive_gate": True,
            }
            (baseline / "metrics.json").write_text(json.dumps(baseline_metrics), encoding="utf-8")
            (candidate / "metrics.json").write_text(json.dumps(candidate_metrics), encoding="utf-8")
            (candidate / "operating_point.json").write_text(json.dumps(candidate_operating_point), encoding="utf-8")

            report = build_compare_report(
                Namespace(
                    baseline=str(baseline),
                    candidate=[str(candidate)],
                    min_top2=0.6,
                    min_macro_f1=0.35,
                    max_control_fpr=0.25,
                    max_non_technique_fpr=0.25,
                    max_ece=0.2,
                )
            )

        candidate_report = report["candidates"][0]
        self.assertEqual(report["regression_gates"]["top2_accuracy"], 0.0)
        self.assertEqual(report["regression_gates"]["clip_macro_f1"], 0.0)
        self.assertEqual(report["regression_gates"]["control_false_positive_rate"], 0.0)
        self.assertEqual(report["regression_gates"]["non_technique_false_positive_rate"], 0.0)
        self.assertEqual(report["regression_gates"]["expected_calibration_error"], 0.0)
        self.assertAlmostEqual(candidate_report["delta_vs_baseline"]["top2_accuracy"], 0.1)
        self.assertTrue(candidate_report["gates"]["top2_accuracy"]["pass"])
        self.assertTrue(candidate_report["gates"]["control_false_positive_rate"]["pass"])
        self.assertTrue(candidate_report["gates"]["non_technique_false_positive_rate"]["pass"])
        self.assertTrue(candidate_report["regression_gates"]["control_false_positive_rate_delta"]["pass"])
        self.assertTrue(candidate_report["regression_gates"]["non_technique_false_positive_rate_delta"]["pass"])
        self.assertEqual(candidate_report["operating_point"]["confidence_threshold"], 0.35)
        self.assertEqual(candidate_report["operating_point"]["control_false_positive_rate"], 0.18)
        self.assertEqual(candidate_report["operating_point"]["non_technique_false_positive_rate"], 0.18)
        self.assertTrue(candidate_report["operating_point"]["passes_non_technique_false_positive_gate"])
        self.assertEqual(
            sorted(candidate_report["evaluation_artifact_sha256"]),
            sorted(REQUIRED_EVALUATION_ARTIFACTS),
        )
        self.assertTrue(candidate_report["promotion"]["eligible"])
        self.assertEqual(candidate_report["promotion"]["failed_gates"], [])
        self.assertEqual(candidate_report["promotion"]["unknown_gates"], [])

    def test_compare_runs_rejects_different_evaluation_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline"
            candidate = root / "candidate"
            baseline_manifest = root / "baseline_manifest.jsonl"
            candidate_manifest = root / "candidate_manifest.jsonl"
            baseline_manifest.write_text('{"recording_id":"app:baseline"}\n', encoding="utf-8")
            candidate_manifest.write_text('{"recording_id":"app:candidate"}\n', encoding="utf-8")
            write_eval_artifacts(baseline, manifest_path=baseline_manifest)
            write_eval_artifacts(candidate, manifest_path=candidate_manifest)
            metrics = {
                "top2_accuracy": 0.7,
                "clip_macro_f1": 0.45,
                "control_false_positive_rate": 0.1,
                "non_technique_false_positive_rate": 0.1,
                "expected_calibration_error": 0.1,
            }
            (baseline / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            (candidate / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

            with self.assertRaises(ValueError) as exc:
                build_compare_report(
                    Namespace(
                        baseline=str(baseline),
                        candidate=[str(candidate)],
                        min_top2=0.6,
                        min_macro_f1=0.35,
                        max_control_fpr=0.25,
                        max_non_technique_fpr=0.25,
                        max_ece=0.2,
                    )
                )

        self.assertIn("evaluation directories are not comparable", str(exc.exception))
        self.assertIn("different manifest sha256", str(exc.exception))

    def test_compare_runs_requires_at_least_one_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "baseline"
            write_eval_artifacts(baseline)
            (baseline / "metrics.json").write_text(json.dumps({"top2_accuracy": 0.6}), encoding="utf-8")

            with self.assertRaises(ValueError) as exc:
                build_compare_report(
                    Namespace(
                        baseline=str(baseline),
                        candidate=[],
                        min_top2=0.6,
                        min_macro_f1=0.35,
                        max_control_fpr=0.25,
                        max_non_technique_fpr=0.25,
                        max_ece=0.2,
                    )
                )

        self.assertIn("at least one --candidate", str(exc.exception))

    def test_compare_runs_rejects_invalid_gate_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline"
            candidate = root / "candidate"
            write_eval_artifacts(baseline)
            write_eval_artifacts(candidate)
            (baseline / "metrics.json").write_text(json.dumps({"top2_accuracy": 0.6}), encoding="utf-8")
            (candidate / "metrics.json").write_text(json.dumps({"top2_accuracy": 0.7}), encoding="utf-8")

            with self.assertRaises(ValueError) as absolute:
                build_compare_report(
                    Namespace(
                        baseline=str(baseline),
                        candidate=[str(candidate)],
                        min_top2=1.2,
                        min_macro_f1=0.35,
                        max_control_fpr=0.25,
                        max_non_technique_fpr=0.25,
                        max_ece=0.2,
                    )
                )
            with self.assertRaises(ValueError) as delta:
                build_compare_report(
                    Namespace(
                        baseline=str(baseline),
                        candidate=[str(candidate)],
                        min_top2=0.6,
                        min_macro_f1=0.35,
                        max_control_fpr=0.25,
                        max_non_technique_fpr=0.25,
                        max_ece=0.2,
                        min_top2_delta=-1.5,
                    )
                )

        self.assertIn("min_top2", str(absolute.exception))
        self.assertIn("between 0.0 and 1.0", str(absolute.exception))
        self.assertIn("min_top2_delta", str(delta.exception))
        self.assertIn("between -1.0 and 1.0", str(delta.exception))

    def test_compare_runs_rejects_non_finite_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline"
            candidate = root / "candidate"
            write_eval_artifacts(baseline)
            write_eval_artifacts(candidate)
            baseline_metrics = {
                "top2_accuracy": 0.6,
                "clip_macro_f1": 0.35,
                "control_false_positive_rate": 0.2,
                "non_technique_false_positive_rate": 0.2,
                "expected_calibration_error": 0.18,
            }
            candidate_metrics = dict(baseline_metrics)
            candidate_metrics["top2_accuracy"] = float("nan")
            (baseline / "metrics.json").write_text(json.dumps(baseline_metrics), encoding="utf-8")
            (candidate / "metrics.json").write_text(json.dumps(candidate_metrics), encoding="utf-8")

            with self.assertRaises(ValueError) as exc:
                build_compare_report(
                    Namespace(
                        baseline=str(baseline),
                        candidate=[str(candidate)],
                        min_top2=0.6,
                        min_macro_f1=0.35,
                        max_control_fpr=0.25,
                        max_non_technique_fpr=0.25,
                        max_ece=0.2,
                    )
                )

        self.assertIn("metric top2_accuracy must be finite", str(exc.exception))

    def test_compare_runs_reports_failed_promotion_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline"
            candidate = root / "candidate"
            write_eval_artifacts(baseline)
            write_eval_artifacts(candidate)
            baseline_metrics = {
                "top2_accuracy": 0.6,
                "clip_macro_f1": 0.35,
                "control_false_positive_rate": 0.2,
                "non_technique_false_positive_rate": 0.2,
                "expected_calibration_error": 0.18,
            }
            candidate_metrics = {
                "top2_accuracy": 0.5,
                "clip_macro_f1": 0.34,
                "control_false_positive_rate": 0.4,
                "non_technique_false_positive_rate": 0.4,
                "expected_calibration_error": 0.25,
            }
            (baseline / "metrics.json").write_text(json.dumps(baseline_metrics), encoding="utf-8")
            (candidate / "metrics.json").write_text(json.dumps(candidate_metrics), encoding="utf-8")

            report = build_compare_report(
                Namespace(
                    baseline=str(baseline),
                    candidate=[str(candidate)],
                    min_top2=0.6,
                    min_macro_f1=0.35,
                    max_control_fpr=0.25,
                    max_non_technique_fpr=0.25,
                    max_ece=0.2,
                )
            )

        promotion = report["candidates"][0]["promotion"]
        self.assertFalse(promotion["eligible"])
        self.assertEqual(
            promotion["failed_gates"],
            [
                "clip_macro_f1",
                "clip_macro_f1_delta",
                "control_false_positive_rate",
                "control_false_positive_rate_delta",
                "expected_calibration_error",
                "expected_calibration_error_delta",
                "non_technique_false_positive_rate",
                "non_technique_false_positive_rate_delta",
                "top2_accuracy",
                "top2_accuracy_delta",
            ],
        )

    def test_compare_runs_rejects_candidate_that_regresses_against_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline"
            candidate = root / "candidate"
            write_eval_artifacts(baseline)
            write_eval_artifacts(candidate)
            baseline_metrics = {
                "top2_accuracy": 0.7,
                "clip_macro_f1": 0.45,
                "control_false_positive_rate": 0.1,
                "non_technique_false_positive_rate": 0.1,
                "expected_calibration_error": 0.1,
            }
            candidate_metrics = {
                "top2_accuracy": 0.7,
                "clip_macro_f1": 0.45,
                "control_false_positive_rate": 0.12,
                "non_technique_false_positive_rate": 0.12,
                "expected_calibration_error": 0.1,
            }
            (baseline / "metrics.json").write_text(json.dumps(baseline_metrics), encoding="utf-8")
            (candidate / "metrics.json").write_text(json.dumps(candidate_metrics), encoding="utf-8")

            report = build_compare_report(
                Namespace(
                    baseline=str(baseline),
                    candidate=[str(candidate)],
                    min_top2=0.6,
                    min_macro_f1=0.35,
                    max_control_fpr=0.25,
                    max_non_technique_fpr=0.25,
                    max_ece=0.2,
                )
            )

        candidate_report = report["candidates"][0]
        self.assertTrue(candidate_report["gates"]["control_false_positive_rate"]["pass"])
        self.assertFalse(candidate_report["regression_gates"]["control_false_positive_rate_delta"]["pass"])
        self.assertFalse(candidate_report["promotion"]["eligible"])
        self.assertEqual(
            candidate_report["promotion"]["failed_gates"],
            ["control_false_positive_rate_delta", "non_technique_false_positive_rate_delta"],
        )

    def test_compare_runs_rejects_incomplete_evaluation_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline"
            candidate = root / "candidate"
            write_eval_artifacts(baseline)
            candidate.mkdir()
            (baseline / "metrics.json").write_text(json.dumps({"top2_accuracy": 0.6}), encoding="utf-8")
            (candidate / "metrics.json").write_text(json.dumps({"top2_accuracy": 0.7}), encoding="utf-8")

            with self.assertRaises(FileNotFoundError) as exc:
                build_compare_report(
                    Namespace(
                        baseline=str(baseline),
                        candidate=[str(candidate)],
                        min_top2=0.6,
                        min_macro_f1=0.35,
                        max_control_fpr=0.25,
                        max_non_technique_fpr=0.25,
                        max_ece=0.2,
                    )
                )

        self.assertIn("missing required evaluation artifact", str(exc.exception))
        self.assertIn("predictions.csv", str(exc.exception))

    def test_compare_runs_rejects_tampered_evaluation_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline"
            candidate = root / "candidate"
            write_eval_artifacts(baseline)
            artifacts = write_eval_artifacts(candidate)
            (baseline / "metrics.json").write_text(json.dumps({"top2_accuracy": 0.6}), encoding="utf-8")
            (candidate / "metrics.json").write_text(json.dumps({"top2_accuracy": 0.7}), encoding="utf-8")
            artifacts["manifest"].write_text("{\"changed\": true}\n", encoding="utf-8")

            with self.assertRaises(ValueError) as exc:
                build_compare_report(
                    Namespace(
                        baseline=str(baseline),
                        candidate=[str(candidate)],
                        min_top2=0.6,
                        min_macro_f1=0.35,
                        max_control_fpr=0.25,
                        max_non_technique_fpr=0.25,
                        max_ece=0.2,
                    )
                )

        self.assertIn("failed evaluation provenance verification", str(exc.exception))
        self.assertIn("evaluation_config.manifest:sha256", str(exc.exception))

    def test_evaluator_treats_multi_family_records_as_multiple(self) -> None:
        record = {
            "recording_id": "app:multi",
            "labels": {"families": ["breathy", "vibrato"], "techniques": ["breathy", "vibrato"]},
        }

        self.assertEqual(gold_family(record), "multiple")

    def test_evaluator_validates_normalized_and_legacy_eval_records(self) -> None:
        normalized_record = {
            "recording_id": "app:vibrato",
            "dataset": "app_recordings",
            "audio_path": "take.wav",
            "recording_domain": "app_user",
            "label_source": "coach_review",
            "split_group": "singer_001",
            "labels": {"families": ["vibrato"], "techniques": ["vibrato"]},
        }
        legacy_record = {
            "stem": "gtsinger-control",
            "wav_path": "take.wav",
            "family": "breathy",
        }

        validate_eval_records([normalized_record, legacy_record], source="eval.jsonl")

    def test_evaluator_rejects_empty_eval_records(self) -> None:
        with self.assertRaises(ValueError) as exc:
            validate_eval_records([], source="empty.jsonl")

        self.assertIn("empty.jsonl has no records for evaluation", str(exc.exception))

    def test_evaluator_reports_invalid_eval_manifest_records(self) -> None:
        bad_record = {
            "recording_id": "app:bad",
            "dataset": "app_recordings",
            "audio_path": "",
            "recording_domain": "app_user",
            "label_source": "coach_review",
            "split_group": "",
            "labels": {"families": ["vibrato"], "techniques": ["vibrato"]},
        }

        with self.assertRaises(ValueError) as exc:
            validate_eval_records([bad_record], source="eval.jsonl")

        self.assertIn("audio_path must be a non-empty string", str(exc.exception))
        self.assertIn("split_group must be a non-empty string", str(exc.exception))

    def test_evaluator_metrics_cover_thresholds_and_false_positives(self) -> None:
        rows = [
            {
                "gold_family": "control",
                "predicted_family": "vibrato",
                "detected_family": "vibrato",
                "detected_confidence": 0.80,
                "primary_technique_score": 0.70,
                "voiced_ratio": 0.90,
                "ranked_families": ["vibrato", "control"],
                "technique_scores": {"vibrato": 0.70},
            },
            {
                "gold_family": "vibrato",
                "predicted_family": "vibrato",
                "detected_family": "vibrato",
                "detected_confidence": 0.70,
                "primary_technique_score": 0.65,
                "voiced_ratio": 0.85,
                "ranked_families": ["vibrato", "breathy"],
                "technique_scores": {"vibrato": 0.65},
            },
            {
                "gold_family": "breathy",
                "predicted_family": "glissando",
                "detected_family": "glissando",
                "detected_confidence": 0.20,
                "primary_technique_score": 0.55,
                "voiced_ratio": 0.80,
                "ranked_families": ["glissando", "vibrato"],
                "technique_scores": {"glissando": 0.55},
            },
            {
                "gold_family": "falsetto",
                "predicted_family": "falsetto",
                "detected_family": "falsetto",
                "detected_confidence": 0.90,
                "primary_technique_score": 0.10,
                "voiced_ratio": 0.04,
                "ranked_families": ["control", "falsetto"],
                "technique_scores": {"falsetto": 0.10},
            },
            {
                "gold_family": "none",
                "predicted_family": "breathy",
                "detected_family": "breathy",
                "detected_confidence": 0.85,
                "primary_technique_score": 0.80,
                "voiced_ratio": 0.90,
                "ranked_families": ["breathy", "control"],
                "technique_scores": {"breathy": 0.80},
            },
            {
                "gold_family": "unclear",
                "predicted_family": "unclear",
                "detected_family": "vibrato",
                "detected_confidence": 0.20,
                "primary_technique_score": 0.40,
                "voiced_ratio": 0.90,
                "ranked_families": ["vibrato", "control"],
                "technique_scores": {"vibrato": 0.40},
            },
            {
                "gold_family": "multiple",
                "predicted_family": "vibrato",
                "detected_family": "vibrato",
                "detected_confidence": 0.88,
                "primary_technique_score": 0.78,
                "voiced_ratio": 0.90,
                "ranked_families": ["vibrato", "breathy"],
                "technique_scores": {"vibrato": 0.78, "breathy": 0.76},
            },
        ]

        self.assertAlmostEqual(prediction_accuracy(rows), 3 / 7)
        self.assertAlmostEqual(top_k_accuracy(rows, 2), 3 / 7)
        self.assertEqual(false_positive_rate(rows), 1.0)
        self.assertEqual(false_positive_rate(rows, negative_gold_families={"control", "none", "unclear"}), 2 / 3)
        self.assertEqual(
            predicted_family_with_thresholds(rows[2], confidence_threshold=0.35, technique_threshold=0.30),
            "unclear",
        )
        self.assertEqual(
            predicted_family_with_thresholds(rows[3], confidence_threshold=0.35, technique_threshold=0.30),
            "not_enough_voice",
        )
        self.assertEqual(
            predicted_family_with_thresholds(
                {
                    "predicted_family": "breathy",
                    "detected_confidence": 0.85,
                    "primary_technique_score": 0.80,
                    "voiced_ratio": 0.90,
                },
                confidence_threshold=0.35,
                technique_threshold=0.30,
            ),
            "breathy",
        )
        sweep = threshold_sweep(rows, [0.35], [0.30])
        self.assertEqual(sweep[0]["prediction_counts"]["unclear"], 2)
        self.assertEqual(sweep[0]["prediction_counts"]["not_enough_voice"], 1)
        self.assertEqual(sweep[0]["non_technique_false_positive_rate"], 2 / 3)

    def test_evaluator_selects_operating_point_with_control_fpr_gate(self) -> None:
        sweep = [
            {
                "confidence_threshold": 0.25,
                "technique_threshold": 0.20,
                "macro_f1": 0.70,
                "prediction_accuracy": 0.78,
                "technique_macro_f1": 0.60,
                "control_false_positive_rate": 0.50,
                "non_technique_false_positive_rate": 0.50,
            },
            {
                "confidence_threshold": 0.35,
                "technique_threshold": 0.30,
                "macro_f1": 0.62,
                "prediction_accuracy": 0.74,
                "technique_macro_f1": 0.58,
                "control_false_positive_rate": 0.20,
                "non_technique_false_positive_rate": 0.20,
            },
            {
                "confidence_threshold": 0.40,
                "technique_threshold": 0.30,
                "macro_f1": 0.61,
                "prediction_accuracy": 0.76,
                "technique_macro_f1": 0.58,
                "control_false_positive_rate": 0.10,
                "non_technique_false_positive_rate": 0.10,
            },
        ]

        selected = select_operating_point(sweep, max_control_fpr=0.25)

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected["confidence_threshold"], 0.35)
        self.assertEqual(selected["technique_threshold"], 0.30)
        self.assertTrue(selected["passes_control_false_positive_gate"])
        self.assertTrue(selected["passes_non_technique_false_positive_gate"])

    def test_evaluator_operating_point_uses_non_technique_fpr_gate(self) -> None:
        sweep = [
            {
                "confidence_threshold": 0.25,
                "technique_threshold": 0.20,
                "macro_f1": 0.80,
                "prediction_accuracy": 0.82,
                "technique_macro_f1": 0.60,
                "control_false_positive_rate": 0.10,
                "non_technique_false_positive_rate": 0.40,
            },
            {
                "confidence_threshold": 0.40,
                "technique_threshold": 0.30,
                "macro_f1": 0.70,
                "prediction_accuracy": 0.74,
                "technique_macro_f1": 0.58,
                "control_false_positive_rate": 0.12,
                "non_technique_false_positive_rate": 0.20,
            },
        ]

        selected = select_operating_point(sweep, max_control_fpr=0.25, max_non_technique_fpr=0.25)

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected["confidence_threshold"], 0.40)
        self.assertTrue(selected["passes_control_false_positive_gate"])
        self.assertTrue(selected["passes_non_technique_false_positive_gate"])

    def test_evaluator_confidence_calibration_bins_accuracy(self) -> None:
        rows = [
            {"gold_family": "vibrato", "predicted_family": "vibrato", "detected_confidence": 0.20},
            {"gold_family": "breathy", "predicted_family": "vibrato", "detected_confidence": 0.30},
            {"gold_family": "control", "predicted_family": "control", "detected_confidence": 0.80},
            {"gold_family": "falsetto", "predicted_family": "control", "detected_confidence": 0.90},
        ]

        calibration = confidence_calibration(rows, bins=2)

        self.assertEqual(calibration["bins"][0]["count"], 2)
        self.assertEqual(calibration["bins"][1]["count"], 2)
        self.assertEqual(calibration["bins"][0]["accuracy"], 0.5)
        self.assertEqual(calibration["bins"][1]["accuracy"], 0.5)
        self.assertAlmostEqual(calibration["expected_calibration_error"], 0.3)

    def test_evaluator_rejects_invalid_probability_thresholds(self) -> None:
        self.assertEqual(parse_thresholds("0,0.25,1"), [0.0, 0.25, 1.0])
        with self.assertRaises(ValueError) as low:
            parse_thresholds("-0.1,0.5")
        with self.assertRaises(ValueError) as high:
            parse_thresholds("0.5,1.1")
        with self.assertRaises(ValueError) as gate:
            require_probability(1.5, name="max_control_fpr")

        self.assertIn("between 0.0 and 1.0", str(low.exception))
        self.assertIn("between 0.0 and 1.0", str(high.exception))
        self.assertIn("max_control_fpr", str(gate.exception))

    def test_evaluator_rejects_non_finite_json_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as exc:
                write_json(Path(tmp) / "metrics.json", {"metrics": {"top2_accuracy": float("nan")}})

        self.assertIn("non-finite JSON value", str(exc.exception))

    def test_preflight_reports_missing_required_training_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = build_preflight_report(
                Namespace(
                    gtsinger_root=str(root / "GTSinger"),
                    vocalset_root=str(root / "VocalSet"),
                    app_labels=str(root / "review_labels.csv"),
                    checkpoint=str(root / "missing.pth"),
                    metadata=str(root / "technique_demo_metadata.json"),
                )
            )

        self.assertFalse(report["ok"])
        self.assertIn("dataset:gtsinger", report["required_failures"])
        downloader_check = next(check for check in report["checks"] if check["name"] == "python_module:huggingface_hub")
        self.assertTrue(downloader_check["required"])
        self.assertTrue(any("requirements-training.txt" in step for step in report["next_steps"]))
        self.assertTrue(any("download_dataset" in step for step in report["next_steps"]))
        self.assertFalse(any("app recordings" in step for step in report["next_steps"]))
        self.assertTrue(any("plan_app_collection" in step for step in report["optional_next_steps"]))
        self.assertTrue(any("app recordings" in step for step in report["optional_next_steps"]))

    def test_preflight_requests_collection_materialization_after_plan_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = root / "collection_plan.csv"
            plan.write_text("plan_id,singer_id,intended_family,suggested_filename\n", encoding="utf-8")

            report = build_preflight_report(
                Namespace(
                    gtsinger_root=str(root / "GTSinger"),
                    vocalset_root=str(root / "VocalSet"),
                    app_labels=str(root / "review_labels.csv"),
                    app_collection_plan=str(plan),
                    app_collection_plan_json=str(root / "collection_plan.json"),
                    app_collection_root=str(root / "app_recordings"),
                    app_collection_checklist=str(root / "collection_checklist.csv"),
                    app_collection_materialize_report=str(root / "collection_materialize_report.json"),
                    app_collection_packet_dir=str(root / "collection_packet"),
                    app_collection_packet_summary=str(root / "collection_packet_summary.json"),
                    checkpoint=str(root / "missing.pth"),
                    metadata=str(root / "technique_demo_metadata.json"),
                )
            )

        self.assertTrue(any("materialize_app_collection" in step for step in report["optional_next_steps"]))
        self.assertFalse(any("export_app_collection_packet" in step for step in report["optional_next_steps"]))

    def test_preflight_requests_collection_packet_after_checklist_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = root / "collection_plan.csv"
            checklist = root / "collection_checklist.csv"
            missing = root / "collection_missing.csv"
            materialize_report = root / "collection_materialize_report.json"
            plan.write_text("plan_id,singer_id,intended_family,suggested_filename\n", encoding="utf-8")
            checklist.write_text("plan_id,singer_id,intended_family,expected_audio_path,exists\n", encoding="utf-8")
            missing.write_text("plan_id,singer_id,intended_family,expected_audio_path,exists\n", encoding="utf-8")
            materialize_report.write_text("{}\n", encoding="utf-8")

            report = build_preflight_report(
                Namespace(
                    gtsinger_root=str(root / "GTSinger"),
                    vocalset_root=str(root / "VocalSet"),
                    app_labels=str(root / "review_labels.csv"),
                    app_collection_plan=str(plan),
                    app_collection_plan_json=str(root / "collection_plan.json"),
                    app_collection_root=str(root / "app_recordings"),
                    app_collection_checklist=str(checklist),
                    app_collection_missing=str(missing),
                    app_collection_materialize_report=str(materialize_report),
                    app_collection_packet_dir=str(root / "collection_packet"),
                    app_collection_packet_summary=str(root / "collection_packet_summary.json"),
                    checkpoint=str(root / "missing.pth"),
                    metadata=str(root / "technique_demo_metadata.json"),
                )
            )

        self.assertTrue(any("export_app_collection_packet" in step for step in report["optional_next_steps"]))
        self.assertFalse(
            any("Materialize the app collection plan into recording folders" in step for step in report["optional_next_steps"])
        )
        self.assertTrue(any("--validate-wav-files" in step for step in report["optional_next_steps"]))

    def test_preflight_validates_existing_app_label_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "review_labels.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "audio_path",
                        "recording_id",
                        "singer_id",
                        "intended_family",
                        "mix",
                        "falsetto",
                        "breathy",
                        "pharyngeal",
                        "glissando",
                        "vibrato",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "audio_path": "take.wav",
                        "recording_id": "app:valid",
                        "singer_id": "singer_001",
                        "intended_family": "vibrato",
                        "mix": "absent",
                        "falsetto": "absent",
                        "breathy": "absent",
                        "pharyngeal": "absent",
                        "glissando": "absent",
                        "vibrato": "present",
                    }
                )

            report = build_preflight_report(
                Namespace(
                    gtsinger_root=str(root / "GTSinger"),
                    vocalset_root=str(root / "VocalSet"),
                    app_labels=str(csv_path),
                    checkpoint=str(root / "missing.pth"),
                    metadata=str(root / "technique_demo_metadata.json"),
                )
            )

        app_check = next(check for check in report["checks"] if check["name"] == "app_recording_labels")
        self.assertTrue(app_check["ok"])
        self.assertEqual(app_check["detail"]["summary"]["records"], 1)
        self.assertEqual(app_check["detail"]["summary"]["trainability"]["trainable"], 1)
        self.assertEqual(app_check["detail"]["coverage"]["missing_audio_file_count"], 1)
        self.assertEqual(app_check["detail"]["coverage"]["missing_reviewer_id_count"], 1)
        self.assertIn("review CSV references missing audio files", app_check["detail"]["coverage"]["warnings"])
        self.assertIn("review CSV has labeled rows without reviewer_id", app_check["detail"]["coverage"]["warnings"])
        self.assertTrue(any("plan_app_collection" in step for step in report["optional_next_steps"]))

    def test_preflight_reports_invalid_app_label_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "review_labels.csv"
            csv_path.write_text(
                "audio_path,recording_id,singer_id,vibrato\n"
                "take.wav,app:bad,singer_001,loud\n",
                encoding="utf-8",
            )

            report = build_preflight_report(
                Namespace(
                    gtsinger_root=str(root / "GTSinger"),
                    vocalset_root=str(root / "VocalSet"),
                    app_labels=str(csv_path),
                    checkpoint=str(root / "missing.pth"),
                    metadata=str(root / "technique_demo_metadata.json"),
                )
            )

        app_check = next(check for check in report["checks"] if check["name"] == "app_recording_labels")
        self.assertFalse(app_check["ok"])
        self.assertIn("vibrato must be one of", app_check["detail"]["error"])
        self.assertFalse(any("Fix app recording label CSV errors" in step for step in report["next_steps"]))
        self.assertTrue(any("Fix app recording label CSV errors" in step for step in report["optional_next_steps"]))

    def test_preflight_reports_invalid_packaged_metadata_as_optional(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gtsinger = root / "GTSinger"
            gtsinger.mkdir()
            (gtsinger / "take.wav").write_bytes(b"wav")
            (gtsinger / "take.json").write_text("{}\n", encoding="utf-8")
            checkpoint = root / "technique_demo_best.pth"
            checkpoint.write_bytes(b"checkpoint")
            metadata = root / "technique_demo_metadata.json"
            metadata.write_text(
                json.dumps({"packaged_checkpoint": str(root / "missing.pth")}),
                encoding="utf-8",
            )

            with mock.patch("gt_singer_grader.preflight.importlib.util.find_spec", return_value=object()):
                report = build_preflight_report(
                    Namespace(
                        gtsinger_root=str(gtsinger),
                        vocalset_root=str(root / "VocalSet"),
                        app_labels=str(root / "review_labels.csv"),
                        checkpoint=str(checkpoint),
                        metadata=str(metadata),
                    )
                )

        metadata_check = next(check for check in report["checks"] if check["name"] == "packaged_metadata")
        self.assertTrue(report["ok"])
        self.assertFalse(metadata_check["ok"])
        self.assertFalse(metadata_check["required"])
        self.assertIn("packaged_checkpoint", metadata_check["detail"]["failed_checks"])
        self.assertFalse(any("release-verifiable" in step for step in report["next_steps"]))
        self.assertTrue(any("release-verifiable" in step for step in report["optional_next_steps"]))

    def test_experiment_status_starts_with_preflight_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch("gt_singer_grader.preflight.importlib.util.find_spec", return_value=None):
                report = build_experiment_status_report(
                    Namespace(
                        gtsinger_root=str(root / "GTSinger"),
                        vocalset_root=str(root / "VocalSet"),
                        app_labels=str(root / "review_labels.csv"),
                        checkpoint=str(root / "missing.pth"),
                        metadata=str(root / "technique_demo_metadata.json"),
                        baseline_plan=str(root / "runs" / "gtsinger_speaker_aug_v1" / "training_plan.json"),
                        baseline_run_dir=str(root / "runs" / "gtsinger_speaker_aug_v1"),
                        baseline_eval_dir=str(root / "runs" / "gtsinger_speaker_aug_v1" / "eval_val"),
                        vocalset_manifest=str(root / "manifests" / "vocalset.jsonl"),
                        vocalset_plan=str(root / "runs" / "gtsinger_vocalset_v1" / "training_plan.json"),
                        vocalset_run_dir=str(root / "runs" / "gtsinger_vocalset_v1"),
                        vocalset_eval_dir=str(root / "runs" / "gtsinger_vocalset_v1" / "eval_val"),
                        app_manifest=str(root / "manifests" / "app_recordings.jsonl"),
                        app_trainable_manifest=str(root / "manifests" / "app_recordings_trainable.jsonl"),
                        app_eval_only_manifest=str(root / "manifests" / "app_recordings_eval_only.jsonl"),
                        app_val_manifest=str(root / "manifests" / "app_recordings_val.jsonl"),
                        app_eval_manifest=str(root / "manifests" / "app_recordings_eval.jsonl"),
                        app_validation_audit=str(root / "manifests" / "app_recordings_eval_audit.json"),
                        comparison=str(root / "runs" / "run_comparison.json"),
                    )
                )

        self.assertFalse(report["ready_for_packaging_review"])
        self.assertEqual(report["current_stage"]["name"], "blocked_on_preflight")
        self.assertFalse(report["current_stage"]["ready"])
        self.assertIn("dataset:gtsinger", report["preflight"]["required_failures"])
        self.assertTrue(report["next_actions"])
        self.assertIn("Install training dependencies", report["next_actions"][0])
        self.assertFalse(any("release-verifiable" in action for action in report["next_actions"]))
        self.assertFalse(any("app recordings" in action for action in report["next_actions"]))

    def test_experiment_status_is_not_packaging_ready_without_app_adapted_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gtsinger = root / "GTSinger"
            gtsinger.mkdir()
            (gtsinger / "take.wav").write_bytes(b"wav")
            (gtsinger / "take.json").write_text("{}\n", encoding="utf-8")
            checkpoint = root / "technique_demo_best.pth"
            checkpoint.write_bytes(b"checkpoint")

            def write_complete_run(run_dir: Path) -> None:
                (run_dir / "checkpoints").mkdir(parents=True)
                (run_dir / "metrics_history.jsonl").write_text('{"epoch": 1}\n', encoding="utf-8")
                (run_dir / "best_metrics.json").write_text("{}\n", encoding="utf-8")
                (run_dir / "train_manifest.jsonl").write_text("{}\n", encoding="utf-8")
                (run_dir / "val_manifest.jsonl").write_text("{}\n", encoding="utf-8")
                (run_dir / "checkpoints" / "best.pth").write_bytes(b"checkpoint")
                (run_dir / "run_config.json").write_text(
                    json.dumps(
                        {
                            "artifacts": {
                                "train_manifest": file_metadata(run_dir / "train_manifest.jsonl"),
                                "val_manifest": file_metadata(run_dir / "val_manifest.jsonl"),
                            }
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )

            baseline_run = root / "runs" / "baseline"
            vocalset_run = root / "runs" / "vocalset"
            balanced_run = root / "runs" / "balanced"
            for run_dir in (baseline_run, vocalset_run, balanced_run):
                write_complete_run(run_dir)
                (run_dir / "training_plan.json").write_text('{"ok": true, "errors": []}\n', encoding="utf-8")
                write_eval_artifacts(run_dir / "eval_val")

            manifests = root / "manifests"
            manifests.mkdir()
            app_labels = root / "review_labels.csv"
            app_labels.write_text(
                "audio_path,recording_id,families,techniques\n"
                "raw/a.wav,app:a,breathy,breathy\n",
                encoding="utf-8",
            )
            app_eval_manifest = manifests / "app_recordings_eval.jsonl"
            for name in (
                "vocalset.jsonl",
                "vocalset_balanced_120.jsonl",
                "app_recordings.jsonl",
                "app_recordings_trainable.jsonl",
                "app_recordings_eval_only.jsonl",
                "app_recordings_train.jsonl",
                "app_recordings_val.jsonl",
                "app_recordings_eval.jsonl",
            ):
                (manifests / name).write_text("{}\n", encoding="utf-8")
            os.utime(app_labels, (1_700_000_000, 1_700_000_000))
            os.utime(manifests / "app_recordings.jsonl", (1_700_000_010, 1_700_000_010))
            for name in (
                "app_recordings_trainable.jsonl",
                "app_recordings_eval_only.jsonl",
                "app_recordings_train.jsonl",
                "app_recordings_val.jsonl",
                "app_recordings_eval.jsonl",
            ):
                os.utime(manifests / name, (1_700_000_020, 1_700_000_020))
            audit = manifests / "app_recordings_eval_audit.json"
            write_app_validation_audit(audit, manifest_path=app_eval_manifest)

            comparison = root / "runs" / "run_comparison.json"
            comparison.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "path": str(vocalset_run / "eval_val"),
                                "evaluation_artifact_sha256": eval_artifact_hashes(vocalset_run / "eval_val"),
                                "promotion": {"eligible": True, "failed_gates": [], "unknown_gates": []},
                            }
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with mock.patch("gt_singer_grader.preflight.importlib.util.find_spec", return_value=object()):
                report = build_experiment_status_report(
                    Namespace(
                        gtsinger_root=str(gtsinger),
                        vocalset_root=str(root / "VocalSet"),
                        app_labels=str(app_labels),
                        checkpoint=str(checkpoint),
                        metadata=str(root / "technique_demo_metadata.json"),
                        baseline_plan=str(baseline_run / "training_plan.json"),
                        baseline_run_dir=str(baseline_run),
                        baseline_eval_dir=str(baseline_run / "eval_val"),
                        vocalset_manifest=str(manifests / "vocalset.jsonl"),
                        vocalset_plan=str(vocalset_run / "training_plan.json"),
                        vocalset_run_dir=str(vocalset_run),
                        vocalset_eval_dir=str(vocalset_run / "eval_val"),
                        vocalset_balanced_manifest=str(manifests / "vocalset_balanced_120.jsonl"),
                        vocalset_balanced_plan=str(balanced_run / "training_plan.json"),
                        vocalset_balanced_run_dir=str(balanced_run),
                        vocalset_balanced_eval_dir=str(balanced_run / "eval_val"),
                        app_manifest=str(manifests / "app_recordings.jsonl"),
                        app_trainable_manifest=str(manifests / "app_recordings_trainable.jsonl"),
                        app_eval_only_manifest=str(manifests / "app_recordings_eval_only.jsonl"),
                        app_train_manifest=str(manifests / "app_recordings_train.jsonl"),
                        app_val_manifest=str(manifests / "app_recordings_val.jsonl"),
                        app_eval_manifest=str(app_eval_manifest),
                        app_validation_audit=str(audit),
                        app_baseline_eval_dir=str(baseline_run / "eval_app"),
                        app_adapted_train_manifest=str(manifests / "app_adapted_train.jsonl"),
                        app_adapted_plan=str(root / "runs" / "app_adapted" / "training_plan.json"),
                        app_adapted_run_dir=str(root / "runs" / "app_adapted"),
                        app_adapted_eval_dir=str(root / "runs" / "app_adapted" / "eval_app"),
                        comparison=str(comparison),
                        balanced_comparison=str(root / "runs" / "run_comparison_balanced120.json"),
                        app_adapted_comparison=str(root / "runs" / "run_comparison_app_adapted.json"),
                    )
                )

        self.assertTrue(report["comparison"]["candidate_eligible"])
        self.assertTrue(report["app_validation_audit"]["ready_for_mvp_validation"])
        self.assertFalse(report["ready_for_packaging_review"])
        self.assertEqual(report["current_stage"]["name"], "evaluate_app_baseline")

    def test_experiment_status_advances_completed_baseline_to_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gtsinger = root / "GTSinger"
            gtsinger.mkdir()
            (gtsinger / "take.wav").write_bytes(b"wav")
            (gtsinger / "take.json").write_text("{}\n", encoding="utf-8")
            checkpoint = root / "technique_demo_best.pth"
            checkpoint.write_bytes(b"checkpoint")
            run_dir = root / "runs" / "gtsinger_speaker_aug_v1"
            (run_dir / "checkpoints").mkdir(parents=True)
            (run_dir / "training_plan.json").write_text(
                json.dumps(
                    {
                        "ok": True,
                        "source": "gtsinger",
                        "split": {"train_examples": 2, "val_examples": 2},
                        "errors": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "metrics_history.jsonl").write_text('{"epoch": 1}\n', encoding="utf-8")
            for name in ("best_metrics.json", "train_manifest.jsonl", "val_manifest.jsonl"):
                (run_dir / name).write_text("{}\n", encoding="utf-8")
            (run_dir / "checkpoints" / "best.pth").write_bytes(b"checkpoint")
            train_manifest = run_dir / "train_manifest.jsonl"
            val_manifest = run_dir / "val_manifest.jsonl"
            config = {
                "artifacts": {
                    "train_manifest": file_metadata(train_manifest),
                    "val_manifest": file_metadata(val_manifest),
                }
            }
            (run_dir / "run_config.json").write_text(json.dumps(config), encoding="utf-8")

            with mock.patch("gt_singer_grader.preflight.importlib.util.find_spec", return_value=object()):
                report = build_experiment_status_report(
                    Namespace(
                        gtsinger_root=str(gtsinger),
                        vocalset_root=str(root / "VocalSet"),
                        app_labels=str(root / "review_labels.csv"),
                        checkpoint=str(checkpoint),
                        metadata=str(root / "technique_demo_metadata.json"),
                        baseline_plan=str(run_dir / "training_plan.json"),
                        baseline_run_dir=str(run_dir),
                        baseline_eval_dir=str(run_dir / "eval_val"),
                        vocalset_manifest=str(root / "manifests" / "vocalset.jsonl"),
                        vocalset_plan=str(root / "runs" / "gtsinger_vocalset_v1" / "training_plan.json"),
                        vocalset_run_dir=str(root / "runs" / "gtsinger_vocalset_v1"),
                        vocalset_eval_dir=str(root / "runs" / "gtsinger_vocalset_v1" / "eval_val"),
                        app_manifest=str(root / "manifests" / "app_recordings.jsonl"),
                        app_trainable_manifest=str(root / "manifests" / "app_recordings_trainable.jsonl"),
                        app_eval_only_manifest=str(root / "manifests" / "app_recordings_eval_only.jsonl"),
                        app_val_manifest=str(root / "manifests" / "app_recordings_val.jsonl"),
                        app_eval_manifest=str(root / "manifests" / "app_recordings_eval.jsonl"),
                        app_validation_audit=str(root / "manifests" / "app_recordings_eval_audit.json"),
                        comparison=str(root / "runs" / "run_comparison.json"),
                    )
                )

        self.assertTrue(report["preflight"]["ok"])
        self.assertTrue(report["baseline_run"]["complete"])
        self.assertFalse(report["baseline_eval"]["complete"])
        self.assertEqual(report["current_stage"]["name"], "evaluate_gtsinger_baseline")
        self.assertTrue(report["current_stage"]["ready"])
        self.assertEqual(len(report["next_actions"]), 2)
        self.assertIn("verify_run", report["next_actions"][0])
        self.assertIn("evaluate", report["next_actions"][1])

    def test_experiment_status_plans_baseline_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gtsinger = root / "GTSinger"
            gtsinger.mkdir()
            (gtsinger / "take.wav").write_bytes(b"wav")
            (gtsinger / "take.json").write_text("{}\n", encoding="utf-8")
            checkpoint = root / "technique_demo_best.pth"
            checkpoint.write_bytes(b"checkpoint")

            with mock.patch("gt_singer_grader.preflight.importlib.util.find_spec", return_value=object()):
                report = build_experiment_status_report(
                    Namespace(
                        gtsinger_root=str(gtsinger),
                        vocalset_root=str(root / "VocalSet"),
                        app_labels=str(root / "review_labels.csv"),
                        checkpoint=str(checkpoint),
                        metadata=str(root / "technique_demo_metadata.json"),
                        baseline_plan=str(root / "runs" / "gtsinger_speaker_aug_v1" / "training_plan.json"),
                        baseline_run_dir=str(root / "runs" / "gtsinger_speaker_aug_v1"),
                        baseline_eval_dir=str(root / "runs" / "gtsinger_speaker_aug_v1" / "eval_val"),
                        vocalset_manifest=str(root / "manifests" / "vocalset.jsonl"),
                        vocalset_plan=str(root / "runs" / "gtsinger_vocalset_v1" / "training_plan.json"),
                        vocalset_run_dir=str(root / "runs" / "gtsinger_vocalset_v1"),
                        vocalset_eval_dir=str(root / "runs" / "gtsinger_vocalset_v1" / "eval_val"),
                        app_manifest=str(root / "manifests" / "app_recordings.jsonl"),
                        app_trainable_manifest=str(root / "manifests" / "app_recordings_trainable.jsonl"),
                        app_eval_only_manifest=str(root / "manifests" / "app_recordings_eval_only.jsonl"),
                        app_val_manifest=str(root / "manifests" / "app_recordings_val.jsonl"),
                        app_eval_manifest=str(root / "manifests" / "app_recordings_eval.jsonl"),
                        app_validation_audit=str(root / "manifests" / "app_recordings_eval_audit.json"),
                        comparison=str(root / "runs" / "run_comparison.json"),
                    )
                )

        self.assertEqual(report["current_stage"]["name"], "plan_gtsinger_baseline")
        self.assertIn("plan_training", report["next_actions"][0])
        self.assertIn("--output-json", report["next_actions"][0])

    def test_experiment_status_replans_failed_training_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gtsinger = root / "GTSinger"
            gtsinger.mkdir()
            (gtsinger / "take.wav").write_bytes(b"wav")
            (gtsinger / "take.json").write_text("{}\n", encoding="utf-8")
            checkpoint = root / "technique_demo_best.pth"
            checkpoint.write_bytes(b"checkpoint")
            plan_path = root / "runs" / "gtsinger_speaker_aug_v1" / "training_plan.json"
            plan_path.parent.mkdir(parents=True)
            plan_path.write_text(
                json.dumps({"ok": False, "errors": ["validation split has no records"]}) + "\n",
                encoding="utf-8",
            )

            with mock.patch("gt_singer_grader.preflight.importlib.util.find_spec", return_value=object()):
                report = build_experiment_status_report(
                    Namespace(
                        gtsinger_root=str(gtsinger),
                        vocalset_root=str(root / "VocalSet"),
                        app_labels=str(root / "review_labels.csv"),
                        checkpoint=str(checkpoint),
                        metadata=str(root / "technique_demo_metadata.json"),
                        baseline_plan=str(plan_path),
                        baseline_run_dir=str(root / "runs" / "gtsinger_speaker_aug_v1"),
                        baseline_eval_dir=str(root / "runs" / "gtsinger_speaker_aug_v1" / "eval_val"),
                        vocalset_manifest=str(root / "manifests" / "vocalset.jsonl"),
                        vocalset_plan=str(root / "runs" / "gtsinger_vocalset_v1" / "training_plan.json"),
                        vocalset_run_dir=str(root / "runs" / "gtsinger_vocalset_v1"),
                        vocalset_eval_dir=str(root / "runs" / "gtsinger_vocalset_v1" / "eval_val"),
                        app_manifest=str(root / "manifests" / "app_recordings.jsonl"),
                        app_trainable_manifest=str(root / "manifests" / "app_recordings_trainable.jsonl"),
                        app_eval_only_manifest=str(root / "manifests" / "app_recordings_eval_only.jsonl"),
                        app_val_manifest=str(root / "manifests" / "app_recordings_val.jsonl"),
                        app_eval_manifest=str(root / "manifests" / "app_recordings_eval.jsonl"),
                        app_validation_audit=str(root / "manifests" / "app_recordings_eval_audit.json"),
                        comparison=str(root / "runs" / "run_comparison.json"),
                    )
                )

        self.assertFalse(report["baseline_plan"]["ok"])
        self.assertEqual(report["current_stage"]["name"], "plan_gtsinger_baseline")
        self.assertEqual(report["current_stage"]["detail"]["plan_errors"], ["validation split has no records"])
        self.assertIn("plan_training", report["next_actions"][0])

    def test_experiment_status_steps_through_app_manifest_pipeline(self) -> None:
        args = Namespace(
            gtsinger_root="./gt_singer_grader/data/GTSinger",
            baseline_plan="./gt_singer_grader/runs/gtsinger_speaker_aug_v1/training_plan.json",
            baseline_run_dir="./gt_singer_grader/runs/gtsinger_speaker_aug_v1",
            baseline_eval_dir="./gt_singer_grader/runs/gtsinger_speaker_aug_v1/eval_val",
            vocalset_root="./gt_singer_grader/data/VocalSet",
            vocalset_manifest="./gt_singer_grader/manifests/vocalset.jsonl",
            vocalset_plan="./gt_singer_grader/runs/gtsinger_vocalset_v1/training_plan.json",
            vocalset_run_dir="./gt_singer_grader/runs/gtsinger_vocalset_v1",
            vocalset_eval_dir="./gt_singer_grader/runs/gtsinger_vocalset_v1/eval_val",
            app_labels="./gt_singer_grader/data/app_recordings/review_labels.csv",
            checkpoint="./gt_singer_grader/models/technique_demo_best.pth",
            metadata="./gt_singer_grader/models/technique_demo_metadata.json",
            app_manifest="./gt_singer_grader/manifests/app_recordings.jsonl",
            app_trainable_manifest="./gt_singer_grader/manifests/app_recordings_trainable.jsonl",
            app_eval_only_manifest="./gt_singer_grader/manifests/app_recordings_eval_only.jsonl",
            app_val_manifest="./gt_singer_grader/manifests/app_recordings_val.jsonl",
            app_eval_manifest="./gt_singer_grader/manifests/app_recordings_eval.jsonl",
            app_validation_audit="./gt_singer_grader/manifests/app_recordings_eval_audit.json",
            comparison="./gt_singer_grader/runs/run_comparison.json",
        )
        commands = experiment_command_list(args)
        self.assertIn("--training-plan", commands["train_baseline"])
        self.assertIn(args.baseline_plan, commands["train_baseline"])
        self.assertIn("--training-plan", commands["train_vocalset_candidate"])
        self.assertIn(args.vocalset_plan, commands["train_vocalset_candidate"])
        self.assertIn("--run-config", commands["evaluate_baseline"])
        self.assertIn(f"{args.baseline_run_dir}/run_config.json", commands["evaluate_baseline"])
        self.assertIn("--run-config", commands["evaluate_vocalset_candidate"])
        self.assertIn(f"{args.vocalset_run_dir}/run_config.json", commands["evaluate_vocalset_candidate"])
        self.assertIn("package_candidate", commands["package_candidate"])
        self.assertIn(args.app_validation_audit, commands["package_candidate"])
        self.assertIn("--strict-family-coverage", commands["split_app_trainable_manifest"])
        self.assertIn("plan_app_collection", commands["plan_app_collection"])
        self.assertIn("--output-csv", commands["plan_app_collection"])
        report = {
            "preflight": {"ok": True},
            "baseline_plan": {"exists": True, "ok": True},
            "baseline_run": {"complete": True},
            "baseline_eval": {"complete": True},
            "vocalset_manifest": {"exists": True},
            "vocalset_plan": {"exists": True, "ok": True},
            "vocalset_run": {"complete": True},
            "vocalset_eval": {"complete": True},
            "comparison": {"exists": True, "candidate_eligible": True},
            "app_manifest": {"exists": True},
            "app_trainable_manifest": {"exists": False},
            "app_eval_only_manifest": {"exists": False},
            "app_val_manifest": {"exists": False},
            "app_eval_manifest": {"exists": False},
            "app_validation_audit": {"ready_for_mvp_validation": False},
            "app_baseline_eval": {"complete": False, "missing_artifacts": ["metrics.json"]},
            "app_adapted_train_manifest": {"exists": False},
            "app_adapted_plan": {"exists": False, "ok": False},
            "app_adapted_run": {"complete": False},
            "app_adapted_eval": {"complete": False, "missing_artifacts": ["metrics.json"]},
            "app_adapted_comparison": {"exists": False, "candidate_eligible": False},
        }

        report["app_manifest"]["current_for_sources"] = False
        report["app_manifest"]["stale_source_files"] = [args.app_labels]
        self.assertIn("is stale for its source files", experiment_next_actions(report, commands)[0])
        self.assertEqual(experiment_current_stage(report)["name"], "app_manifest_stale")

        report["app_manifest"]["current_for_sources"] = True
        report["app_manifest"]["stale_source_files"] = []
        self.assertIn("filter_manifest", experiment_next_actions(report, commands)[0])
        report["app_trainable_manifest"]["exists"] = True
        report["app_eval_only_manifest"]["exists"] = True
        self.assertIn("split_manifest", experiment_next_actions(report, commands)[0])
        report["app_val_manifest"]["exists"] = True
        self.assertIn("merge_manifest", experiment_next_actions(report, commands)[0])
        report["app_eval_manifest"]["exists"] = True
        self.assertIn("audit_app_validation", experiment_next_actions(report, commands)[0])
        report["app_validation_audit"]["ready_for_mvp_validation"] = True
        self.assertIn("gt_singer_grader.evaluate", experiment_next_actions(report, commands)[0])

    def test_experiment_status_rejects_ineligible_comparison_before_app_pipeline(self) -> None:
        args = Namespace(
            gtsinger_root="./gt_singer_grader/data/GTSinger",
            baseline_plan="./gt_singer_grader/runs/gtsinger_speaker_aug_v1/training_plan.json",
            baseline_run_dir="./gt_singer_grader/runs/gtsinger_speaker_aug_v1",
            baseline_eval_dir="./gt_singer_grader/runs/gtsinger_speaker_aug_v1/eval_val",
            vocalset_root="./gt_singer_grader/data/VocalSet",
            vocalset_manifest="./gt_singer_grader/manifests/vocalset.jsonl",
            vocalset_plan="./gt_singer_grader/runs/gtsinger_vocalset_v1/training_plan.json",
            vocalset_run_dir="./gt_singer_grader/runs/gtsinger_vocalset_v1",
            vocalset_eval_dir="./gt_singer_grader/runs/gtsinger_vocalset_v1/eval_val",
            app_labels="./gt_singer_grader/data/app_recordings/review_labels.csv",
            checkpoint="./gt_singer_grader/models/technique_demo_best.pth",
            metadata="./gt_singer_grader/models/technique_demo_metadata.json",
            app_manifest="./gt_singer_grader/manifests/app_recordings.jsonl",
            app_trainable_manifest="./gt_singer_grader/manifests/app_recordings_trainable.jsonl",
            app_eval_only_manifest="./gt_singer_grader/manifests/app_recordings_eval_only.jsonl",
            app_val_manifest="./gt_singer_grader/manifests/app_recordings_val.jsonl",
            app_eval_manifest="./gt_singer_grader/manifests/app_recordings_eval.jsonl",
            app_validation_audit="./gt_singer_grader/manifests/app_recordings_eval_audit.json",
            comparison="./gt_singer_grader/runs/run_comparison.json",
        )
        report = {
            "preflight": {"ok": True},
            "baseline_plan": {"exists": True, "ok": True},
            "baseline_run": {"complete": True},
            "baseline_eval": {"complete": True},
            "vocalset_manifest": {"exists": True},
            "vocalset_plan": {"exists": True, "ok": True},
            "vocalset_run": {"complete": True},
            "vocalset_eval": {"complete": True},
            "comparison": {
                "exists": True,
                "candidate_eligible": False,
                "failed_gates": ["control_false_positive_rate_delta"],
                "unknown_gates": [],
            },
            "app_manifest": {"exists": False},
            "app_trainable_manifest": {"exists": False},
            "app_eval_only_manifest": {"exists": False},
            "app_val_manifest": {"exists": False},
            "app_eval_manifest": {"exists": False},
            "app_validation_audit": {"ready_for_mvp_validation": False},
        }

        actions = experiment_next_actions(report, experiment_command_list(args))

        self.assertIn("not promotion-eligible", actions[0])
        self.assertIn("control_false_positive_rate_delta", actions[0])

    def test_experiment_status_tries_balanced_vocalset_after_full_vocalset_regresses(self) -> None:
        args = Namespace(
            gtsinger_root="./gt_singer_grader/data/GTSinger",
            baseline_plan="./gt_singer_grader/runs/gtsinger_song_aug_v1/training_plan.json",
            baseline_run_dir="./gt_singer_grader/runs/gtsinger_song_aug_v1",
            baseline_eval_dir="./gt_singer_grader/runs/gtsinger_song_aug_v1/eval_val",
            vocalset_root="./gt_singer_grader/data/VocalSet",
            vocalset_manifest="./gt_singer_grader/manifests/vocalset.jsonl",
            vocalset_plan="./gt_singer_grader/runs/gtsinger_vocalset_song_v1/training_plan.json",
            vocalset_run_dir="./gt_singer_grader/runs/gtsinger_vocalset_song_v1",
            vocalset_eval_dir="./gt_singer_grader/runs/gtsinger_vocalset_song_v1/eval_val",
            vocalset_balanced_manifest="./gt_singer_grader/manifests/vocalset_balanced_120.jsonl",
            vocalset_balanced_summary="./gt_singer_grader/manifests/vocalset_balanced_120_summary.json",
            vocalset_balanced_max_per_family=120,
            vocalset_balanced_plan="./gt_singer_grader/runs/gtsinger_vocalset_balanced120_song_v1/training_plan.json",
            vocalset_balanced_run_dir="./gt_singer_grader/runs/gtsinger_vocalset_balanced120_song_v1",
            vocalset_balanced_eval_dir="./gt_singer_grader/runs/gtsinger_vocalset_balanced120_song_v1/eval_val",
            app_labels="./gt_singer_grader/data/app_recordings/review_labels.csv",
            checkpoint="./gt_singer_grader/models/technique_demo_best.pth",
            metadata="./gt_singer_grader/models/technique_demo_metadata.json",
            app_manifest="./gt_singer_grader/manifests/app_recordings.jsonl",
            app_trainable_manifest="./gt_singer_grader/manifests/app_recordings_trainable.jsonl",
            app_eval_only_manifest="./gt_singer_grader/manifests/app_recordings_eval_only.jsonl",
            app_val_manifest="./gt_singer_grader/manifests/app_recordings_val.jsonl",
            app_eval_manifest="./gt_singer_grader/manifests/app_recordings_eval.jsonl",
            app_validation_audit="./gt_singer_grader/manifests/app_recordings_eval_audit.json",
            comparison="./gt_singer_grader/runs/run_comparison.json",
            balanced_comparison="./gt_singer_grader/runs/run_comparison_balanced120.json",
        )
        report = {
            "preflight": {"ok": True, "checks": []},
            "baseline_plan": {"exists": True, "ok": True},
            "baseline_run": {"complete": True},
            "baseline_eval": {"complete": True},
            "vocalset_manifest": {"exists": True},
            "vocalset_plan": {"exists": True, "ok": True},
            "vocalset_run": {"complete": True},
            "vocalset_eval": {"complete": True},
            "comparison": {
                "exists": True,
                "candidate_eligible": False,
                "failed_gates": ["clip_macro_f1_delta"],
                "unknown_gates": [],
                "evaluation_artifact_match": True,
            },
            "vocalset_balanced_manifest": {"exists": False},
            "vocalset_balanced_plan": {"exists": False, "ok": False},
            "vocalset_balanced_run": {"complete": False},
            "vocalset_balanced_eval": {"complete": False, "missing_artifacts": []},
            "balanced_comparison": {"exists": False, "candidate_eligible": False},
            "app_manifest": {"exists": False},
            "app_trainable_manifest": {"exists": False},
            "app_eval_only_manifest": {"exists": False},
            "app_val_manifest": {"exists": False},
            "app_eval_manifest": {"exists": False},
            "app_validation_audit": {"ready_for_mvp_validation": False},
        }

        actions = experiment_next_actions(report, experiment_command_list(args))
        stage = experiment_current_stage(report)

        self.assertIn("sample_manifest", actions[0])
        self.assertEqual(stage["name"], "build_balanced_vocalset_manifest")

        report["preflight"]["checks"] = [
            {
                "name": "app_recording_labels",
                "ok": False,
                "detail": "./gt_singer_grader/data/app_recordings/review_labels.csv",
            }
        ]
        report["vocalset_balanced_manifest"]["exists"] = True
        report["vocalset_balanced_plan"] = {"exists": True, "ok": True}
        report["vocalset_balanced_run"]["complete"] = True
        report["vocalset_balanced_eval"] = {"complete": True}
        report["balanced_comparison"] = {
            "exists": True,
            "candidate_eligible": False,
            "failed_gates": ["expected_calibration_error_delta"],
            "unknown_gates": [],
            "evaluation_artifact_match": True,
        }

        actions = experiment_next_actions(report, experiment_command_list(args))
        stage = experiment_current_stage(report)

        self.assertIn("plan_app_collection", actions[0])
        self.assertEqual(stage["name"], "plan_app_recording_collection")
        self.assertIn("balanced_vocalset_failed_gates", stage["detail"])

        report["app_collection_plan"] = {"exists": True}
        actions = experiment_next_actions(report, experiment_command_list(args))
        stage = experiment_current_stage(report)

        self.assertIn("Full VocalSet candidate is not promotion-eligible", actions[0])
        self.assertIn("Balanced VocalSet candidate is not promotion-eligible", actions[0])
        self.assertIn("materialize_app_collection", actions[0])
        self.assertEqual(stage["name"], "materialize_app_collection")

    def test_experiment_status_trains_app_adapted_candidate_after_public_candidates_regress(self) -> None:
        args = Namespace(
            gtsinger_root="./gt_singer_grader/data/GTSinger",
            baseline_plan="./gt_singer_grader/runs/gtsinger_song_aug_v1/training_plan.json",
            baseline_run_dir="./gt_singer_grader/runs/gtsinger_song_aug_v1",
            baseline_eval_dir="./gt_singer_grader/runs/gtsinger_song_aug_v1/eval_val",
            vocalset_root="./gt_singer_grader/data/VocalSet",
            vocalset_manifest="./gt_singer_grader/manifests/vocalset.jsonl",
            vocalset_plan="./gt_singer_grader/runs/gtsinger_vocalset_song_v1/training_plan.json",
            vocalset_run_dir="./gt_singer_grader/runs/gtsinger_vocalset_song_v1",
            vocalset_eval_dir="./gt_singer_grader/runs/gtsinger_vocalset_song_v1/eval_val",
            app_labels="./gt_singer_grader/data/app_recordings/review_labels.csv",
            checkpoint="./gt_singer_grader/models/technique_demo_best.pth",
            metadata="./gt_singer_grader/models/technique_demo_metadata.json",
            app_manifest="./gt_singer_grader/manifests/app_recordings.jsonl",
            app_trainable_manifest="./gt_singer_grader/manifests/app_recordings_trainable.jsonl",
            app_eval_only_manifest="./gt_singer_grader/manifests/app_recordings_eval_only.jsonl",
            app_train_manifest="./gt_singer_grader/manifests/app_recordings_train.jsonl",
            app_val_manifest="./gt_singer_grader/manifests/app_recordings_val.jsonl",
            app_eval_manifest="./gt_singer_grader/manifests/app_recordings_eval.jsonl",
            app_validation_audit="./gt_singer_grader/manifests/app_recordings_eval_audit.json",
            app_adapted_train_manifest="./gt_singer_grader/manifests/app_adapted_train.jsonl",
            app_baseline_eval_dir="./gt_singer_grader/runs/gtsinger_song_aug_v1/eval_app",
            app_adapted_plan="./gt_singer_grader/runs/gtsinger_app_adapted_v1/training_plan.json",
            app_adapted_run_dir="./gt_singer_grader/runs/gtsinger_app_adapted_v1",
            app_adapted_eval_dir="./gt_singer_grader/runs/gtsinger_app_adapted_v1/eval_app",
            comparison="./gt_singer_grader/runs/run_comparison.json",
            balanced_comparison="./gt_singer_grader/runs/run_comparison_balanced120.json",
            app_adapted_comparison="./gt_singer_grader/runs/run_comparison_app_adapted.json",
        )
        commands = experiment_command_list(args)
        self.assertIn(args.app_train_manifest, commands["split_app_trainable_manifest"])
        self.assertIn(args.app_adapted_train_manifest, commands["merge_app_adapted_train_manifest"])
        self.assertIn(args.app_baseline_eval_dir, commands["compare_app_adapted_runs"])
        self.assertIn("package_candidate", commands["package_app_adapted_candidate"])
        self.assertIn(args.app_adapted_comparison, commands["package_app_adapted_candidate"])

        report = {
            "preflight": {"ok": True, "checks": []},
            "baseline_plan": {"exists": True, "ok": True},
            "baseline_run": {"complete": True},
            "baseline_eval": {"complete": True},
            "vocalset_manifest": {"exists": True},
            "vocalset_plan": {"exists": True, "ok": True},
            "vocalset_run": {"complete": True},
            "vocalset_eval": {"complete": True},
            "comparison": {
                "exists": True,
                "candidate_eligible": False,
                "failed_gates": ["clip_macro_f1_delta"],
                "unknown_gates": [],
                "evaluation_artifact_match": True,
            },
            "vocalset_balanced_manifest": {"exists": True},
            "vocalset_balanced_plan": {"exists": True, "ok": True},
            "vocalset_balanced_run": {"complete": True},
            "vocalset_balanced_eval": {"complete": True},
            "balanced_comparison": {
                "exists": True,
                "candidate_eligible": False,
                "failed_gates": ["expected_calibration_error_delta"],
                "unknown_gates": [],
                "evaluation_artifact_match": True,
            },
            "app_manifest": {"exists": True},
            "app_trainable_manifest": {"exists": True},
            "app_eval_only_manifest": {"exists": True},
            "app_train_manifest": {"exists": True},
            "app_val_manifest": {"exists": True},
            "app_eval_manifest": {"exists": True},
            "app_validation_audit": {
                "exists": True,
                "ready_for_mvp_validation": True,
                "manifest_match": True,
            },
            "app_baseline_eval": {"complete": False, "missing_artifacts": ["metrics.json"]},
            "app_adapted_train_manifest": {"exists": False},
            "app_adapted_plan": {"exists": False, "ok": False},
            "app_adapted_run": {"complete": False},
            "app_adapted_eval": {"complete": False, "missing_artifacts": []},
            "app_adapted_comparison": {"exists": False, "candidate_eligible": False},
        }

        actions = experiment_next_actions(report, commands)
        stage = experiment_current_stage(report)
        self.assertIn("evaluate", actions[0])
        self.assertIn(args.app_baseline_eval_dir, actions[0])
        self.assertEqual(stage["name"], "evaluate_app_baseline")

        report["app_baseline_eval"] = {"complete": True}
        actions = experiment_next_actions(report, commands)
        stage = experiment_current_stage(report)
        self.assertIn("merge_manifest", actions[0])
        self.assertEqual(stage["name"], "merge_app_adapted_train_manifest")

        report["app_adapted_train_manifest"]["exists"] = True
        actions = experiment_next_actions(report, commands)
        stage = experiment_current_stage(report)
        self.assertIn("plan_training", actions[0])
        self.assertEqual(stage["name"], "plan_app_adapted_candidate")

        report["app_adapted_plan"] = {"exists": True, "ok": True}
        actions = experiment_next_actions(report, commands)
        stage = experiment_current_stage(report)
        self.assertIn("gt_singer_grader.train", actions[0])
        self.assertEqual(stage["name"], "train_app_adapted_candidate")

        report["app_adapted_run"]["complete"] = True
        actions = experiment_next_actions(report, commands)
        stage = experiment_current_stage(report)
        self.assertIn("evaluate", actions[0])
        self.assertEqual(stage["name"], "evaluate_app_adapted_candidate")

        report["app_adapted_eval"] = {"complete": True}
        actions = experiment_next_actions(report, commands)
        stage = experiment_current_stage(report)
        self.assertIn("compare_runs", actions[0])
        self.assertEqual(stage["name"], "compare_app_adapted_candidates")

        report["app_adapted_comparison"] = {
            "exists": True,
            "candidate_eligible": True,
            "evaluation_artifact_match": True,
        }
        self.assertIn("package_candidate", experiment_next_actions(report, commands)[0])
        self.assertEqual(experiment_current_stage(report)["name"], "package_app_adapted_candidate")

    def test_experiment_status_reruns_comparison_when_candidate_is_missing(self) -> None:
        args = Namespace(
            gtsinger_root="./gt_singer_grader/data/GTSinger",
            baseline_plan="./gt_singer_grader/runs/gtsinger_speaker_aug_v1/training_plan.json",
            baseline_run_dir="./gt_singer_grader/runs/gtsinger_speaker_aug_v1",
            baseline_eval_dir="./gt_singer_grader/runs/gtsinger_speaker_aug_v1/eval_val",
            vocalset_root="./gt_singer_grader/data/VocalSet",
            vocalset_manifest="./gt_singer_grader/manifests/vocalset.jsonl",
            vocalset_plan="./gt_singer_grader/runs/gtsinger_vocalset_v1/training_plan.json",
            vocalset_run_dir="./gt_singer_grader/runs/gtsinger_vocalset_v1",
            vocalset_eval_dir="./gt_singer_grader/runs/gtsinger_vocalset_v1/eval_val",
            app_labels="./gt_singer_grader/data/app_recordings/review_labels.csv",
            checkpoint="./gt_singer_grader/models/technique_demo_best.pth",
            metadata="./gt_singer_grader/models/technique_demo_metadata.json",
            app_manifest="./gt_singer_grader/manifests/app_recordings.jsonl",
            app_trainable_manifest="./gt_singer_grader/manifests/app_recordings_trainable.jsonl",
            app_eval_only_manifest="./gt_singer_grader/manifests/app_recordings_eval_only.jsonl",
            app_val_manifest="./gt_singer_grader/manifests/app_recordings_val.jsonl",
            app_eval_manifest="./gt_singer_grader/manifests/app_recordings_eval.jsonl",
            app_validation_audit="./gt_singer_grader/manifests/app_recordings_eval_audit.json",
            comparison="./gt_singer_grader/runs/run_comparison.json",
        )
        report = {
            "preflight": {"ok": True},
            "baseline_plan": {"exists": True, "ok": True},
            "baseline_run": {"complete": True},
            "baseline_eval": {"complete": True},
            "vocalset_manifest": {"exists": True},
            "vocalset_plan": {"exists": True, "ok": True},
            "vocalset_run": {"complete": True},
            "vocalset_eval": {"complete": True},
            "comparison": {
                "path": "./gt_singer_grader/runs/run_comparison.json",
                "exists": True,
                "candidate_found": False,
                "candidate_eligible": False,
                "error": "candidate eval dir not found in comparison report",
            },
            "app_manifest": {"exists": False},
            "app_trainable_manifest": {"exists": False},
            "app_eval_only_manifest": {"exists": False},
            "app_val_manifest": {"exists": False},
            "app_eval_manifest": {"exists": False},
            "app_validation_audit": {"ready_for_mvp_validation": False},
        }

        actions = experiment_next_actions(report, experiment_command_list(args))
        stage = experiment_current_stage(report)

        self.assertIn("does not contain the configured candidate", actions[0])
        self.assertIn("compare_runs", actions[0])
        self.assertEqual(stage["name"], "comparison_candidate_missing")
        self.assertTrue(stage["ready"])
        self.assertIn("candidate eval dir not found", stage["detail"]["error"])

    def test_experiment_status_reruns_stale_comparison_before_app_pipeline(self) -> None:
        args = Namespace(
            gtsinger_root="./gt_singer_grader/data/GTSinger",
            baseline_plan="./gt_singer_grader/runs/gtsinger_speaker_aug_v1/training_plan.json",
            baseline_run_dir="./gt_singer_grader/runs/gtsinger_speaker_aug_v1",
            baseline_eval_dir="./gt_singer_grader/runs/gtsinger_speaker_aug_v1/eval_val",
            vocalset_root="./gt_singer_grader/data/VocalSet",
            vocalset_manifest="./gt_singer_grader/manifests/vocalset.jsonl",
            vocalset_plan="./gt_singer_grader/runs/gtsinger_vocalset_v1/training_plan.json",
            vocalset_run_dir="./gt_singer_grader/runs/gtsinger_vocalset_v1",
            vocalset_eval_dir="./gt_singer_grader/runs/gtsinger_vocalset_v1/eval_val",
            app_labels="./gt_singer_grader/data/app_recordings/review_labels.csv",
            checkpoint="./gt_singer_grader/models/technique_demo_best.pth",
            metadata="./gt_singer_grader/models/technique_demo_metadata.json",
            app_manifest="./gt_singer_grader/manifests/app_recordings.jsonl",
            app_trainable_manifest="./gt_singer_grader/manifests/app_recordings_trainable.jsonl",
            app_eval_only_manifest="./gt_singer_grader/manifests/app_recordings_eval_only.jsonl",
            app_val_manifest="./gt_singer_grader/manifests/app_recordings_val.jsonl",
            app_eval_manifest="./gt_singer_grader/manifests/app_recordings_eval.jsonl",
            app_validation_audit="./gt_singer_grader/manifests/app_recordings_eval_audit.json",
            comparison="./gt_singer_grader/runs/run_comparison.json",
        )
        report = {
            "preflight": {"ok": True},
            "baseline_plan": {"exists": True, "ok": True},
            "baseline_run": {"complete": True},
            "baseline_eval": {"complete": True},
            "vocalset_manifest": {"exists": True},
            "vocalset_plan": {"exists": True, "ok": True},
            "vocalset_run": {"complete": True},
            "vocalset_eval": {"complete": True},
            "comparison": {
                "exists": True,
                "candidate_eligible": False,
                "promotion_eligible": True,
                "evaluation_artifact_match": False,
                "evaluation_artifact_failed_checks": ["comparison.evaluation_artifact_sha256:metrics.json"],
                "failed_gates": [],
                "unknown_gates": [],
            },
            "app_manifest": {"exists": False},
            "app_trainable_manifest": {"exists": False},
            "app_eval_only_manifest": {"exists": False},
            "app_val_manifest": {"exists": False},
            "app_eval_manifest": {"exists": False},
            "app_validation_audit": {"ready_for_mvp_validation": False},
        }

        actions = experiment_next_actions(report, experiment_command_list(args))
        stage = experiment_current_stage(report)

        self.assertIn("Comparison report is stale", actions[0])
        self.assertIn("compare_runs", actions[0])
        self.assertEqual(stage["name"], "comparison_evaluation_artifacts_stale")
        self.assertTrue(stage["ready"])
        self.assertIn("comparison.evaluation_artifact_sha256:metrics.json", stage["detail"]["failed_checks"])

    def test_experiment_status_reruns_stale_app_validation_audit_before_packaging(self) -> None:
        args = Namespace(
            gtsinger_root="./gt_singer_grader/data/GTSinger",
            baseline_plan="./gt_singer_grader/runs/gtsinger_speaker_aug_v1/training_plan.json",
            baseline_run_dir="./gt_singer_grader/runs/gtsinger_speaker_aug_v1",
            baseline_eval_dir="./gt_singer_grader/runs/gtsinger_speaker_aug_v1/eval_val",
            vocalset_root="./gt_singer_grader/data/VocalSet",
            vocalset_manifest="./gt_singer_grader/manifests/vocalset.jsonl",
            vocalset_plan="./gt_singer_grader/runs/gtsinger_vocalset_v1/training_plan.json",
            vocalset_run_dir="./gt_singer_grader/runs/gtsinger_vocalset_v1",
            vocalset_eval_dir="./gt_singer_grader/runs/gtsinger_vocalset_v1/eval_val",
            app_labels="./gt_singer_grader/data/app_recordings/review_labels.csv",
            checkpoint="./gt_singer_grader/models/technique_demo_best.pth",
            metadata="./gt_singer_grader/models/technique_demo_metadata.json",
            app_manifest="./gt_singer_grader/manifests/app_recordings.jsonl",
            app_trainable_manifest="./gt_singer_grader/manifests/app_recordings_trainable.jsonl",
            app_eval_only_manifest="./gt_singer_grader/manifests/app_recordings_eval_only.jsonl",
            app_val_manifest="./gt_singer_grader/manifests/app_recordings_val.jsonl",
            app_eval_manifest="./gt_singer_grader/manifests/app_recordings_eval.jsonl",
            app_validation_audit="./gt_singer_grader/manifests/app_recordings_eval_audit.json",
            comparison="./gt_singer_grader/runs/run_comparison.json",
        )
        report = {
            "preflight": {"ok": True},
            "baseline_plan": {"exists": True, "ok": True},
            "baseline_run": {"complete": True},
            "baseline_eval": {"complete": True},
            "vocalset_manifest": {"exists": True},
            "vocalset_plan": {"exists": True, "ok": True},
            "vocalset_run": {"complete": True},
            "vocalset_eval": {"complete": True},
            "comparison": {"exists": True, "candidate_eligible": True, "evaluation_artifact_match": True},
            "app_manifest": {"exists": True},
            "app_trainable_manifest": {"exists": True},
            "app_eval_only_manifest": {"exists": True},
            "app_val_manifest": {"exists": True},
            "app_eval_manifest": {"exists": True},
            "app_validation_audit": {
                "exists": True,
                "ready_for_mvp_validation": False,
                "audit_ready": True,
                "manifest_match": False,
                "manifest_failed_checks": ["app_validation_audit.manifest:sha256"],
            },
        }

        actions = experiment_next_actions(report, experiment_command_list(args))
        stage = experiment_current_stage(report)

        self.assertIn("App validation audit is stale", actions[0])
        self.assertIn("audit_app_validation", actions[0])
        self.assertEqual(stage["name"], "app_validation_audit_stale")
        self.assertTrue(stage["ready"])
        self.assertIn("app_validation_audit.manifest:sha256", stage["detail"]["failed_checks"])

    def test_experiment_status_requests_app_labels_before_building_app_manifest(self) -> None:
        args = Namespace(
            gtsinger_root="./gt_singer_grader/data/GTSinger",
            baseline_plan="./gt_singer_grader/runs/gtsinger_speaker_aug_v1/training_plan.json",
            baseline_run_dir="./gt_singer_grader/runs/gtsinger_speaker_aug_v1",
            baseline_eval_dir="./gt_singer_grader/runs/gtsinger_speaker_aug_v1/eval_val",
            vocalset_root="./gt_singer_grader/data/VocalSet",
            vocalset_manifest="./gt_singer_grader/manifests/vocalset.jsonl",
            vocalset_plan="./gt_singer_grader/runs/gtsinger_vocalset_v1/training_plan.json",
            vocalset_run_dir="./gt_singer_grader/runs/gtsinger_vocalset_v1",
            vocalset_eval_dir="./gt_singer_grader/runs/gtsinger_vocalset_v1/eval_val",
            app_labels="./gt_singer_grader/data/app_recordings/review_labels.csv",
            checkpoint="./gt_singer_grader/models/technique_demo_best.pth",
            metadata="./gt_singer_grader/models/technique_demo_metadata.json",
            app_manifest="./gt_singer_grader/manifests/app_recordings.jsonl",
            app_trainable_manifest="./gt_singer_grader/manifests/app_recordings_trainable.jsonl",
            app_eval_only_manifest="./gt_singer_grader/manifests/app_recordings_eval_only.jsonl",
            app_val_manifest="./gt_singer_grader/manifests/app_recordings_val.jsonl",
            app_eval_manifest="./gt_singer_grader/manifests/app_recordings_eval.jsonl",
            app_validation_audit="./gt_singer_grader/manifests/app_recordings_eval_audit.json",
            comparison="./gt_singer_grader/runs/run_comparison.json",
        )
        report = {
            "preflight": {
                "ok": True,
                "checks": [
                    {
                        "name": "app_recording_labels",
                        "ok": False,
                        "detail": "./gt_singer_grader/data/app_recordings/review_labels.csv",
                    }
                ],
            },
            "baseline_plan": {"exists": True, "ok": True},
            "baseline_run": {"complete": True},
            "baseline_eval": {"complete": True},
            "vocalset_manifest": {"exists": True},
            "vocalset_plan": {"exists": True, "ok": True},
            "vocalset_run": {"complete": True},
            "vocalset_eval": {"complete": True},
            "comparison": {"exists": True, "candidate_eligible": True},
            "app_manifest": {"exists": False},
            "app_trainable_manifest": {"exists": False},
            "app_eval_only_manifest": {"exists": False},
            "app_val_manifest": {"exists": False},
            "app_eval_manifest": {"exists": False},
            "app_validation_audit": {"ready_for_mvp_validation": False},
        }

        actions = experiment_next_actions(report, experiment_command_list(args))
        stage = experiment_current_stage(report)

        self.assertIn("plan_app_collection", actions[0])
        self.assertIn("--output-csv", actions[0])
        self.assertEqual(stage["name"], "plan_app_recording_collection")
        self.assertTrue(stage["ready"])

    def test_experiment_status_app_label_request_includes_collection_plan_size(self) -> None:
        args = Namespace(
            gtsinger_root="./gt_singer_grader/data/GTSinger",
            baseline_plan="./gt_singer_grader/runs/gtsinger_speaker_aug_v1/training_plan.json",
            baseline_run_dir="./gt_singer_grader/runs/gtsinger_speaker_aug_v1",
            baseline_eval_dir="./gt_singer_grader/runs/gtsinger_speaker_aug_v1/eval_val",
            vocalset_root="./gt_singer_grader/data/VocalSet",
            vocalset_manifest="./gt_singer_grader/manifests/vocalset.jsonl",
            vocalset_plan="./gt_singer_grader/runs/gtsinger_vocalset_v1/training_plan.json",
            vocalset_run_dir="./gt_singer_grader/runs/gtsinger_vocalset_v1",
            vocalset_eval_dir="./gt_singer_grader/runs/gtsinger_vocalset_v1/eval_val",
            app_labels="./gt_singer_grader/data/app_recordings/review_labels.csv",
            checkpoint="./gt_singer_grader/models/technique_demo_best.pth",
            metadata="./gt_singer_grader/models/technique_demo_metadata.json",
            app_manifest="./gt_singer_grader/manifests/app_recordings.jsonl",
            app_trainable_manifest="./gt_singer_grader/manifests/app_recordings_trainable.jsonl",
            app_eval_only_manifest="./gt_singer_grader/manifests/app_recordings_eval_only.jsonl",
            app_val_manifest="./gt_singer_grader/manifests/app_recordings_val.jsonl",
            app_eval_manifest="./gt_singer_grader/manifests/app_recordings_eval.jsonl",
            app_validation_audit="./gt_singer_grader/manifests/app_recordings_eval_audit.json",
            comparison="./gt_singer_grader/runs/run_comparison.json",
        )
        report = {
            "preflight": {
                "ok": True,
                "checks": [
                    {
                        "name": "app_recording_labels",
                        "ok": False,
                        "detail": "./gt_singer_grader/data/app_recordings/review_labels.csv",
                    }
                ],
            },
            "baseline_plan": {"exists": True, "ok": True},
            "baseline_run": {"complete": True},
            "baseline_eval": {"complete": True},
            "vocalset_manifest": {"exists": True},
            "vocalset_plan": {"exists": True, "ok": True},
            "vocalset_run": {"complete": True},
            "vocalset_eval": {"complete": True},
            "comparison": {"exists": True, "candidate_eligible": True},
            "app_collection_plan": {"exists": True, "planned_records": 140, "planned_groups": 20},
            "app_prepare_report": {"exists": False},
            "app_manifest": {"exists": False},
            "app_trainable_manifest": {"exists": False},
            "app_eval_only_manifest": {"exists": False},
            "app_val_manifest": {"exists": False},
            "app_eval_manifest": {"exists": False},
            "app_validation_audit": {"ready_for_mvp_validation": False},
        }

        actions = experiment_next_actions(report, experiment_command_list(args))

        self.assertIn("140 clips across 20 singer groups", actions[0])
        self.assertIn("materialize_app_collection", actions[0])

    def test_experiment_status_requests_collection_packet_after_materialization(self) -> None:
        args = Namespace(
            gtsinger_root="./gt_singer_grader/data/GTSinger",
            baseline_plan="./gt_singer_grader/runs/gtsinger_speaker_aug_v1/training_plan.json",
            baseline_run_dir="./gt_singer_grader/runs/gtsinger_speaker_aug_v1",
            baseline_eval_dir="./gt_singer_grader/runs/gtsinger_speaker_aug_v1/eval_val",
            vocalset_root="./gt_singer_grader/data/VocalSet",
            vocalset_manifest="./gt_singer_grader/manifests/vocalset.jsonl",
            vocalset_plan="./gt_singer_grader/runs/gtsinger_vocalset_v1/training_plan.json",
            vocalset_run_dir="./gt_singer_grader/runs/gtsinger_vocalset_v1",
            vocalset_eval_dir="./gt_singer_grader/runs/gtsinger_vocalset_v1/eval_val",
            app_labels="./gt_singer_grader/data/app_recordings/review_labels.csv",
            checkpoint="./gt_singer_grader/models/technique_demo_best.pth",
            metadata="./gt_singer_grader/models/technique_demo_metadata.json",
            app_manifest="./gt_singer_grader/manifests/app_recordings.jsonl",
            app_trainable_manifest="./gt_singer_grader/manifests/app_recordings_trainable.jsonl",
            app_eval_only_manifest="./gt_singer_grader/manifests/app_recordings_eval_only.jsonl",
            app_val_manifest="./gt_singer_grader/manifests/app_recordings_val.jsonl",
            app_eval_manifest="./gt_singer_grader/manifests/app_recordings_eval.jsonl",
            app_validation_audit="./gt_singer_grader/manifests/app_recordings_eval_audit.json",
            comparison="./gt_singer_grader/runs/run_comparison.json",
        )
        report = {
            "preflight": {
                "ok": True,
                "checks": [
                    {
                        "name": "app_recording_labels",
                        "ok": False,
                        "detail": "./gt_singer_grader/data/app_recordings/review_labels.csv",
                    }
                ],
            },
            "baseline_plan": {"exists": True, "ok": True},
            "baseline_run": {"complete": True},
            "baseline_eval": {"complete": True},
            "vocalset_manifest": {"exists": True},
            "vocalset_plan": {"exists": True, "ok": True},
            "vocalset_run": {"complete": True},
            "vocalset_eval": {"complete": True},
            "comparison": {"exists": True, "candidate_eligible": True},
            "app_collection_plan": {"exists": True, "planned_records": 140, "planned_groups": 20},
            "app_collection_materialize": {
                "checklist_exists": True,
                "checklist_path": "./gt_singer_grader/data/app_recordings/collection_checklist.csv",
                "report_exists": True,
                "existing_audio_files": 0,
                "missing_audio_files": 140,
                "missing_csv": "./gt_singer_grader/data/app_recordings/collection_missing.csv",
            },
            "app_collection_packet": {
                "exists": False,
                "summary_exists": False,
                "sheet_count": 0,
                "ok": False,
            },
            "app_prepare_report": {"exists": False},
            "app_manifest": {"exists": False},
            "app_trainable_manifest": {"exists": False},
            "app_eval_only_manifest": {"exists": False},
            "app_val_manifest": {"exists": False},
            "app_eval_manifest": {"exists": False},
            "app_validation_audit": {"ready_for_mvp_validation": False},
        }

        actions = experiment_next_actions(report, experiment_command_list(args))
        stage = experiment_current_stage(report)

        self.assertIn("140 missing audio files", actions[0])
        self.assertIn("export_app_collection_packet", actions[0])
        self.assertEqual(stage["name"], "export_app_collection_packet")
        self.assertTrue(stage["ready"])
        self.assertEqual(stage["detail"]["collection_packet"]["sheet_count"], 0)

    def test_experiment_status_app_label_request_includes_materialized_collection_progress(self) -> None:
        args = Namespace(
            gtsinger_root="./gt_singer_grader/data/GTSinger",
            baseline_plan="./gt_singer_grader/runs/gtsinger_speaker_aug_v1/training_plan.json",
            baseline_run_dir="./gt_singer_grader/runs/gtsinger_speaker_aug_v1",
            baseline_eval_dir="./gt_singer_grader/runs/gtsinger_speaker_aug_v1/eval_val",
            vocalset_root="./gt_singer_grader/data/VocalSet",
            vocalset_manifest="./gt_singer_grader/manifests/vocalset.jsonl",
            vocalset_plan="./gt_singer_grader/runs/gtsinger_vocalset_v1/training_plan.json",
            vocalset_run_dir="./gt_singer_grader/runs/gtsinger_vocalset_v1",
            vocalset_eval_dir="./gt_singer_grader/runs/gtsinger_vocalset_v1/eval_val",
            app_labels="./gt_singer_grader/data/app_recordings/review_labels.csv",
            checkpoint="./gt_singer_grader/models/technique_demo_best.pth",
            metadata="./gt_singer_grader/models/technique_demo_metadata.json",
            app_manifest="./gt_singer_grader/manifests/app_recordings.jsonl",
            app_trainable_manifest="./gt_singer_grader/manifests/app_recordings_trainable.jsonl",
            app_eval_only_manifest="./gt_singer_grader/manifests/app_recordings_eval_only.jsonl",
            app_val_manifest="./gt_singer_grader/manifests/app_recordings_val.jsonl",
            app_eval_manifest="./gt_singer_grader/manifests/app_recordings_eval.jsonl",
            app_validation_audit="./gt_singer_grader/manifests/app_recordings_eval_audit.json",
            comparison="./gt_singer_grader/runs/run_comparison.json",
        )
        report = {
            "preflight": {
                "ok": True,
                "checks": [
                    {
                        "name": "app_recording_labels",
                        "ok": False,
                        "detail": "./gt_singer_grader/data/app_recordings/review_labels.csv",
                    }
                ],
            },
            "baseline_plan": {"exists": True, "ok": True},
            "baseline_run": {"complete": True},
            "baseline_eval": {"complete": True},
            "vocalset_manifest": {"exists": True},
            "vocalset_plan": {"exists": True, "ok": True},
            "vocalset_run": {"complete": True},
            "vocalset_eval": {"complete": True},
            "comparison": {"exists": True, "candidate_eligible": True},
            "app_collection_plan": {"exists": True, "planned_records": 140, "planned_groups": 20},
            "app_collection_materialize": {
                "checklist_exists": True,
                "checklist_path": "./gt_singer_grader/data/app_recordings/collection_checklist.csv",
                "report_exists": True,
                "existing_audio_files": 0,
                "missing_audio_files": 140,
                "missing_csv": "./gt_singer_grader/data/app_recordings/collection_missing.csv",
            },
            "app_collection_packet": {
                "exists": True,
                "summary_exists": True,
                "index_exists": True,
                "checklist_match": True,
                "sheet_count": 20,
                "ok": True,
            },
            "app_prepare_report": {"exists": False},
            "app_manifest": {"exists": False},
            "app_trainable_manifest": {"exists": False},
            "app_eval_only_manifest": {"exists": False},
            "app_val_manifest": {"exists": False},
            "app_eval_manifest": {"exists": False},
            "app_validation_audit": {"ready_for_mvp_validation": False},
        }

        actions = experiment_next_actions(report, experiment_command_list(args))
        stage = experiment_current_stage(report)

        self.assertIn("140 clips across 20 singer groups", actions[0])
        self.assertIn("140 missing audio files", actions[0])
        self.assertIn("collection_checklist.csv", actions[0])
        self.assertIn("collection_missing.csv", actions[0])
        self.assertIn("--require-audio-files", actions[0])
        self.assertIn("--validate-wav-files", actions[0])
        self.assertNotIn("export_app_collection_packet", actions[0])
        self.assertEqual(stage["name"], "collect_or_fix_app_recording_labels")
        self.assertEqual(stage["detail"]["collection_materialize"]["missing_audio_files"], 140)

    def test_experiment_status_surfaces_prepare_report_gaps_before_app_labels(self) -> None:
        args = Namespace(
            gtsinger_root="./gt_singer_grader/data/GTSinger",
            baseline_plan="./gt_singer_grader/runs/gtsinger_speaker_aug_v1/training_plan.json",
            baseline_run_dir="./gt_singer_grader/runs/gtsinger_speaker_aug_v1",
            baseline_eval_dir="./gt_singer_grader/runs/gtsinger_speaker_aug_v1/eval_val",
            vocalset_root="./gt_singer_grader/data/VocalSet",
            vocalset_manifest="./gt_singer_grader/manifests/vocalset.jsonl",
            vocalset_plan="./gt_singer_grader/runs/gtsinger_vocalset_v1/training_plan.json",
            vocalset_run_dir="./gt_singer_grader/runs/gtsinger_vocalset_v1",
            vocalset_eval_dir="./gt_singer_grader/runs/gtsinger_vocalset_v1/eval_val",
            app_labels="./gt_singer_grader/data/app_recordings/review_labels.csv",
            app_prepare_report="./gt_singer_grader/data/app_recordings/prepare_report.json",
            checkpoint="./gt_singer_grader/models/technique_demo_best.pth",
            metadata="./gt_singer_grader/models/technique_demo_metadata.json",
            app_manifest="./gt_singer_grader/manifests/app_recordings.jsonl",
            app_trainable_manifest="./gt_singer_grader/manifests/app_recordings_trainable.jsonl",
            app_eval_only_manifest="./gt_singer_grader/manifests/app_recordings_eval_only.jsonl",
            app_val_manifest="./gt_singer_grader/manifests/app_recordings_val.jsonl",
            app_eval_manifest="./gt_singer_grader/manifests/app_recordings_eval.jsonl",
            app_validation_audit="./gt_singer_grader/manifests/app_recordings_eval_audit.json",
            comparison="./gt_singer_grader/runs/run_comparison.json",
        )
        report = {
            "preflight": {
                "ok": True,
                "checks": [
                    {
                        "name": "app_recording_labels",
                        "ok": False,
                        "detail": "./gt_singer_grader/data/app_recordings/review_labels.csv",
                    }
                ],
            },
            "baseline_plan": {"exists": True, "ok": True},
            "baseline_run": {"complete": True},
            "baseline_eval": {"complete": True},
            "vocalset_manifest": {"exists": True},
            "vocalset_plan": {"exists": True, "ok": True},
            "vocalset_run": {"complete": True},
            "vocalset_eval": {"complete": True},
            "comparison": {"exists": True, "candidate_eligible": True},
            "app_collection_plan": {"exists": True},
            "app_prepare_report": {
                "exists": True,
                "collection_plan_fully_matched": False,
                "missing_collection_plan_suggestions": ["raw/singer_002/breathy.wav"],
                "unplanned_audio_paths": ["raw/extra/take.wav"],
            },
            "app_manifest": {"exists": False},
            "app_trainable_manifest": {"exists": False},
            "app_eval_only_manifest": {"exists": False},
            "app_val_manifest": {"exists": False},
            "app_eval_manifest": {"exists": False},
            "app_validation_audit": {"ready_for_mvp_validation": False},
        }

        actions = experiment_next_actions(report, experiment_command_list(args))
        stage = experiment_current_stage(report)

        self.assertIn("collection-plan mismatches", actions[0])
        self.assertIn("prepare_app_recordings", actions[0])
        self.assertEqual(stage["name"], "collect_or_fix_app_recording_labels")
        self.assertEqual(stage["detail"]["prepare_report"]["missing_collection_plan_suggestions"], ["raw/singer_002/breathy.wav"])

    def test_experiment_status_requires_app_label_coverage_before_manifest_build(self) -> None:
        args = Namespace(
            gtsinger_root="./gt_singer_grader/data/GTSinger",
            baseline_plan="./gt_singer_grader/runs/gtsinger_speaker_aug_v1/training_plan.json",
            baseline_run_dir="./gt_singer_grader/runs/gtsinger_speaker_aug_v1",
            baseline_eval_dir="./gt_singer_grader/runs/gtsinger_speaker_aug_v1/eval_val",
            vocalset_root="./gt_singer_grader/data/VocalSet",
            vocalset_manifest="./gt_singer_grader/manifests/vocalset.jsonl",
            vocalset_plan="./gt_singer_grader/runs/gtsinger_vocalset_v1/training_plan.json",
            vocalset_run_dir="./gt_singer_grader/runs/gtsinger_vocalset_v1",
            vocalset_eval_dir="./gt_singer_grader/runs/gtsinger_vocalset_v1/eval_val",
            app_labels="./gt_singer_grader/data/app_recordings/review_labels.csv",
            app_audio_root=".",
            checkpoint="./gt_singer_grader/models/technique_demo_best.pth",
            metadata="./gt_singer_grader/models/technique_demo_metadata.json",
            app_manifest="./gt_singer_grader/manifests/app_recordings.jsonl",
            app_trainable_manifest="./gt_singer_grader/manifests/app_recordings_trainable.jsonl",
            app_eval_only_manifest="./gt_singer_grader/manifests/app_recordings_eval_only.jsonl",
            app_val_manifest="./gt_singer_grader/manifests/app_recordings_val.jsonl",
            app_eval_manifest="./gt_singer_grader/manifests/app_recordings_eval.jsonl",
            app_validation_audit="./gt_singer_grader/manifests/app_recordings_eval_audit.json",
            comparison="./gt_singer_grader/runs/run_comparison.json",
        )
        report = {
            "preflight": {
                "ok": True,
                "checks": [
                    {
                        "name": "app_recording_labels",
                        "ok": True,
                        "detail": {"coverage": {"ready_for_collection_target": True}},
                    }
                ],
            },
            "baseline_plan": {"exists": True, "ok": True},
            "baseline_run": {"complete": True},
            "baseline_eval": {"complete": True},
            "vocalset_manifest": {"exists": True},
            "vocalset_plan": {"exists": True, "ok": True},
            "vocalset_run": {"complete": True},
            "vocalset_eval": {"complete": True},
            "comparison": {"exists": True, "candidate_eligible": True},
            "app_label_coverage": {
                "ready_for_collection_target": False,
                "missing_audio_file_count": 1,
                "missing_target_families": {"vibrato": 2},
                "negative_shortfall": 1,
                "group_shortfall": 0,
                "unlabeled_records": 1,
                "intended_family_mismatch_count": 1,
                "missing_reviewer_id_count": 1,
                "warnings": ["review CSV references missing audio files"],
            },
            "app_manifest": {"exists": False},
            "app_trainable_manifest": {"exists": False},
            "app_eval_only_manifest": {"exists": False},
            "app_val_manifest": {"exists": False},
            "app_eval_manifest": {"exists": False},
            "app_validation_audit": {"ready_for_mvp_validation": False},
        }

        commands = experiment_command_list(args)
        actions = experiment_next_actions(report, commands)
        stage = experiment_current_stage(report)

        self.assertIn("plan_app_collection", actions[0])
        self.assertIn("missing target families {vibrato:2}", actions[0])
        self.assertIn("negative shortfall 1", actions[0])
        self.assertIn("1 unlabeled rows", actions[0])
        self.assertIn("1 missing audio files", actions[0])
        self.assertIn("1 intended-family mismatches", actions[0])
        self.assertIn("1 labeled rows missing reviewer_id", actions[0])
        self.assertIn("app_label_coverage", actions[1])
        self.assertIn("--require-audio-files", actions[1])
        self.assertIn("--strict", actions[1])
        self.assertEqual(stage["name"], "fix_app_label_coverage")
        self.assertFalse(stage["ready"])
        self.assertEqual(stage["detail"]["missing_audio_file_count"], 1)
        self.assertEqual(stage["detail"]["intended_family_mismatch_count"], 1)
        self.assertEqual(stage["detail"]["missing_reviewer_id_count"], 1)

    def test_package_candidate_requires_promotion_eligible_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "best.pth"
            checkpoint.write_bytes(b"checkpoint")
            eval_dir = root / "gtsinger_app_adapted_v1" / "eval_app"
            artifacts = write_eval_artifacts(eval_dir)
            comparison = root / "comparison.json"
            audit = root / "app_recordings_eval_audit.json"
            write_app_validation_audit(audit, manifest_path=artifacts["manifest"], records=42)
            write_app_domain_comparison(
                comparison,
                eval_dir,
                app_manifest=artifacts["manifest"],
                candidate_metrics={"top2_accuracy": 0.7},
            )
            output_checkpoint = root / "models" / "technique_demo_best.pth"
            metadata_path = root / "models" / "technique_demo_metadata.json"

            metadata = package_candidate(
                checkpoint=str(checkpoint),
                comparison_path=str(comparison),
                candidate_eval_dir=str(eval_dir),
                output_checkpoint=str(output_checkpoint),
                metadata_path=str(metadata_path),
                app_validation_audit_path=str(audit),
            )
            packaged_bytes = output_checkpoint.read_bytes()
            saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            source_checkpoint_sha256 = sha256_file(checkpoint)
            packaged_checkpoint_sha256 = sha256_file(output_checkpoint)
            comparison_report_sha256 = sha256_file(comparison)
            app_validation_audit_sha256 = sha256_file(audit)
            verification = verify_metadata(saved_metadata)

        self.assertEqual(packaged_bytes, b"checkpoint")
        self.assertTrue(verification["ok"])
        self.assertEqual(verification["failed_checks"], [])
        self.assertTrue(metadata["promotion"]["eligible"])
        self.assertTrue(saved_metadata["promotion"]["eligible"])
        self.assertEqual(saved_metadata["source_checkpoint_sha256"], source_checkpoint_sha256)
        self.assertEqual(saved_metadata["packaged_checkpoint_sha256"], packaged_checkpoint_sha256)
        self.assertEqual(saved_metadata["comparison_report_sha256"], comparison_report_sha256)
        self.assertEqual(saved_metadata["app_validation_audit_sha256"], app_validation_audit_sha256)
        self.assertEqual(saved_metadata["metrics"]["top2_accuracy"], 0.7)
        self.assertEqual(saved_metadata["gates"]["absolute"]["top2_accuracy"], 0.6)
        self.assertEqual(saved_metadata["app_validation_audit"]["records"], 42)
        self.assertEqual(saved_metadata["app_validation_audit_report"], str(audit))
        self.assertTrue(saved_metadata["app_validation_manifest"]["ok"])
        self.assertTrue(saved_metadata["app_domain_comparison"]["ok"])
        self.assertIn("gtsinger_song_aug_v1", saved_metadata["app_domain_comparison"]["baseline_path"])
        self.assertTrue(saved_metadata["evaluation_verification"]["ok"])
        self.assertTrue(saved_metadata["evaluated_checkpoint"]["ok"])
        self.assertEqual(saved_metadata["model_contract"]["name"], "nanopitch_technique_detector")
        self.assertEqual(saved_metadata["model_contract"]["families"], list(FAMILY_NAMES))
        self.assertEqual(saved_metadata["model_contract"]["techniques"], list(TECHNIQUE_KEYS))
        self.assertEqual(saved_metadata["model_contract"]["runtime_response"], "axis_result")
        self.assertEqual(sorted(saved_metadata["evaluation_artifact_sha256"]), sorted(REQUIRED_EVALUATION_ARTIFACTS))
        self.assertEqual(saved_metadata["missing_evaluation_artifacts"], [])

    def test_package_candidate_rejects_non_app_adapted_candidate_for_product_packaging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "best.pth"
            checkpoint.write_bytes(b"checkpoint")
            eval_dir = root / "gtsinger_vocalset_song_v1" / "eval_val"
            write_eval_artifacts(eval_dir)
            comparison = root / "comparison.json"
            comparison.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "path": str(eval_dir),
                                "evaluation_artifact_sha256": eval_artifact_hashes(eval_dir),
                                "promotion": {
                                    "eligible": True,
                                    "failed_gates": [],
                                    "unknown_gates": [],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            audit = root / "app_recordings_eval_audit.json"
            write_app_validation_audit(audit)

            with self.assertRaises(SystemExit) as exc:
                package_candidate(
                    checkpoint=str(checkpoint),
                    comparison_path=str(comparison),
                    candidate_eval_dir=str(eval_dir),
                    output_checkpoint=str(root / "models" / "technique_demo_best.pth"),
                    metadata_path=str(root / "models" / "technique_demo_metadata.json"),
                    app_validation_audit_path=str(audit),
                )

        self.assertIn("requires an app-adapted candidate", str(exc.exception))
        self.assertIn("candidate_kind=vocalset", str(exc.exception))

    def test_candidate_kind_does_not_treat_app_eval_as_app_adapted(self) -> None:
        self.assertEqual(candidate_kind("runs/gtsinger_app_adapted_v1/eval_app"), "app_adapted")
        self.assertEqual(candidate_kind("runs/gtsinger_song_aug_v1/eval_app"), "unknown")

    def test_package_candidate_rejects_baseline_app_eval_for_product_packaging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "best.pth"
            checkpoint.write_bytes(b"checkpoint")
            eval_dir = root / "gtsinger_song_aug_v1" / "eval_app"
            write_eval_artifacts(eval_dir)
            comparison = root / "comparison.json"
            comparison.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "path": str(eval_dir),
                                "evaluation_artifact_sha256": eval_artifact_hashes(eval_dir),
                                "promotion": {
                                    "eligible": True,
                                    "failed_gates": [],
                                    "unknown_gates": [],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            audit = root / "app_recordings_eval_audit.json"
            write_app_validation_audit(audit)

            with self.assertRaises(SystemExit) as exc:
                package_candidate(
                    checkpoint=str(checkpoint),
                    comparison_path=str(comparison),
                    candidate_eval_dir=str(eval_dir),
                    output_checkpoint=str(root / "models" / "technique_demo_best.pth"),
                    metadata_path=str(root / "models" / "technique_demo_metadata.json"),
                    app_validation_audit_path=str(audit),
                )

        self.assertIn("requires an app-adapted candidate", str(exc.exception))
        self.assertIn("candidate_kind=unknown", str(exc.exception))

    def test_verify_package_reports_tampered_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "best.pth"
            checkpoint.write_bytes(b"checkpoint")
            eval_dir = root / "gtsinger_app_adapted_v1" / "eval_app"
            artifacts = write_eval_artifacts(eval_dir)
            comparison = root / "comparison.json"
            audit = root / "app_recordings_eval_audit.json"
            write_app_validation_audit(audit, manifest_path=artifacts["manifest"])
            write_app_domain_comparison(comparison, eval_dir, app_manifest=artifacts["manifest"])
            output_checkpoint = root / "models" / "technique_demo_best.pth"
            metadata = package_candidate(
                checkpoint=str(checkpoint),
                comparison_path=str(comparison),
                candidate_eval_dir=str(eval_dir),
                output_checkpoint=str(output_checkpoint),
                metadata_path=str(root / "models" / "technique_demo_metadata.json"),
                app_validation_audit_path=str(audit),
            )
            output_checkpoint.write_bytes(b"tampered")

            verification = verify_metadata(metadata)

        self.assertFalse(verification["ok"])
        self.assertIn("packaged_checkpoint", verification["failed_checks"])

    def test_verify_package_rechecks_packaged_checkpoint_against_evaluation_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "best.pth"
            checkpoint.write_bytes(b"checkpoint")
            eval_dir = root / "gtsinger_app_adapted_v1" / "eval_app"
            artifacts = write_eval_artifacts(eval_dir)
            comparison = root / "comparison.json"
            audit = root / "app_recordings_eval_audit.json"
            write_app_validation_audit(audit, manifest_path=artifacts["manifest"])
            write_app_domain_comparison(comparison, eval_dir, app_manifest=artifacts["manifest"])
            output_checkpoint = root / "models" / "technique_demo_best.pth"
            metadata = package_candidate(
                checkpoint=str(checkpoint),
                comparison_path=str(comparison),
                candidate_eval_dir=str(eval_dir),
                output_checkpoint=str(output_checkpoint),
                metadata_path=str(root / "models" / "technique_demo_metadata.json"),
                app_validation_audit_path=str(audit),
            )
            output_checkpoint.write_bytes(b"tampered")
            metadata["packaged_checkpoint_sha256"] = sha256_file(output_checkpoint)

            verification = verify_metadata(metadata)

        self.assertFalse(verification["ok"])
        self.assertNotIn("packaged_checkpoint", verification["failed_checks"])
        self.assertIn("packaged_checkpoint_matches_evaluation:sha256", verification["failed_checks"])

    def test_verify_package_reports_semantic_metadata_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "best.pth"
            checkpoint.write_bytes(b"checkpoint")
            eval_dir = root / "gtsinger_app_adapted_v1" / "eval_app"
            artifacts = write_eval_artifacts(eval_dir)
            comparison = root / "comparison.json"
            audit = root / "app_recordings_eval_audit.json"
            write_app_validation_audit(audit, manifest_path=artifacts["manifest"])
            write_app_domain_comparison(comparison, eval_dir, app_manifest=artifacts["manifest"])
            metadata = package_candidate(
                checkpoint=str(checkpoint),
                comparison_path=str(comparison),
                candidate_eval_dir=str(eval_dir),
                output_checkpoint=str(root / "models" / "technique_demo_best.pth"),
                metadata_path=str(root / "models" / "technique_demo_metadata.json"),
                app_validation_audit_path=str(audit),
            )
            metadata["promotion"] = {
                "eligible": False,
                "failed_gates": ["top2_accuracy"],
                "unknown_gates": [],
            }
            metadata["app_validation_audit"] = {
                "ready_for_mvp_validation": False,
                "warnings": ["target family coverage is below threshold"],
            }
            metadata["missing_evaluation_artifacts"] = ["predictions.csv"]
            metadata["evaluation_artifact_sha256"].pop("predictions.csv")
            metadata["evaluation_verification"] = {
                "ok": False,
                "failed_checks": ["evaluation_config.manifest:sha256"],
            }
            metadata["evaluated_checkpoint"] = {
                "ok": False,
                "failed_checks": ["packaged_checkpoint:evaluated_sha256"],
            }
            metadata["comparison_evaluation_artifacts"] = {
                "ok": False,
                "failed_checks": ["comparison.evaluation_artifact_sha256:metrics.json"],
            }
            metadata["app_domain_comparison"] = {
                "ok": False,
                "failed_checks": ["comparison.baseline:app_eval_dir"],
            }
            metadata["candidate_eval_dir"] = str(root / "missing_candidate_eval")

            verification = verify_metadata(metadata)

        self.assertFalse(verification["ok"])
        self.assertIn("promotion_eligible", verification["failed_checks"])
        self.assertIn("app_validation_audit_ready", verification["failed_checks"])
        self.assertIn("no_missing_evaluation_artifacts", verification["failed_checks"])
        self.assertIn("required_evaluation_artifacts", verification["failed_checks"])
        self.assertIn("evaluation_verification_ok", verification["failed_checks"])
        self.assertIn("evaluated_checkpoint_match", verification["failed_checks"])
        self.assertIn("comparison_evaluation_artifacts_match", verification["failed_checks"])
        self.assertIn("comparison_candidate_match", verification["failed_checks"])
        self.assertIn("app_domain_comparison_match", verification["failed_checks"])

    def test_verify_package_requires_app_domain_comparison_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "best.pth"
            checkpoint.write_bytes(b"checkpoint")
            eval_dir = root / "gtsinger_app_adapted_v1" / "eval_app"
            artifacts = write_eval_artifacts(eval_dir)
            comparison = root / "comparison.json"
            audit = root / "app_recordings_eval_audit.json"
            write_app_validation_audit(audit, manifest_path=artifacts["manifest"])
            write_app_domain_comparison(comparison, eval_dir, app_manifest=artifacts["manifest"])
            metadata = package_candidate(
                checkpoint=str(checkpoint),
                comparison_path=str(comparison),
                candidate_eval_dir=str(eval_dir),
                output_checkpoint=str(root / "models" / "technique_demo_best.pth"),
                metadata_path=str(root / "models" / "technique_demo_metadata.json"),
                app_validation_audit_path=str(audit),
            )
            metadata.pop("app_domain_comparison")

            verification = verify_metadata(metadata)

        self.assertFalse(verification["ok"])
        self.assertIn("app_domain_comparison_match", verification["failed_checks"])

    def test_package_candidate_requires_app_validation_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "best.pth"
            checkpoint.write_bytes(b"checkpoint")
            eval_dir = root / "gtsinger_app_adapted_v1" / "eval_app"
            write_eval_artifacts(eval_dir)
            comparison = root / "comparison.json"
            comparison.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "path": str(eval_dir),
                                "evaluation_artifact_sha256": eval_artifact_hashes(eval_dir),
                                "promotion": {
                                    "eligible": True,
                                    "failed_gates": [],
                                    "unknown_gates": [],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(SystemExit) as exc:
                package_candidate(
                    checkpoint=str(checkpoint),
                    comparison_path=str(comparison),
                    candidate_eval_dir=str(eval_dir),
                    output_checkpoint=str(root / "models" / "technique_demo_best.pth"),
                    metadata_path=str(root / "models" / "technique_demo_metadata.json"),
                )

        self.assertIn("app validation audit is required", str(exc.exception))

    def test_package_candidate_rejects_ineligible_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "best.pth"
            checkpoint.write_bytes(b"checkpoint")
            eval_dir = root / "gtsinger_app_adapted_v1" / "eval_app"
            write_eval_artifacts(eval_dir)
            comparison = root / "comparison.json"
            comparison.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "path": str(eval_dir),
                                "evaluation_artifact_sha256": eval_artifact_hashes(eval_dir),
                                "promotion": {
                                    "eligible": False,
                                    "failed_gates": ["control_false_positive_rate_delta"],
                                    "unknown_gates": [],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(SystemExit) as exc:
                package_candidate(
                    checkpoint=str(checkpoint),
                    comparison_path=str(comparison),
                    candidate_eval_dir=str(eval_dir),
                    output_checkpoint=str(root / "models" / "technique_demo_best.pth"),
                    metadata_path=str(root / "models" / "technique_demo_metadata.json"),
                )

        self.assertIn("not promotion-eligible", str(exc.exception))

    def test_package_candidate_rejects_missing_eval_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "best.pth"
            checkpoint.write_bytes(b"checkpoint")
            eval_dir = root / "gtsinger_app_adapted_v1" / "eval_app"
            eval_dir.mkdir(parents=True)
            comparison = root / "comparison.json"
            comparison.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "path": str(eval_dir),
                                "evaluation_artifact_sha256": eval_artifact_hashes(eval_dir),
                                "promotion": {
                                    "eligible": True,
                                    "failed_gates": [],
                                    "unknown_gates": [],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(SystemExit) as exc:
                package_candidate(
                    checkpoint=str(checkpoint),
                    comparison_path=str(comparison),
                    candidate_eval_dir=str(eval_dir),
                    output_checkpoint=str(root / "models" / "technique_demo_best.pth"),
                    metadata_path=str(root / "models" / "technique_demo_metadata.json"),
                )

        self.assertIn("missing required artifact", str(exc.exception))
        self.assertIn("metrics.json", str(exc.exception))

    def test_package_candidate_rejects_tampered_evaluation_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "best.pth"
            checkpoint.write_bytes(b"checkpoint")
            eval_dir = root / "gtsinger_app_adapted_v1" / "eval_app"
            artifacts = write_eval_artifacts(eval_dir)
            artifacts["manifest"].write_text("{\"changed\": true}\n", encoding="utf-8")
            comparison = root / "comparison.json"
            comparison.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "path": str(eval_dir),
                                "evaluation_artifact_sha256": eval_artifact_hashes(eval_dir),
                                "promotion": {
                                    "eligible": True,
                                    "failed_gates": [],
                                    "unknown_gates": [],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(SystemExit) as exc:
                package_candidate(
                    checkpoint=str(checkpoint),
                    comparison_path=str(comparison),
                    candidate_eval_dir=str(eval_dir),
                    output_checkpoint=str(root / "models" / "technique_demo_best.pth"),
                    metadata_path=str(root / "models" / "technique_demo_metadata.json"),
                )

        self.assertIn("failed provenance verification", str(exc.exception))
        self.assertIn("evaluation_config.manifest:sha256", str(exc.exception))

    def test_package_candidate_rejects_checkpoint_not_used_for_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "best.pth"
            checkpoint.write_bytes(b"different checkpoint")
            eval_dir = root / "gtsinger_app_adapted_v1" / "eval_app"
            write_eval_artifacts(eval_dir)
            comparison = root / "comparison.json"
            comparison.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "path": str(eval_dir),
                                "evaluation_artifact_sha256": eval_artifact_hashes(eval_dir),
                                "promotion": {
                                    "eligible": True,
                                    "failed_gates": [],
                                    "unknown_gates": [],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(SystemExit) as exc:
                package_candidate(
                    checkpoint=str(checkpoint),
                    comparison_path=str(comparison),
                    candidate_eval_dir=str(eval_dir),
                    output_checkpoint=str(root / "models" / "technique_demo_best.pth"),
                    metadata_path=str(root / "models" / "technique_demo_metadata.json"),
                )

        self.assertIn("checkpoint does not match", str(exc.exception))
        self.assertIn("packaged_checkpoint:evaluated_sha256", str(exc.exception))

    def test_package_candidate_rejects_stale_comparison_eval_artifact_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "best.pth"
            checkpoint.write_bytes(b"checkpoint")
            eval_dir = root / "gtsinger_app_adapted_v1" / "eval_app"
            write_eval_artifacts(eval_dir)
            comparison = root / "comparison.json"
            stale_hashes = eval_artifact_hashes(eval_dir)
            stale_hashes["metrics.json"] = "0" * 64
            comparison.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "path": str(eval_dir),
                                "evaluation_artifact_sha256": stale_hashes,
                                "promotion": {
                                    "eligible": True,
                                    "failed_gates": [],
                                    "unknown_gates": [],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(SystemExit) as exc:
                package_candidate(
                    checkpoint=str(checkpoint),
                    comparison_path=str(comparison),
                    candidate_eval_dir=str(eval_dir),
                    output_checkpoint=str(root / "models" / "technique_demo_best.pth"),
                    metadata_path=str(root / "models" / "technique_demo_metadata.json"),
                )

        self.assertIn("comparison report does not match", str(exc.exception))
        self.assertIn("comparison.evaluation_artifact_sha256:metrics.json", str(exc.exception))

    def test_package_candidate_rejects_non_app_baseline_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "best.pth"
            checkpoint.write_bytes(b"checkpoint")
            eval_dir = root / "gtsinger_app_adapted_v1" / "eval_app"
            artifacts = write_eval_artifacts(eval_dir)
            baseline_eval_dir = root / "gtsinger_song_aug_v1" / "eval_val"
            write_eval_artifacts(baseline_eval_dir, manifest_path=artifacts["manifest"])
            comparison = root / "comparison.json"
            comparison.write_text(
                json.dumps(
                    {
                        "baseline": {
                            "path": str(baseline_eval_dir),
                            "evaluation_artifact_sha256": eval_artifact_hashes(baseline_eval_dir),
                        },
                        "candidates": [
                            {
                                "path": str(eval_dir),
                                "evaluation_artifact_sha256": eval_artifact_hashes(eval_dir),
                                "promotion": {
                                    "eligible": True,
                                    "failed_gates": [],
                                    "unknown_gates": [],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            audit = root / "app_recordings_eval_audit.json"
            write_app_validation_audit(audit, manifest_path=artifacts["manifest"])

            with self.assertRaises(SystemExit) as exc:
                package_candidate(
                    checkpoint=str(checkpoint),
                    comparison_path=str(comparison),
                    candidate_eval_dir=str(eval_dir),
                    output_checkpoint=str(root / "models" / "technique_demo_best.pth"),
                    metadata_path=str(root / "models" / "technique_demo_metadata.json"),
                    app_validation_audit_path=str(audit),
                )

        self.assertIn("requires candidate and baseline app evaluations", str(exc.exception))
        self.assertIn("comparison.baseline:app_eval_dir", str(exc.exception))

    def test_package_candidate_rejects_eval_manifest_that_differs_from_app_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "best.pth"
            checkpoint.write_bytes(b"checkpoint")
            eval_dir = root / "gtsinger_app_adapted_v1" / "eval_app"
            write_eval_artifacts(eval_dir)
            audited_manifest = root / "app_recordings_eval.jsonl"
            audited_manifest.write_text("{\"recording_id\":\"app:1\"}\n", encoding="utf-8")
            comparison = root / "comparison.json"
            write_app_domain_comparison(comparison, eval_dir, app_manifest=audited_manifest)
            audit = root / "app_recordings_eval_audit.json"
            write_app_validation_audit(audit, manifest_path=audited_manifest)

            with self.assertRaises(SystemExit) as exc:
                package_candidate(
                    checkpoint=str(checkpoint),
                    comparison_path=str(comparison),
                    candidate_eval_dir=str(eval_dir),
                    output_checkpoint=str(root / "models" / "technique_demo_best.pth"),
                    metadata_path=str(root / "models" / "technique_demo_metadata.json"),
                    app_validation_audit_path=str(audit),
                )

        self.assertIn("audited app validation manifest", str(exc.exception))
        self.assertIn("comparison.candidate.manifest:sha256", str(exc.exception))

    def test_package_candidate_rejects_stale_app_validation_audit_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "best.pth"
            checkpoint.write_bytes(b"checkpoint")
            eval_dir = root / "gtsinger_app_adapted_v1" / "eval_app"
            write_eval_artifacts(eval_dir)
            comparison = root / "comparison.json"
            comparison.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "path": str(eval_dir),
                                "evaluation_artifact_sha256": eval_artifact_hashes(eval_dir),
                                "promotion": {
                                    "eligible": True,
                                    "failed_gates": [],
                                    "unknown_gates": [],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            audit = root / "app_recordings_eval_audit.json"
            manifest = write_app_validation_audit(audit)
            manifest.write_text("{\"changed\": true}\n", encoding="utf-8")

            with self.assertRaises(SystemExit) as exc:
                package_candidate(
                    checkpoint=str(checkpoint),
                    comparison_path=str(comparison),
                    candidate_eval_dir=str(eval_dir),
                    output_checkpoint=str(root / "models" / "technique_demo_best.pth"),
                    metadata_path=str(root / "models" / "technique_demo_metadata.json"),
                    app_validation_audit_path=str(audit),
                )

        self.assertIn("app validation audit does not match", str(exc.exception))
        self.assertIn("app_validation_audit.manifest:sha256", str(exc.exception))

    def test_package_candidate_rejects_failing_app_validation_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "best.pth"
            checkpoint.write_bytes(b"checkpoint")
            eval_dir = root / "gtsinger_app_adapted_v1" / "eval_app"
            write_eval_artifacts(eval_dir)
            comparison = root / "comparison.json"
            comparison.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "path": str(eval_dir),
                                "evaluation_artifact_sha256": eval_artifact_hashes(eval_dir),
                                "promotion": {
                                    "eligible": True,
                                    "failed_gates": [],
                                    "unknown_gates": [],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            audit = root / "app_recordings_eval_audit.json"
            write_app_validation_audit(
                audit,
                ready=False,
                warnings=["target family coverage is below threshold"],
            )

            with self.assertRaises(SystemExit) as exc:
                package_candidate(
                    checkpoint=str(checkpoint),
                    comparison_path=str(comparison),
                    candidate_eval_dir=str(eval_dir),
                    output_checkpoint=str(root / "models" / "technique_demo_best.pth"),
                    metadata_path=str(root / "models" / "technique_demo_metadata.json"),
                    app_validation_audit_path=str(audit),
                )

        self.assertIn("app validation audit is not ready", str(exc.exception))

    def test_downloader_missing_dependency_message_is_actionable(self) -> None:
        try:
            snapshot_download = load_snapshot_download()
        except SystemExit as exc:
            self.assertIn("requirements-training.txt", str(exc))
        else:
            self.assertTrue(callable(snapshot_download))

    def test_downloader_defaults_to_training_required_files(self) -> None:
        self.assertEqual(
            allow_patterns("English"),
            [
                "English/**/*.wav",
                "English/**/*.json",
            ],
        )
        self.assertEqual(allow_patterns("English", all_files=True), ["English/**"])
        self.assertEqual(ignore_patterns("English"), ["English/**/Paired_Speech_Group/**"])
        self.assertIsNone(ignore_patterns("English", include_speech=True))


if __name__ == "__main__":
    unittest.main()
