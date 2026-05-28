"""Build normalized training manifests from supported technique datasets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from .constants import PRIMARY_FAMILY_TO_TECHNIQUES
from .manifest import normalize_label_list, validate_record, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a normalized technique-model manifest")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gt = subparsers.add_parser("gtsinger", help="Build a manifest from a GT Singer directory")
    gt.add_argument("--root", required=True, help="Path to downloaded GT Singer data")
    gt.add_argument("--language", default="English")
    gt.add_argument("--output", required=True)
    gt.add_argument("--include-speech", action="store_true")

    app = subparsers.add_parser("app-recordings", help="Build a manifest from a labeled CSV of app recordings")
    app.add_argument("--csv", required=True, help="CSV with audio_path plus family/technique labels")
    app.add_argument("--output", required=True)
    app.add_argument("--dataset-name", default="app_recordings")
    app.add_argument("--recording-domain", default="app_user")
    return parser.parse_args()


def _validate_records(records: list[dict[str, Any]]) -> None:
    errors: list[str] = []
    for index, record in enumerate(records, start=1):
        errors.extend(validate_record(record, line_number=index))
    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)


def build_gtsinger_manifest(root: str, language: str, include_speech: bool) -> list[dict[str, Any]]:
    from .data import scan_gt_singer

    records = scan_gt_singer(root, language=language, include_speech=include_speech)
    output: list[dict[str, Any]] = []
    for record in records:
        family = record.clip_label
        family_name = "control" if record.role in {"control", "speech"} else record.family
        techniques = list(PRIMARY_FAMILY_TO_TECHNIQUES.get(family_name, ()))
        output.append(
            {
                "recording_id": f"gtsinger:{record.speaker}:{record.technique_folder}:{record.song}:{record.stem}",
                "dataset": "gtsinger",
                "audio_path": record.wav_path,
                "json_path": record.json_path,
                "recording_domain": "studio",
                "label_source": "gtsinger_folder_and_alignment_json",
                "split_group": record.speaker,
                "speaker_id": record.speaker,
                "song_id": record.song,
                "labels": {
                    "families": [family_name],
                    "techniques": techniques,
                    "clip_role": record.role,
                },
                "metadata": {
                    "language": language,
                    "technique_folder": record.technique_folder,
                    "split_group_song": record.split_group,
                    "clip_label_index": int(family),
                },
            }
        )
    return output


def build_app_recordings_manifest(csv_path: str, dataset_name: str, recording_domain: str) -> list[dict[str, Any]]:
    """Build a manifest from simple user-label CSV exports.

    Required CSV columns:
      audio_path, families, techniques

    Optional columns:
      recording_id, singer_id, song_id, split_group, label_source, notes
    """
    output: list[dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"audio_path", "families", "techniques"} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"CSV missing required column(s): {', '.join(sorted(missing))}")

        for row_number, row in enumerate(reader, start=2):
            audio_path = (row.get("audio_path") or "").strip()
            recording_id = (row.get("recording_id") or "").strip() or f"{dataset_name}:{row_number - 1}:{Path(audio_path).stem}"
            singer_id = (row.get("singer_id") or "").strip()
            song_id = (row.get("song_id") or "").strip()
            split_group = (row.get("split_group") or "").strip() or singer_id or recording_id
            output.append(
                {
                    "recording_id": recording_id,
                    "dataset": dataset_name,
                    "audio_path": audio_path,
                    "recording_domain": recording_domain,
                    "label_source": (row.get("label_source") or "human_labeled_app_recording").strip(),
                    "split_group": split_group,
                    "speaker_id": singer_id,
                    "song_id": song_id,
                    "labels": {
                        "families": normalize_label_list(row.get("families")),
                        "techniques": normalize_label_list(row.get("techniques")),
                    },
                    "metadata": {
                        "notes": (row.get("notes") or "").strip(),
                    },
                }
            )
    return output


def main() -> None:
    args = parse_args()
    if args.command == "gtsinger":
        records = build_gtsinger_manifest(args.root, args.language, args.include_speech)
    elif args.command == "app-recordings":
        records = build_app_recordings_manifest(args.csv, args.dataset_name, args.recording_domain)
    else:
        raise SystemExit(f"unknown command: {args.command}")

    _validate_records(records)
    write_jsonl(args.output, records)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
