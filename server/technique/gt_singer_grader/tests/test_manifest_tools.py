from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from gt_singer_grader.build_manifest import build_app_recordings_manifest
from gt_singer_grader.manifest import (
    normalize_label_list,
    read_jsonl,
    validate_record,
    write_jsonl,
)


class ManifestToolsTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
