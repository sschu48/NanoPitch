from __future__ import annotations

import unittest

from api import build_axis_result, default_checkpoint
from gt_singer_grader.section_detection import aggregate_section_evidence, section_windows


class SectionDetectionTest(unittest.TestCase):
    def test_section_windows_overlap_and_cover_tail(self) -> None:
        windows = section_windows(total_frames=1200, window_frames=500, stride_frames=250)

        self.assertEqual(windows[0], (0, 500))
        self.assertEqual(windows[1], (250, 750))
        self.assertEqual(windows[-1], (700, 1200))
        self.assertEqual(windows, sorted(set(windows)))

    def test_section_windows_short_recording_uses_single_window(self) -> None:
        self.assertEqual(section_windows(total_frames=320, window_frames=500, stride_frames=250), [(0, 320)])

    def test_aggregate_requires_adjacent_evidence_for_multi_section_recording(self) -> None:
        sections = [
            {"primary_technique": "vibrato", "primary_technique_score": 0.62, "voiced_ratio": 0.8, "start_s": 0.0, "end_s": 5.0},
            {"primary_technique": "vibrato", "primary_technique_score": 0.57, "voiced_ratio": 0.7, "start_s": 2.5, "end_s": 7.5},
            {"primary_technique": "breathy", "primary_technique_score": 0.73, "voiced_ratio": 0.9, "start_s": 7.5, "end_s": 12.5},
        ]

        evidence = aggregate_section_evidence(sections)

        self.assertEqual(evidence[0]["technique"], "vibrato")
        self.assertTrue(evidence[0]["detected"])
        self.assertEqual(evidence[0]["adjacent_section_run"], 2)
        breathy = next(item for item in evidence if item["technique"] == "breathy")
        self.assertFalse(breathy["detected"])

    def test_aggregate_allows_single_short_section(self) -> None:
        evidence = aggregate_section_evidence(
            [
                {
                    "primary_technique": "breathy",
                    "primary_technique_score": 0.61,
                    "voiced_ratio": 0.8,
                    "start_s": 0.0,
                    "end_s": 3.2,
                }
            ]
        )

        self.assertEqual(evidence[0]["technique"], "breathy")
        self.assertTrue(evidence[0]["detected"])

    def test_axis_result_exposes_sections_without_changing_existing_fields(self) -> None:
        summary = {
            "detected_family": "vibrato",
            "detected_confidence": 0.7,
            "primary_technique": "vibrato",
            "primary_technique_score": 0.6,
            "voiced_ratio": 0.8,
            "family_margin": 0.2,
            "dominant_techniques": [{"technique": "vibrato", "score": 0.6}],
            "technique_scores": {"vibrato": 0.6},
            "family_probabilities": {"vibrato": 0.7},
            "technique_sections": [
                {
                    "start_s": 0.0,
                    "end_s": 5.0,
                    "primary_technique": "vibrato",
                    "primary_technique_score": 0.6,
                    "voiced_ratio": 0.8,
                }
            ],
            "section_technique_evidence": [
                {
                    "technique": "vibrato",
                    "detected": True,
                    "max_score": 0.6,
                }
            ],
            "section_detection_config": {
                "window_seconds": 5.0,
                "stride_seconds": 2.5,
                "model_context_seconds": 10.0,
            },
        }

        axis = build_axis_result(summary, {"status": "detected", "headline": "Detected vibrato"})

        self.assertEqual(axis["axis"], "technique")
        self.assertEqual(axis["metrics"]["detected_family"], "vibrato")
        self.assertEqual(axis["metrics"]["section_techniques"], {"vibrato": 0.6})
        self.assertEqual(axis["sections"], summary["technique_sections"])
        self.assertEqual(axis["section_detection_config"]["model_context_seconds"], 10.0)

    def test_default_checkpoint_stays_packaged_demo_model(self) -> None:
        self.assertEqual(default_checkpoint().name, "technique_demo_best.pth")
        self.assertEqual(default_checkpoint().parent.name, "models")


if __name__ == "__main__":
    unittest.main()
