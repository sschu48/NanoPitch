"""Section-level technique detection helpers."""

from __future__ import annotations


def section_windows(total_frames: int, window_frames: int, stride_frames: int) -> list[tuple[int, int]]:
    if total_frames <= 0:
        return [(0, 1)]
    if window_frames <= 0:
        raise ValueError("window_frames must be positive")
    if stride_frames <= 0:
        raise ValueError("stride_frames must be positive")
    if total_frames <= window_frames:
        return [(0, total_frames)]

    windows: list[tuple[int, int]] = []
    start = 0
    while start + window_frames <= total_frames:
        windows.append((start, start + window_frames))
        start += stride_frames

    final = (total_frames - window_frames, total_frames)
    if windows[-1] != final:
        windows.append(final)
    return windows


def aggregate_section_evidence(
    sections: list[dict[str, object]],
    *,
    min_score: float = 0.30,
    min_voiced_ratio: float = 0.15,
    min_adjacent_sections: int = 2,
) -> list[dict[str, object]]:
    evidence: dict[str, dict[str, object]] = {}
    single_section_recording = len(sections) <= 1
    required_run = 1 if single_section_recording else min_adjacent_sections

    for index, section in enumerate(sections):
        technique = section.get("primary_technique")
        if not isinstance(technique, str) or not technique:
            continue
        score = float(section.get("primary_technique_score", 0.0))
        voiced_ratio = float(section.get("voiced_ratio", 0.0))
        if score < min_score or voiced_ratio < min_voiced_ratio:
            continue

        row = evidence.setdefault(
            technique,
            {
                "technique": technique,
                "section_count": 0,
                "max_score": 0.0,
                "mean_score": 0.0,
                "first_start_s": section.get("start_s"),
                "last_end_s": section.get("end_s"),
                "_score_sum": 0.0,
                "_indices": [],
            },
        )
        row["section_count"] = int(row["section_count"]) + 1
        row["max_score"] = max(float(row["max_score"]), score)
        row["_score_sum"] = float(row["_score_sum"]) + score
        row["mean_score"] = float(row["_score_sum"]) / int(row["section_count"])
        row["last_end_s"] = section.get("end_s")
        indices = row["_indices"]
        if isinstance(indices, list):
            indices.append(index)

    aggregated: list[dict[str, object]] = []
    for row in evidence.values():
        indices = row.pop("_indices", [])
        row.pop("_score_sum", None)
        longest_run = 0
        current_run = 0
        previous_index: int | None = None
        for index in indices if isinstance(indices, list) else []:
            if previous_index is None or int(index) == previous_index + 1:
                current_run += 1
            else:
                current_run = 1
            previous_index = int(index)
            longest_run = max(longest_run, current_run)
        row["adjacent_section_run"] = longest_run
        row["detected"] = longest_run >= required_run
        aggregated.append(row)

    return sorted(
        aggregated,
        key=lambda item: (bool(item["detected"]), float(item["max_score"]), int(item["section_count"])),
        reverse=True,
    )
