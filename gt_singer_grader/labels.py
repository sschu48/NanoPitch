"""Alignment parsing and frame-label construction."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from .constants import SILENCE_TOKENS, TECHNIQUE_INDEX, TECHNIQUE_KEYS


def load_alignment(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _frame_bounds(start_time: float, end_time: float, n_frames: int, hop_seconds: float) -> tuple[int, int]:
    start = max(0, int(math.floor(start_time / hop_seconds)))
    end = min(n_frames, int(math.ceil(end_time / hop_seconds)))
    return start, max(start, end)


def build_frame_labels(
    entries: list[dict[str, Any]],
    n_frames: int,
    hop_seconds: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Build frame-wise VAD and technique targets from GT Singer JSON."""
    vad = np.zeros(n_frames, dtype=np.float32)
    techniques = np.zeros((n_frames, len(TECHNIQUE_KEYS)), dtype=np.float32)

    metadata: dict[str, str] = {}
    for field in ("singing_method", "pace", "range", "emotion"):
        for entry in entries:
            value = str(entry.get(field, "")).strip()
            if value:
                metadata[field] = value
                break

    for entry in entries:
        phonemes = entry.get("ph") or []
        ph_start = entry.get("ph_start") or []
        ph_end = entry.get("ph_end") or []

        if not phonemes:
            word = str(entry.get("word", "")).strip()
            start_time = float(entry.get("start_time", 0.0))
            end_time = float(entry.get("end_time", start_time))
            left, right = _frame_bounds(start_time, end_time, n_frames, hop_seconds)
            if word and word not in SILENCE_TOKENS:
                vad[left:right] = 1.0
            continue

        for index, phoneme in enumerate(phonemes):
            start_time = float(ph_start[index]) if index < len(ph_start) else float(entry.get("start_time", 0.0))
            end_time = float(ph_end[index]) if index < len(ph_end) else float(entry.get("end_time", start_time))
            left, right = _frame_bounds(start_time, end_time, n_frames, hop_seconds)
            if right <= left:
                continue

            if str(phoneme).strip() not in SILENCE_TOKENS:
                vad[left:right] = 1.0

            for technique_name in TECHNIQUE_KEYS:
                values = entry.get(technique_name) or []
                if index < len(values) and str(values[index]).strip() == "1":
                    techniques[left:right, TECHNIQUE_INDEX[technique_name]] = 1.0

    return vad, techniques, metadata

