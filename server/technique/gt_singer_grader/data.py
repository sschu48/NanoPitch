"""Dataset indexing and PyTorch dataset wrappers."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .constants import (
    DEFAULT_MAX_SECONDS,
    DEFAULT_N_MELS,
    FAMILY_TO_INDEX,
    FRAME_HOP_SECONDS,
    GROUP_NAME_TO_FAMILY,
    SAMPLE_RATE,
    TECHNIQUE_FOLDER_TO_FAMILY,
)
from .features import augment_user_recording_audio, load_wav_mono, log_mel_spectrogram
from .labels import build_frame_labels, load_alignment


@dataclass(frozen=True)
class SampleRecord:
    speaker: str
    technique_folder: str
    family: str
    role: str
    song: str
    stem: str
    wav_path: str
    json_path: str
    split_group: str

    @property
    def clip_label(self) -> int:
        if self.role in {"control", "speech"}:
            return FAMILY_TO_INDEX["control"]
        return FAMILY_TO_INDEX[self.family]


def _resolve_family(technique_dir_name: str, group_name: str) -> str | None:
    """Map a GT Singer folder/group pair to a clip-level class family."""
    if group_name in {"Control_Group", "Paired_Speech_Group"}:
        return "control"

    if group_name in GROUP_NAME_TO_FAMILY:
        return GROUP_NAME_TO_FAMILY[group_name]

    return TECHNIQUE_FOLDER_TO_FAMILY.get(technique_dir_name)


def scan_gt_singer(root: str, language: str = "English", include_speech: bool = False) -> list[SampleRecord]:
    """Scan the GT Singer tree and return one record per WAV/JSON pair."""
    root_path = Path(root)
    language_root = root_path / language if (root_path / language).exists() else root_path
    if not language_root.exists():
        raise FileNotFoundError(f"dataset root not found: {language_root}")

    records: list[SampleRecord] = []
    for speaker_dir in sorted(path for path in language_root.iterdir() if path.is_dir()):
        for technique_dir in sorted(path for path in speaker_dir.iterdir() if path.is_dir()):
            folder_family = TECHNIQUE_FOLDER_TO_FAMILY.get(technique_dir.name)
            if folder_family is None:
                continue

            for song_dir in sorted(path for path in technique_dir.iterdir() if path.is_dir()):
                for group_dir in sorted(path for path in song_dir.iterdir() if path.is_dir()):
                    group_name = group_dir.name
                    if group_name == "Paired_Speech_Group" and not include_speech:
                        continue
                    if group_name == "Control_Group":
                        role = "control"
                    elif group_name == "Paired_Speech_Group":
                        role = "speech"
                    else:
                        role = "emphasis"

                    family = _resolve_family(technique_dir.name, group_name)
                    if family is None:
                        continue

                    for wav_path in sorted(group_dir.glob("*.wav")):
                        json_path = wav_path.with_suffix(".json")
                        if not json_path.exists():
                            continue
                        split_group = f"{speaker_dir.name}|{technique_dir.name}|{song_dir.name}"
                        records.append(
                            SampleRecord(
                                speaker=speaker_dir.name,
                                technique_folder=technique_dir.name,
                                family=family,
                                role=role,
                                song=song_dir.name,
                                stem=wav_path.stem,
                                wav_path=str(wav_path),
                                json_path=str(json_path),
                                split_group=split_group,
                            )
                        )

    if not records:
        raise RuntimeError(f"no GT Singer records found under: {language_root}")
    return records


def split_records(
    records: Iterable[SampleRecord],
    val_ratio: float = 0.2,
    seed: int = 1337,
    group_by: str = "song",
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    """Split records while keeping related clips on one side of the split."""
    if group_by not in {"song", "speaker"}:
        raise ValueError(f"unknown split group: {group_by}")

    def split_key(record: SampleRecord) -> str:
        if group_by == "speaker":
            return record.speaker
        return record.split_group

    family_groups: dict[str, list[str]] = defaultdict(list)
    record_list = list(records)
    for record in record_list:
        family_groups[record.family].append(split_key(record))

    rng = random.Random(seed)
    val_groups: set[str] = set()
    for family, groups in family_groups.items():
        unique_groups = sorted(set(groups))
        rng.shuffle(unique_groups)
        if len(unique_groups) <= 1 or val_ratio <= 0.0:
            continue
        n_val = max(1, int(round(len(unique_groups) * val_ratio)))
        n_val = min(n_val, len(unique_groups) - 1)
        val_groups.update(unique_groups[:n_val])

    train_records = [record for record in record_list if split_key(record) not in val_groups]
    val_records = [record for record in record_list if split_key(record) in val_groups]
    return train_records, val_records


def write_manifest(path: str, records: Iterable[SampleRecord]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record)) + "\n")


def summarize_records(records: Iterable[SampleRecord]) -> dict[str, int]:
    counter = Counter()
    for record in records:
        counter[f"{record.family}:{record.role}"] += 1
    return dict(sorted(counter.items()))


class GTSingerTechniqueDataset(Dataset):
    """Clip sampler that turns GT Singer records into tensors."""

    def __init__(
        self,
        records: Iterable[SampleRecord],
        *,
        sample_rate: int = SAMPLE_RATE,
        n_mels: int = DEFAULT_N_MELS,
        hop_seconds: float = FRAME_HOP_SECONDS,
        max_seconds: float = DEFAULT_MAX_SECONDS,
        training: bool = True,
        audio_augmentation: bool = False,
    ) -> None:
        self.records = list(records)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_seconds = hop_seconds
        self.max_frames = max(1, int(round(max_seconds / hop_seconds)))
        self.training = training
        self.audio_augmentation = audio_augmentation

    def __len__(self) -> int:
        return len(self.records)

    def _choose_window(self, total_frames: int) -> tuple[int, int]:
        if total_frames <= self.max_frames:
            return 0, total_frames
        if self.training:
            start = random.randint(0, total_frames - self.max_frames)
        else:
            start = (total_frames - self.max_frames) // 2
        return start, start + self.max_frames

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        record = self.records[index]
        audio = load_wav_mono(record.wav_path, self.sample_rate)
        if self.training and self.audio_augmentation:
            audio = augment_user_recording_audio(audio, self.sample_rate)
        mel = log_mel_spectrogram(audio, sample_rate=self.sample_rate, n_mels=self.n_mels, hop_seconds=self.hop_seconds)
        mel = (mel - mel.mean()) / mel.std().clamp_min(1e-5)

        alignment = load_alignment(record.json_path)
        vad_target_np, technique_target_np, metadata = build_frame_labels(
            alignment,
            n_frames=mel.size(0),
            hop_seconds=self.hop_seconds,
        )

        vad_target = torch.from_numpy(vad_target_np)
        technique_target = torch.from_numpy(technique_target_np)

        start, end = self._choose_window(mel.size(0))
        mel = mel[start:end]
        vad_target = vad_target[start:end]
        technique_target = technique_target[start:end]

        frame_mask = torch.ones(mel.size(0), dtype=torch.float32)
        if mel.size(0) < self.max_frames:
            pad_frames = self.max_frames - mel.size(0)
            mel = F.pad(mel, (0, 0, 0, pad_frames))
            vad_target = F.pad(vad_target, (0, pad_frames))
            technique_target = F.pad(technique_target, (0, 0, 0, pad_frames))
            frame_mask = F.pad(frame_mask, (0, pad_frames))

        technique_presence = technique_target.max(dim=0).values

        return {
            "mel": mel.float(),
            "vad_target": vad_target.float(),
            "technique_target": technique_target.float(),
            "technique_presence": technique_presence.float(),
            "frame_mask": frame_mask.float(),
            "clip_label": int(record.clip_label),
            "family": record.family,
            "role": record.role,
            "speaker": record.speaker,
            "song": record.song,
            "stem": record.stem,
            "singing_method": metadata.get("singing_method", ""),
            "pace": metadata.get("pace", ""),
            "range": metadata.get("range", ""),
            "emotion": metadata.get("emotion", ""),
        }
