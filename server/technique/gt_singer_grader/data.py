"""Dataset indexing and PyTorch dataset wrappers."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

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
    TECHNIQUE_INDEX,
    TECHNIQUE_FOLDER_TO_FAMILY,
)
from .features import augment_user_recording_audio, load_wav_mono, log_mel_spectrogram
from .labels import build_frame_labels, load_alignment
from .manifest import normalize_label_list, read_jsonl, trainability_reason
from .split_health import split_coverage_errors, trainable_technique_families, validate_val_ratio


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
    validate_val_ratio(val_ratio)

    def split_key(record: SampleRecord) -> str:
        if group_by == "speaker":
            return record.speaker
        return record.split_group

    record_list = list(records)
    unique_groups = sorted({split_key(record) for record in record_list})

    rng = random.Random(seed)
    rng.shuffle(unique_groups)
    if len(unique_groups) <= 1 or val_ratio <= 0.0:
        return record_list, []

    n_val = max(1, int(round(len(unique_groups) * val_ratio)))
    n_val = min(n_val, len(unique_groups) - 1)
    val_groups: set[str] = set()
    target_families = trainable_technique_families(record_list)

    def can_add(candidate: str) -> bool:
        next_val_groups = val_groups | {candidate}
        next_train = [record for record in record_list if split_key(record) not in next_val_groups]
        next_val = [record for record in record_list if split_key(record) in next_val_groups]
        if split_coverage_errors(next_train, source="train split", purpose="training"):
            return False
        if split_coverage_errors(next_val, source="validation split", purpose="validation"):
            return False
        return target_families <= trainable_technique_families(next_train)

    for family in sorted(target_families):
        current_val = [record for record in record_list if split_key(record) in val_groups]
        if family in trainable_technique_families(current_val):
            continue
        for candidate in unique_groups:
            if candidate in val_groups:
                continue
            candidate_records = [record for record in record_list if split_key(record) == candidate]
            if family not in trainable_technique_families(candidate_records):
                continue
            if can_add(candidate):
                val_groups.add(candidate)
                break

    target_val_groups = max(n_val, len(val_groups))
    for candidate in unique_groups:
        if len(val_groups) >= target_val_groups:
            continue
        if candidate in val_groups:
            continue
        if can_add(candidate):
            val_groups.add(candidate)

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


def summarize_manifest_records(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    counter = Counter()
    for record in records:
        family = manifest_family(record)
        dataset = str(record.get("dataset") or "unknown")
        counter[f"{dataset}:{family}"] += 1
    return dict(sorted(counter.items()))


def read_training_manifest(path: str) -> list[dict[str, Any]]:
    return read_jsonl(path)


def manifest_audio_path(record: dict[str, Any]) -> str:
    audio_path = record.get("audio_path") or record.get("wav_path")
    if not isinstance(audio_path, str) or not audio_path:
        raise ValueError(f"manifest record has no audio path: {record.get('recording_id') or record.get('stem')}")
    return audio_path


def manifest_json_path(record: dict[str, Any]) -> str:
    json_path = record.get("json_path")
    return json_path if isinstance(json_path, str) else ""


def manifest_family(record: dict[str, Any]) -> str:
    if record.get("role") in {"control", "speech"}:
        return "control"
    family = record.get("family")
    if isinstance(family, str) and family:
        return family
    labels = record.get("labels")
    if isinstance(labels, dict):
        families = normalize_label_list(labels.get("families"))
        if families:
            return families[0]
    raise ValueError(f"manifest record has no family label: {record.get('recording_id') or record.get('stem')}")


def manifest_techniques(record: dict[str, Any]) -> list[str]:
    labels = record.get("labels")
    if isinstance(labels, dict):
        return normalize_label_list(labels.get("techniques"))
    return []


def _choose_window(total_frames: int, max_frames: int, training: bool) -> tuple[int, int]:
    if total_frames <= max_frames:
        return 0, total_frames
    if training:
        start = random.randint(0, total_frames - max_frames)
    else:
        start = (total_frames - max_frames) // 2
    return start, start + max_frames


def _pad_example(
    mel: torch.Tensor,
    vad_target: torch.Tensor,
    technique_target: torch.Tensor,
    max_frames: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    frame_mask = torch.ones(mel.size(0), dtype=torch.float32)
    if mel.size(0) < max_frames:
        pad_frames = max_frames - mel.size(0)
        mel = F.pad(mel, (0, 0, 0, pad_frames))
        vad_target = F.pad(vad_target, (0, pad_frames))
        technique_target = F.pad(technique_target, (0, 0, 0, pad_frames))
        frame_mask = F.pad(frame_mask, (0, pad_frames))
    return mel, vad_target, technique_target, frame_mask


def _weak_clip_targets(n_frames: int, techniques: Iterable[str]) -> tuple[torch.Tensor, torch.Tensor]:
    vad_target = torch.ones(n_frames, dtype=torch.float32)
    technique_target = torch.zeros((n_frames, len(TECHNIQUE_INDEX)), dtype=torch.float32)
    for technique in techniques:
        index = TECHNIQUE_INDEX.get(technique)
        if index is not None:
            technique_target[:, index] = 1.0
    return vad_target, technique_target


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
        return _choose_window(total_frames, self.max_frames, self.training)

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

        mel, vad_target, technique_target, frame_mask = _pad_example(
            mel,
            vad_target,
            technique_target,
            self.max_frames,
        )

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


class ManifestTechniqueDataset(Dataset):
    """Clip sampler for normalized manifests and legacy GT Singer manifests."""

    def __init__(
        self,
        records: Iterable[dict[str, Any]],
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

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        record = self.records[index]
        family = manifest_family(record)
        reason = trainability_reason(record)
        if reason != "trainable" or family not in FAMILY_TO_INDEX:
            raise ValueError(f"record cannot be used as a training example yet: {reason}")

        audio_path = manifest_audio_path(record)
        audio = load_wav_mono(audio_path, self.sample_rate)
        if self.training and self.audio_augmentation:
            audio = augment_user_recording_audio(audio, self.sample_rate)
        mel = log_mel_spectrogram(audio, sample_rate=self.sample_rate, n_mels=self.n_mels, hop_seconds=self.hop_seconds)
        mel = (mel - mel.mean()) / mel.std().clamp_min(1e-5)

        json_path = manifest_json_path(record)
        if json_path:
            alignment = load_alignment(json_path)
            vad_target_np, technique_target_np, metadata = build_frame_labels(
                alignment,
                n_frames=mel.size(0),
                hop_seconds=self.hop_seconds,
            )
            vad_target = torch.from_numpy(vad_target_np)
            technique_target = torch.from_numpy(technique_target_np)
        else:
            vad_target, technique_target = _weak_clip_targets(mel.size(0), manifest_techniques(record))
            metadata = {}

        start, end = _choose_window(mel.size(0), self.max_frames, self.training)
        mel = mel[start:end]
        vad_target = vad_target[start:end]
        technique_target = technique_target[start:end]

        mel, vad_target, technique_target, frame_mask = _pad_example(
            mel,
            vad_target,
            technique_target,
            self.max_frames,
        )
        technique_presence = technique_target.max(dim=0).values
        labels = record.get("labels")
        clip_role = labels.get("clip_role", "") if isinstance(labels, dict) else ""

        return {
            "mel": mel.float(),
            "vad_target": vad_target.float(),
            "technique_target": technique_target.float(),
            "technique_presence": technique_presence.float(),
            "frame_mask": frame_mask.float(),
            "clip_label": FAMILY_TO_INDEX[family],
            "family": family,
            "role": str(record.get("role") or clip_role),
            "speaker": str(record.get("speaker_id") or record.get("speaker") or ""),
            "song": str(record.get("song_id") or record.get("song") or ""),
            "stem": str(record.get("recording_id") or Path(audio_path).stem),
            "singing_method": str(metadata.get("singing_method", "")),
            "pace": str(metadata.get("pace", "")),
            "range": str(metadata.get("range", "")),
            "emotion": str(metadata.get("emotion", "")),
        }
