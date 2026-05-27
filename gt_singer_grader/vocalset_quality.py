"""Train a VocalSet execution-quality calibrator.

This module keeps GT Singer responsible for technique recognition, then uses
VocalSet audio to learn a first-pass quality score for a requested technique.
VocalSet does not provide explicit good/bad execution ratings, so this trainer
uses weak supervision:

- matching VocalSet technique labels are high-quality targets;
- normal/straight/control-like takes are average-quality targets;
- mismatched target technique pairs are low-quality contrast examples;
- NanoPitch VAD and pitch consistency provide audio-derived quality features.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .constants import FAMILY_NAMES, PRIMARY_FAMILY_TO_TECHNIQUES, TECHNIQUE_FOLDER_TO_FAMILY
from .feedback import summarize_prediction
from .infer import LoadedPredictor, load_predictor, predict_outputs, resolve_default_nanopitch_vad_checkpoint


VOCALSET_TECHNIQUE_TO_FAMILY = {
    "belt": "mixed_voice",
    "belted": "mixed_voice",
    "breathy": "breathy",
    "glissando": "glissando",
    "messa": "control",
    "messa_di_voce": "control",
    "normal": "control",
    "spoken": "control",
    "straight": "control",
    "trill": "vibrato",
    "trillo": "vibrato",
    "vibrato": "vibrato",
    "vocal_fry": "pharyngeal",
}

AVERAGE_TECHNIQUE_TOKENS = {"control", "messa", "messa_di_voce", "normal", "spoken", "straight"}
GOOD_QUALITY_TARGET = 1.0
AVERAGE_QUALITY_TARGET = 0.5
MISMATCH_QUALITY_TARGET = 0.15


@dataclass(frozen=True)
class VocalSetRecord:
    wav_path: str
    family: str
    source_technique: str
    singer: str
    vowel: str
    context: str


def _normalize_token(text: str) -> str:
    return text.strip().lower().replace("-", "_").replace(" ", "_")


def _family_from_parts(parts: list[str]) -> tuple[str | None, str | None]:
    normalized = [_normalize_token(part) for part in parts]
    joined_candidates = []
    for index, token in enumerate(normalized):
        joined_candidates.append(token)
        if index + 1 < len(normalized):
            joined_candidates.append(f"{token}_{normalized[index + 1]}")
        if index + 2 < len(normalized):
            joined_candidates.append(f"{token}_{normalized[index + 1]}_{normalized[index + 2]}")

    for candidate in joined_candidates:
        if candidate in VOCALSET_TECHNIQUE_TO_FAMILY:
            return VOCALSET_TECHNIQUE_TO_FAMILY[candidate], candidate

    for part in parts:
        if part in TECHNIQUE_FOLDER_TO_FAMILY:
            return TECHNIQUE_FOLDER_TO_FAMILY[part], part

    return None, None


def scan_vocalset(root: str) -> list[VocalSetRecord]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"VocalSet root not found: {root_path}")

    records: list[VocalSetRecord] = []
    for wav_path in sorted(root_path.rglob("*.wav")):
        relative = wav_path.relative_to(root_path)
        parts = list(relative.parts)
        stem_tokens = wav_path.stem.split("_")
        family, technique = _family_from_parts(parts + stem_tokens)
        if family is None or technique is None:
            continue

        singer = next((part for part in parts if part[:1].lower() in {"f", "m"} and any(ch.isdigit() for ch in part)), "")
        vowel = next((token for token in stem_tokens if _normalize_token(token) in {"a", "e", "i", "o", "u"}), "")
        context = next(
            (
                token
                for token in stem_tokens
                if _normalize_token(token) in {"arpeggios", "scales", "long", "long_tones", "excerpts"}
            ),
            "",
        )
        records.append(
            VocalSetRecord(
                wav_path=str(wav_path),
                family=family,
                source_technique=technique,
                singer=singer,
                vowel=vowel,
                context=context,
            )
        )

    if not records:
        raise RuntimeError(f"No usable VocalSet WAV files found under: {root_path}")
    return records


def split_records(records: list[VocalSetRecord], val_ratio: float, seed: int) -> tuple[list[VocalSetRecord], list[VocalSetRecord]]:
    rng = random.Random(seed)
    groups = sorted(set(record.singer or Path(record.wav_path).parts[-2] for record in records))
    rng.shuffle(groups)
    n_val = max(1, int(round(len(groups) * val_ratio))) if len(groups) > 1 else 0
    val_groups = set(groups[:n_val])
    train = [record for record in records if (record.singer or Path(record.wav_path).parts[-2]) not in val_groups]
    val = [record for record in records if (record.singer or Path(record.wav_path).parts[-2]) in val_groups]
    return train, val


def _feature_vector(summary: dict[str, object], target_family: str) -> list[float]:
    family_probs = dict(summary["family_probabilities"])  # type: ignore[arg-type]
    technique_scores = dict(summary["technique_scores"])  # type: ignore[arg-type]
    target_keys = PRIMARY_FAMILY_TO_TECHNIQUES[target_family]

    target_prob = float(family_probs.get(target_family, 0.0))
    detected_conf = float(summary.get("detected_confidence", 0.0))
    margin = float(summary.get("family_margin", 0.0))
    voiced_ratio = float(summary.get("voiced_ratio", 0.0))

    if target_family == "control":
        target_strength = 1.0 - max(float(value) for value in technique_scores.values())
        off_target = max(float(value) for value in technique_scores.values())
    else:
        target_strength = sum(float(technique_scores.get(key, 0.0)) for key in target_keys) / max(1, len(target_keys))
        off_target_values = [float(value) for key, value in technique_scores.items() if key not in set(target_keys)]
        off_target = sum(off_target_values) / max(1, len(off_target_values))

    family_features = [float(family_probs.get(family, 0.0)) for family in FAMILY_NAMES]
    technique_features = [float(technique_scores.get(key, 0.0)) for key in ("mix", "falsetto", "breathy", "pharyngeal", "glissando", "vibrato")]
    return [
        target_prob,
        detected_conf,
        margin,
        voiced_ratio,
        target_strength,
        off_target,
        target_strength - off_target,
        *family_features,
        *technique_features,
    ]


def build_feature_cache(
    predictor: LoadedPredictor,
    records: list[VocalSetRecord],
    output_csv: str,
    *,
    max_records: int | None = None,
) -> None:
    selected = records[:max_records] if max_records else records
    fieldnames = ["wav_path", "family", "source_technique", "singer", "vowel", "context", "features_json"]
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in tqdm(selected, desc="Caching VocalSet features", unit="clip"):
            outputs = predict_outputs(predictor, record.wav_path)
            summary = summarize_prediction(outputs)
            writer.writerow(
                {
                    **asdict(record),
                    "features_json": json.dumps(_feature_vector(summary, record.family)),
                }
            )


class QualityFeatureDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], *, seed: int, negatives_per_positive: int = 2) -> None:
        self.examples: list[tuple[list[float], float]] = []
        rng = random.Random(seed)
        families = list(FAMILY_NAMES)
        for row in rows:
            features = json.loads(row["features_json"])
            family = row["family"]
            source_technique = _normalize_token(row.get("source_technique", ""))
            quality_target = AVERAGE_QUALITY_TARGET if source_technique in AVERAGE_TECHNIQUE_TOKENS else GOOD_QUALITY_TARGET
            self.examples.append((features, quality_target))
            negative_families = [item for item in families if item != family]
            rng.shuffle(negative_families)
            for negative_family in negative_families[:negatives_per_positive]:
                mutated = list(features)
                # Recompute target-relative slots from family-prob slots.
                family_offset = 7
                target_index = FAMILY_NAMES.index(negative_family)
                true_index = FAMILY_NAMES.index(family)
                mutated[0] = mutated[family_offset + target_index]
                mutated[2] = max(0.0, mutated[family_offset + target_index] - mutated[family_offset + true_index])
                self.examples.append((mutated, MISMATCH_QUALITY_TARGET))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features, target = self.examples[index]
        return torch.tensor(features, dtype=torch.float32), torch.tensor([target], dtype=torch.float32)


class QualityCalibrator(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def _read_feature_rows(path: str) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def train_calibrator(
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    output_dir: str,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    train_dataset = QualityFeatureDataset(train_rows, seed=seed)
    val_dataset = QualityFeatureDataset(val_rows, seed=seed + 1)
    input_size = len(train_dataset[0][0])
    model = QualityCalibrator(input_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    def run_epoch(loader: DataLoader, train: bool) -> dict[str, float]:
        model.train(train)
        total_loss = 0.0
        total = 0
        total_abs_err = 0.0
        within_band = 0
        for features, target in loader:
            features = features.to(device)
            target = target.to(device)
            with torch.set_grad_enabled(train):
                score = torch.sigmoid(model(features))
                loss = loss_fn(score, target)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            abs_err = torch.abs(score - target)
            total_abs_err += float(abs_err.sum().item())
            within_band += int((abs_err <= 0.2).sum().item())
            total += int(target.numel())
            total_loss += float(loss.item()) * int(target.numel())
        return {
            "loss": total_loss / max(1, total),
            "mae": total_abs_err / max(1, total),
            "within_0p20": within_band / max(1, total),
        }

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    best_score = -math.inf
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(train_loader, train=True)
        val_metrics = run_epoch(val_loader, train=False)
        print(f"epoch={epoch} train={train_metrics} val={val_metrics}")
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "input_size": input_size,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "quality_targets": {
                "professional_matching_technique": GOOD_QUALITY_TARGET,
                "normal_or_control_like_clip": AVERAGE_QUALITY_TARGET,
                "mismatched_target_technique": MISMATCH_QUALITY_TARGET,
            },
            "note": "Weakly supervised VocalSet quality regressor; professional matching technique takes are good, normal/control-like clips are average, and mismatched target techniques are low-quality contrast.",
        }
        torch.save(checkpoint, os.path.join(output_dir, f"epoch_{epoch:03d}.pth"))
        score = -val_metrics["mae"]
        if score >= best_score:
            best_score = score
            torch.save(checkpoint, os.path.join(output_dir, "best.pth"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a VocalSet technique-execution quality calibrator")
    parser.add_argument("--vocalset-root", required=True)
    parser.add_argument("--technique-checkpoint", default="./gt_singer_grader/models/technique_demo_best.pth")
    parser.add_argument("--nanopitch-vad-checkpoint", default=None)
    parser.add_argument("--output-dir", default="./gt_singer_grader/runs/vocalset_quality")
    parser.add_argument("--feature-cache", default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    device = choose_device(args.device)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    records = scan_vocalset(args.vocalset_root)
    train_records, val_records = split_records(records, val_ratio=args.val_ratio, seed=args.seed)
    with open(os.path.join(output_dir, "vocalset_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"train": [asdict(item) for item in train_records], "val": [asdict(item) for item in val_records]}, handle, indent=2)
    print(f"VocalSet records: train={len(train_records)} val={len(val_records)}")

    predictor = load_predictor(
        args.technique_checkpoint,
        device_name=str(device),
        nanopitch_vad_checkpoint=args.nanopitch_vad_checkpoint or resolve_default_nanopitch_vad_checkpoint(),
        use_nanopitch_vad=True,
    )

    feature_cache = args.feature_cache or os.path.join(output_dir, "features.csv")
    if not os.path.exists(feature_cache):
        build_feature_cache(predictor, train_records + val_records, feature_cache, max_records=args.max_records)

    rows = _read_feature_rows(feature_cache)
    train_paths = {record.wav_path for record in train_records}
    train_rows = [row for row in rows if row["wav_path"] in train_paths]
    val_rows = [row for row in rows if row["wav_path"] not in train_paths]
    train_calibrator(
        train_rows,
        val_rows,
        output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=device,
    )


if __name__ == "__main__":
    main()
