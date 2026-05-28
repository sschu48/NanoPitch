"""Inference helpers for the VocalSet execution-quality calibrator."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .constants import FAMILY_NAMES, PRIMARY_FAMILY_TO_TECHNIQUES


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


def feature_vector(summary: dict[str, object], target_family: str) -> list[float]:
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
    technique_features = [
        float(technique_scores.get(key, 0.0))
        for key in ("mix", "falsetto", "breathy", "pharyngeal", "glissando", "vibrato")
    ]
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


def resolve_default_quality_checkpoint() -> str | None:
    root = Path(__file__).resolve().parent
    candidates = [
        root / "models" / "gt_singer_vocalset" / "vocalset_quality_best.pth",
        root / "runs" / "vocalset_quality" / "best.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def load_quality_calibrator(checkpoint_path: str, device: torch.device) -> tuple[QualityCalibrator, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    input_size = int(checkpoint.get("input_size", 20))
    model = QualityCalibrator(input_size)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def predict_quality_score(model: QualityCalibrator, device: torch.device, summary: dict[str, object], target_family: str) -> float:
    features = torch.tensor([feature_vector(summary, target_family)], dtype=torch.float32, device=device)
    with torch.no_grad():
        return float(torch.sigmoid(model(features)).detach().cpu().item())
