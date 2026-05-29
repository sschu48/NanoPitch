"""Shared evaluation artifact requirements."""

from __future__ import annotations

import hashlib
from pathlib import Path


REQUIRED_EVALUATION_ARTIFACTS = (
    "evaluation_config.json",
    "metrics.json",
    "predictions.csv",
    "confusion_matrix.csv",
    "threshold_sweep.json",
    "operating_point.json",
    "calibration.json",
    "calibration.csv",
)


def validate_eval_artifacts(candidate_eval_dir: str | Path) -> list[str]:
    root = Path(candidate_eval_dir)
    return [name for name in REQUIRED_EVALUATION_ARTIFACTS if not (root / name).is_file()]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eval_artifact_hashes(candidate_eval_dir: str | Path) -> dict[str, str]:
    root = Path(candidate_eval_dir)
    return {
        name: sha256_file(root / name)
        for name in REQUIRED_EVALUATION_ARTIFACTS
        if (root / name).is_file()
    }
