"""Run the GT Singer grader on one WAV file."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from .constants import DEFAULT_MAX_SECONDS, DEFAULT_N_MELS, FAMILY_NAMES, FRAME_HOP_SECONDS
from .features import load_wav_mono, log_mel_spectrogram
from .feedback import summarize_prediction, summary_to_json
from .model import TechniqueGraderModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer singing-technique grades from one audio file")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--target-family", choices=FAMILY_NAMES, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


@dataclass
class LoadedPredictor:
    checkpoint_path: str
    device: torch.device
    model: TechniqueGraderModel
    model_kwargs: dict[str, int | float]
    max_seconds: float
    checkpoint_epoch: int | None
    val_metrics: dict[str, float]


def load_predictor(checkpoint_path: str, device_name: str = "auto") -> LoadedPredictor:
    device = choose_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_kwargs = checkpoint.get("model_kwargs", {"n_mels": DEFAULT_N_MELS})
    model = TechniqueGraderModel.from_config(model_kwargs)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    train_args = checkpoint.get("train_args", {})
    max_seconds = float(train_args.get("max_seconds", DEFAULT_MAX_SECONDS))

    return LoadedPredictor(
        checkpoint_path=checkpoint_path,
        device=device,
        model=model,
        model_kwargs=model_kwargs,
        max_seconds=max_seconds,
        checkpoint_epoch=checkpoint.get("epoch"),
        val_metrics=dict(checkpoint.get("val_metrics") or {}),
    )


def _chunk_mel(mel: torch.Tensor, max_frames: int) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")

    chunks: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    start = 0
    total_frames = mel.size(0)
    while start < total_frames:
        end = min(total_frames, start + max_frames)
        chunk = mel[start:end]
        valid_frames = chunk.size(0)
        frame_mask = torch.ones(valid_frames, dtype=mel.dtype)
        if valid_frames < max_frames:
            pad_frames = max_frames - valid_frames
            chunk = torch.nn.functional.pad(chunk, (0, 0, 0, pad_frames))
            frame_mask = torch.nn.functional.pad(frame_mask, (0, pad_frames))
        chunks.append((chunk, frame_mask, valid_frames))
        start = end
    if not chunks:
        frame_mask = torch.ones(1, dtype=mel.dtype)
        chunks.append((mel.new_zeros((1, mel.size(1))), frame_mask, 1))
    return chunks


def predict_outputs(predictor: LoadedPredictor, audio_path: str) -> dict[str, torch.Tensor]:
    audio = load_wav_mono(audio_path)
    mel = log_mel_spectrogram(audio, n_mels=int(predictor.model_kwargs.get("n_mels", DEFAULT_N_MELS)))
    max_frames = max(1, int(round(predictor.max_seconds / FRAME_HOP_SECONDS)))

    chunk_outputs = []
    for chunk, frame_mask, valid_frames in _chunk_mel(mel, max_frames):
        normalized_chunk = (chunk - chunk.mean()) / chunk.std().clamp_min(1e-5)
        with torch.no_grad():
            outputs = predictor.model(
                normalized_chunk.unsqueeze(0).to(predictor.device),
                frame_mask=frame_mask.unsqueeze(0).to(predictor.device),
            )
        chunk_outputs.append(
            {
                "vad_logits": outputs["vad_logits"].detach().cpu()[0, :valid_frames],
                "technique_logits": outputs["technique_logits"].detach().cpu()[0, :valid_frames],
                "clip_logits": outputs["clip_logits"].detach().cpu()[0],
            }
        )

    return {
        "vad_logits": torch.cat([item["vad_logits"] for item in chunk_outputs], dim=0).unsqueeze(0),
        "technique_logits": torch.cat([item["technique_logits"] for item in chunk_outputs], dim=0).unsqueeze(0),
        "clip_logits": torch.stack([item["clip_logits"] for item in chunk_outputs], dim=0).mean(dim=0, keepdim=True),
    }


def predict_summary(
    predictor: LoadedPredictor,
    audio_path: str,
    *,
    target_family: str | None = None,
) -> dict[str, object]:
    outputs = predict_outputs(predictor, audio_path)
    return summarize_prediction(outputs, target_family=target_family)


def main() -> None:
    args = parse_args()
    predictor = load_predictor(args.checkpoint, device_name=args.device)
    summary = predict_summary(predictor, args.audio, target_family=args.target_family)
    text = summary_to_json(summary)
    print(text)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
