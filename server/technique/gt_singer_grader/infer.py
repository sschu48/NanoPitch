"""Run the GT Singer grader on one WAV file."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from .constants import DEFAULT_MAX_SECONDS, DEFAULT_N_MELS, FAMILY_NAMES, FRAME_HOP_SECONDS
from .features import load_wav_mono, log_mel_spectrogram
from .feedback import summarize_prediction, summary_to_json
from .model import TechniqueGraderModel
from .section_detection import aggregate_section_evidence, section_windows

SECTION_WINDOW_SECONDS = 5.0
SECTION_STRIDE_SECONDS = 2.5


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


def _run_mel_window(
    predictor: LoadedPredictor,
    mel: torch.Tensor,
    *,
    start: int,
    end: int,
    model_frames: int,
) -> dict[str, torch.Tensor]:
    window = mel[start:end]
    valid_frames = window.size(0)
    frame_mask = torch.ones(valid_frames, dtype=mel.dtype)
    if valid_frames < model_frames:
        pad_frames = model_frames - valid_frames
        window = torch.nn.functional.pad(window, (0, 0, 0, pad_frames))
        frame_mask = torch.nn.functional.pad(frame_mask, (0, pad_frames))
    normalized_window = (window - window.mean()) / window.std().clamp_min(1e-5)
    with torch.no_grad():
        outputs = predictor.model(
            normalized_window.unsqueeze(0).to(predictor.device),
            frame_mask=frame_mask.unsqueeze(0).to(predictor.device),
        )
    return {
        "vad_logits": outputs["vad_logits"].detach().cpu()[:, :valid_frames],
        "technique_logits": outputs["technique_logits"].detach().cpu()[:, :valid_frames],
        "clip_logits": outputs["clip_logits"].detach().cpu(),
    }


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


def predict_section_summaries(
    predictor: LoadedPredictor,
    audio_path: str,
    *,
    target_family: str | None = None,
    window_seconds: float = SECTION_WINDOW_SECONDS,
    stride_seconds: float = SECTION_STRIDE_SECONDS,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    audio = load_wav_mono(audio_path)
    mel = log_mel_spectrogram(audio, n_mels=int(predictor.model_kwargs.get("n_mels", DEFAULT_N_MELS)))
    model_frames = max(1, int(round(predictor.max_seconds / FRAME_HOP_SECONDS)))
    window_frames = min(model_frames, max(1, int(round(window_seconds / FRAME_HOP_SECONDS))))
    stride_frames = max(1, int(round(stride_seconds / FRAME_HOP_SECONDS)))

    sections: list[dict[str, object]] = []
    for start, end in section_windows(mel.size(0), window_frames, stride_frames):
        outputs = _run_mel_window(predictor, mel, start=start, end=end, model_frames=model_frames)
        summary = summarize_prediction(outputs, target_family=target_family)
        sections.append(
            {
                "start_s": round(start * FRAME_HOP_SECONDS, 3),
                "end_s": round(end * FRAME_HOP_SECONDS, 3),
                "detected_family": summary.get("detected_family"),
                "detected_confidence": summary.get("detected_confidence"),
                "primary_technique": summary.get("primary_technique"),
                "primary_technique_score": summary.get("primary_technique_score"),
                "detection_status": summary.get("detection_status"),
                "voiced_ratio": summary.get("voiced_ratio"),
                "technique_scores": summary.get("technique_scores"),
            }
        )

    return sections, aggregate_section_evidence(sections)


def predict_summary(
    predictor: LoadedPredictor,
    audio_path: str,
    *,
    target_family: str | None = None,
) -> dict[str, object]:
    outputs = predict_outputs(predictor, audio_path)
    summary = summarize_prediction(outputs, target_family=target_family)
    sections, aggregate = predict_section_summaries(predictor, audio_path, target_family=target_family)
    summary["technique_sections"] = sections
    summary["section_technique_evidence"] = aggregate
    summary["section_detection_config"] = {
        "window_seconds": SECTION_WINDOW_SECONDS,
        "stride_seconds": SECTION_STRIDE_SECONDS,
        "model_context_seconds": predictor.max_seconds,
    }
    return summary


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
