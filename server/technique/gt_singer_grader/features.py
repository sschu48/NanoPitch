"""Audio loading and log-mel feature extraction."""

from __future__ import annotations

import math
import random
import wave
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from .constants import DEFAULT_N_MELS, FRAME_HOP_SECONDS, FRAME_WINDOW_SECONDS, SAMPLE_RATE


def load_wav_mono(path: str, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Load a PCM WAV file as mono float32 in [-1, 1]."""
    with wave.open(path, "rb") as handle:
        channels = handle.getnchannels()
        native_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())

    if sample_width == 1:
        audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 3:
        # 24-bit PCM is common in studio datasets; expand to signed int32.
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        audio = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        sign_bit = 1 << 23
        audio = ((audio ^ sign_bit) - sign_bit).astype(np.float32) / float(1 << 23)
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width: {sample_width}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    waveform = torch.from_numpy(audio.copy())
    if native_rate != sample_rate:
        new_length = max(1, int(round(waveform.numel() * sample_rate / native_rate)))
        waveform = F.interpolate(
            waveform.view(1, 1, -1),
            size=new_length,
            mode="linear",
            align_corners=False,
        ).view(-1)
    return waveform


def augment_user_recording_audio(audio: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Lightweight user-recording augmentation for technique training.

    This intentionally uses only torch/stdlib operations so the training
    pipeline stays dependency-light. It simulates common app-recording
    mismatch: gain changes, room noise, mild clipping, and simple mic EQ.
    """
    if audio.numel() == 0:
        return audio

    x = audio.float().clone()

    # User recordings arrive at wildly different levels.
    x = x * random.uniform(0.55, 1.65)

    # Add room/device noise at moderate SNRs.
    if random.random() < 0.75:
        snr_db = random.uniform(8.0, 32.0)
        signal_rms = x.pow(2).mean().sqrt().clamp_min(1e-5)
        noise = torch.randn_like(x)
        noise_rms = noise.pow(2).mean().sqrt().clamp_min(1e-5)
        noise_gain = signal_rms / (noise_rms * (10.0 ** (snr_db / 20.0)))
        x = x + noise * noise_gain

    # Very small echo/reverb tail to mimic untreated rooms.
    if random.random() < 0.35 and x.numel() > sample_rate // 10:
        delay = random.randint(int(0.012 * sample_rate), int(0.08 * sample_rate))
        decay = random.uniform(0.08, 0.28)
        echo = torch.zeros_like(x)
        echo[delay:] = x[:-delay] * decay
        x = x + echo

    # Crude microphone EQ: random low-pass smoothing.
    if random.random() < 0.35:
        kernel_size = random.choice((5, 9, 13, 17))
        y = F.avg_pool1d(
            x.view(1, 1, -1),
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        ).view(-1)
        x = 0.65 * x + 0.35 * y[: x.numel()]

    # Cheap-device clipping or over-hot input.
    if random.random() < 0.30:
        threshold = random.uniform(0.55, 0.95)
        x = x.clamp(-threshold, threshold) / threshold

    peak = x.abs().max().clamp_min(1.0)
    return (x / peak).clamp(-1.0, 1.0)


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


@lru_cache(maxsize=32)
def build_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 30.0,
    fmax: float | None = None,
) -> torch.Tensor:
    """Create a triangular mel filterbank."""
    if fmax is None:
        fmax = sample_rate / 2.0

    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for index in range(n_mels):
        left, center, right = bins[index : index + 3]
        center = max(center, left + 1)
        right = max(right, center + 1)

        for freq in range(left, min(center, filterbank.shape[1])):
            filterbank[index, freq] = (freq - left) / max(center - left, 1)
        for freq in range(center, min(right, filterbank.shape[1])):
            filterbank[index, freq] = (right - freq) / max(right - center, 1)

    return torch.from_numpy(filterbank)


def log_mel_spectrogram(
    audio: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = DEFAULT_N_MELS,
    hop_seconds: float = FRAME_HOP_SECONDS,
    window_seconds: float = FRAME_WINDOW_SECONDS,
) -> torch.Tensor:
    """Return a (frames, n_mels) log-mel tensor."""
    win_length = int(round(window_seconds * sample_rate))
    hop_length = int(round(hop_seconds * sample_rate))
    n_fft = max(256, 2 ** math.ceil(math.log2(win_length)))

    window = torch.hann_window(win_length, dtype=audio.dtype)
    stft = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    power = stft.abs().pow(2.0)
    mel_bank = build_mel_filterbank(sample_rate, n_fft, n_mels).to(power.device, power.dtype)
    mel = torch.matmul(mel_bank, power)
    log_mel = torch.log(mel.clamp_min(1e-8))
    return log_mel.transpose(0, 1).contiguous()
