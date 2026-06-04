"""Standalone singing-technique grading model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .constants import DEFAULT_N_MELS, FAMILY_NAMES, TECHNIQUE_KEYS

ARCHITECTURE_CONV_GRU = "conv_gru"
ARCHITECTURE_BAND_ROFORMER = "band_roformer"
MODEL_ARCHITECTURES = (ARCHITECTURE_CONV_GRU, ARCHITECTURE_BAND_ROFORMER)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class TechniqueGraderModel(nn.Module):
    def __init__(
        self,
        *,
        n_mels: int = DEFAULT_N_MELS,
        conv_size: int = 96,
        hidden_size: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.2,
        num_techniques: int = len(TECHNIQUE_KEYS),
        num_families: int = len(FAMILY_NAMES),
    ) -> None:
        super().__init__()
        self.architecture = ARCHITECTURE_CONV_GRU
        self.n_mels = n_mels
        self.num_techniques = num_techniques
        self.num_families = num_families

        self.conv1 = CausalConv1d(n_mels, conv_size, kernel_size=5)
        self.conv2 = CausalConv1d(conv_size, hidden_size, kernel_size=3)
        self.norm1 = nn.GroupNorm(1, conv_size)
        self.norm2 = nn.GroupNorm(1, hidden_size)

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        self.frame_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.vad_head = nn.Linear(hidden_size, 1)
        self.technique_head = nn.Linear(hidden_size, num_techniques)
        self.attention = nn.Linear(hidden_size, 1)
        self.clip_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_families),
        )

    @classmethod
    def from_config(cls, config: dict[str, int | float]) -> "TechniqueGraderModel":
        model_config = dict(config)
        model_config.pop("architecture", None)
        return cls(**model_config)

    def _attention_pool(
        self,
        frame_features: torch.Tensor,
        frame_mask: torch.Tensor,
        voiced_weights: torch.Tensor,
    ) -> torch.Tensor:
        valid = frame_mask > 0.0
        attn_logits = self.attention(frame_features).squeeze(-1)
        attn_logits = attn_logits + torch.log(voiced_weights.clamp_min(1e-4))
        attn_logits = attn_logits.masked_fill(~valid, -1e4)
        attn = torch.softmax(attn_logits, dim=1)
        weighted_mean = torch.sum(attn.unsqueeze(-1) * frame_features, dim=1)

        masked_features = frame_features.masked_fill(~valid.unsqueeze(-1), -1e4)
        max_pool = masked_features.amax(dim=1)
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        return torch.cat([weighted_mean, max_pool], dim=-1)

    def forward(
        self,
        mel: torch.Tensor,
        *,
        frame_mask: torch.Tensor | None = None,
        voice_activity_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if frame_mask is None:
            frame_mask = torch.ones(mel.shape[:2], device=mel.device, dtype=mel.dtype)

        x = mel.transpose(1, 2)
        x = F.gelu(self.norm1(self.conv1(x)))
        x = F.gelu(self.norm2(self.conv2(x)))
        conv_features = x.transpose(1, 2)

        gru_features, _ = self.gru(conv_features)
        frame_features = self.frame_proj(torch.cat([conv_features, gru_features], dim=-1))

        vad_logits = self.vad_head(frame_features).squeeze(-1)
        technique_logits = self.technique_head(frame_features)

        if voice_activity_mask is None:
            voiced_weights = torch.sigmoid(vad_logits)
        else:
            voiced_weights = voice_activity_mask.float()
        voiced_weights = voiced_weights * frame_mask.float()

        clip_embedding = self._attention_pool(frame_features, frame_mask.float(), voiced_weights)
        clip_logits = self.clip_head(clip_embedding)

        return {
            "vad_logits": vad_logits,
            "technique_logits": technique_logits,
            "clip_logits": clip_logits,
        }


class RotarySelfAttention(nn.Module):
    """Multi-head self-attention with rotary position applied to q/k pairs."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        head_dim = hidden_size // num_heads
        if head_dim % 2 != 0:
            raise ValueError("hidden_size / num_heads must be even for rotary attention")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def _rotary_cache(self, frames: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(frames, device=device, dtype=dtype)
        dims = torch.arange(0, self.head_dim, 2, device=device, dtype=dtype)
        inv_freq = 1.0 / (10000 ** (dims / self.head_dim))
        angles = positions[:, None] * inv_freq[None, :]
        return torch.cos(angles), torch.sin(angles)

    @staticmethod
    def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        rotated = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
        return rotated.flatten(-2)

    def forward(self, x: torch.Tensor, *, frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, frames, _features = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, frames, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, frames, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, frames, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self._rotary_cache(frames, x.device, q.dtype)
        q = self._apply_rotary(q, cos, sin)
        k = self._apply_rotary(k, cos, sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        if frame_mask is not None:
            valid = frame_mask > 0.0
            scores = scores.masked_fill(~valid[:, None, None, :], -1e4)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        y = torch.matmul(attn, v)
        y = y.transpose(1, 2).contiguous().view(batch, frames, self.hidden_size)
        return self.out(y)


class RotaryEncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ff_size: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = RotarySelfAttention(hidden_size, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, hidden_size),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, *, frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.dropout1(self.attn(self.norm1(x), frame_mask=frame_mask))
        x = x + self.dropout2(self.ff(self.norm2(x)))
        if frame_mask is not None:
            x = x * frame_mask.unsqueeze(-1).float()
        return x


class BandRoFormerTechniqueModel(nn.Module):
    """Small non-causal RoFormer-style encoder for log-mel technique grading."""

    def __init__(
        self,
        *,
        n_mels: int = DEFAULT_N_MELS,
        hidden_size: int = 128,
        roformer_layers: int = 4,
        roformer_heads: int = 4,
        roformer_ff_size: int = 256,
        dropout: float = 0.2,
        num_techniques: int = len(TECHNIQUE_KEYS),
        num_families: int = len(FAMILY_NAMES),
    ) -> None:
        super().__init__()
        self.architecture = ARCHITECTURE_BAND_ROFORMER
        self.n_mels = n_mels
        self.num_techniques = num_techniques
        self.num_families = num_families

        self.input_proj = nn.Sequential(
            nn.LayerNorm(n_mels),
            nn.Linear(n_mels, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            [
                RotaryEncoderLayer(
                    hidden_size=hidden_size,
                    num_heads=roformer_heads,
                    ff_size=roformer_ff_size,
                    dropout=dropout,
                )
                for _ in range(roformer_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.vad_head = nn.Linear(hidden_size, 1)
        self.technique_head = nn.Linear(hidden_size, num_techniques)
        self.attention = nn.Linear(hidden_size, 1)
        self.clip_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_families),
        )

    @classmethod
    def from_config(cls, config: dict[str, int | float | str]) -> "BandRoFormerTechniqueModel":
        model_config = dict(config)
        model_config.pop("architecture", None)
        return cls(**model_config)

    def _attention_pool(
        self,
        frame_features: torch.Tensor,
        frame_mask: torch.Tensor,
        voiced_weights: torch.Tensor,
    ) -> torch.Tensor:
        valid = frame_mask > 0.0
        attn_logits = self.attention(frame_features).squeeze(-1)
        attn_logits = attn_logits + torch.log(voiced_weights.clamp_min(1e-4))
        attn_logits = attn_logits.masked_fill(~valid, -1e4)
        attn = torch.softmax(attn_logits, dim=1)
        weighted_mean = torch.sum(attn.unsqueeze(-1) * frame_features, dim=1)

        masked_features = frame_features.masked_fill(~valid.unsqueeze(-1), -1e4)
        max_pool = masked_features.amax(dim=1)
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        return torch.cat([weighted_mean, max_pool], dim=-1)

    def forward(
        self,
        mel: torch.Tensor,
        *,
        frame_mask: torch.Tensor | None = None,
        voice_activity_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if frame_mask is None:
            frame_mask = torch.ones(mel.shape[:2], device=mel.device, dtype=mel.dtype)

        frame_mask = frame_mask.float()
        x = self.input_proj(mel) * frame_mask.unsqueeze(-1)
        for layer in self.layers:
            x = layer(x, frame_mask=frame_mask)
        frame_features = self.final_norm(x)
        frame_features = frame_features * frame_mask.unsqueeze(-1)

        vad_logits = self.vad_head(frame_features).squeeze(-1)
        technique_logits = self.technique_head(frame_features)

        if voice_activity_mask is None:
            voiced_weights = torch.sigmoid(vad_logits)
        else:
            voiced_weights = voice_activity_mask.float()
        voiced_weights = voiced_weights * frame_mask

        clip_embedding = self._attention_pool(frame_features, frame_mask, voiced_weights)
        clip_logits = self.clip_head(clip_embedding)

        return {
            "vad_logits": vad_logits,
            "technique_logits": technique_logits,
            "clip_logits": clip_logits,
        }


def architecture_from_config(config: dict[str, object] | None) -> str:
    if not config:
        return ARCHITECTURE_CONV_GRU
    architecture = config.get("architecture", ARCHITECTURE_CONV_GRU)
    if architecture not in MODEL_ARCHITECTURES:
        raise ValueError(f"unsupported architecture: {architecture!r}")
    return str(architecture)


def build_model_from_config(config: dict[str, int | float | str]) -> nn.Module:
    architecture = architecture_from_config(config)
    if architecture == ARCHITECTURE_CONV_GRU:
        return TechniqueGraderModel.from_config(config)
    if architecture == ARCHITECTURE_BAND_ROFORMER:
        return BandRoFormerTechniqueModel.from_config(config)
    raise ValueError(f"unsupported architecture: {architecture!r}")
