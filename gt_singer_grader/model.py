"""Standalone singing-technique grading model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .constants import DEFAULT_N_MELS, FAMILY_NAMES, TECHNIQUE_KEYS


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
        return cls(**config)

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

