from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

from gt_singer_grader.constants import FAMILY_NAMES, TECHNIQUE_KEYS
if torch is not None:
    from gt_singer_grader.infer import load_predictor
    from gt_singer_grader.model import (
        ARCHITECTURE_BAND_ROFORMER,
        ARCHITECTURE_CONV_GRU,
        TechniqueGraderModel,
        architecture_from_config,
        build_model_from_config,
    )


@unittest.skipIf(torch is None, "torch is required for model architecture tests")
class ModelArchitectureTest(unittest.TestCase):
    def _assert_output_contract(self, outputs: dict[str, torch.Tensor], *, batch: int, frames: int) -> None:
        self.assertEqual(set(outputs), {"vad_logits", "technique_logits", "clip_logits"})
        self.assertEqual(tuple(outputs["vad_logits"].shape), (batch, frames))
        self.assertEqual(tuple(outputs["technique_logits"].shape), (batch, frames, len(TECHNIQUE_KEYS)))
        self.assertEqual(tuple(outputs["clip_logits"].shape), (batch, len(FAMILY_NAMES)))
        for value in outputs.values():
            self.assertTrue(torch.isfinite(value).all())

    def test_conv_gru_factory_defaults_old_config(self) -> None:
        model = build_model_from_config({"n_mels": 16, "conv_size": 12, "hidden_size": 16, "gru_layers": 1})
        self.assertIsInstance(model, TechniqueGraderModel)
        self.assertEqual(architecture_from_config({}), ARCHITECTURE_CONV_GRU)

        mel = torch.randn(2, 7, 16)
        frame_mask = torch.ones(2, 7)
        outputs = model(mel, frame_mask=frame_mask)
        self._assert_output_contract(outputs, batch=2, frames=7)

    def test_band_roformer_output_contract_with_padding(self) -> None:
        model = build_model_from_config(
            {
                "architecture": ARCHITECTURE_BAND_ROFORMER,
                "n_mels": 16,
                "hidden_size": 16,
                "roformer_layers": 2,
                "roformer_heads": 4,
                "roformer_ff_size": 32,
                "dropout": 0.0,
            }
        )
        mel = torch.randn(2, 9, 16)
        frame_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 0, 0, 0],
            ],
            dtype=torch.float32,
        )
        outputs = model(mel, frame_mask=frame_mask)
        self._assert_output_contract(outputs, batch=2, frames=9)

    def test_load_predictor_preserves_old_checkpoint_compatibility(self) -> None:
        model_kwargs = {"n_mels": 16, "conv_size": 12, "hidden_size": 16, "gru_layers": 1, "dropout": 0.0}
        model = TechniqueGraderModel.from_config(model_kwargs)
        with tempfile.TemporaryDirectory() as root:
            checkpoint_path = Path(root) / "old_conv_gru.pth"
            torch.save(
                {
                    "epoch": 1,
                    "model_state": model.state_dict(),
                    "model_kwargs": model_kwargs,
                    "train_args": {"max_seconds": 1.0},
                    "val_metrics": {"clip_acc": 0.5},
                },
                checkpoint_path,
            )

            predictor = load_predictor(str(checkpoint_path), device_name="cpu")
            self.assertEqual(predictor.model_kwargs["architecture"], ARCHITECTURE_CONV_GRU)
            self.assertIsInstance(predictor.model, TechniqueGraderModel)

    def test_load_predictor_constructs_band_roformer_checkpoint(self) -> None:
        model_kwargs = {
            "architecture": ARCHITECTURE_BAND_ROFORMER,
            "n_mels": 16,
            "hidden_size": 16,
            "roformer_layers": 1,
            "roformer_heads": 4,
            "roformer_ff_size": 32,
            "dropout": 0.0,
        }
        model = build_model_from_config(model_kwargs)
        with tempfile.TemporaryDirectory() as root:
            checkpoint_path = Path(root) / "band_roformer.pth"
            torch.save(
                {
                    "epoch": 1,
                    "architecture": ARCHITECTURE_BAND_ROFORMER,
                    "model_state": model.state_dict(),
                    "model_kwargs": model_kwargs,
                    "train_args": {"max_seconds": 1.0},
                    "val_metrics": {"clip_acc": 0.5},
                },
                checkpoint_path,
            )

            predictor = load_predictor(str(checkpoint_path), device_name="cpu")
            self.assertEqual(predictor.model_kwargs["architecture"], ARCHITECTURE_BAND_ROFORMER)
            self.assertEqual(getattr(predictor.model, "architecture"), ARCHITECTURE_BAND_ROFORMER)


if __name__ == "__main__":
    unittest.main()
