import unittest
from unittest.mock import patch

import torch
from torch import nn

from floodclay.losses import MultiLevelDistillationLoss
from floodclay.models.network import TeacherStudentFloodModel


SENSORS = {
    "s1": {"waves": [3.5, 4.0], "gsd": 10.0},
    "s2": {"waves": [0.49, 0.56, 0.66], "gsd": 10.0},
    "ps": {"waves": [0.49, 0.56, 0.66, 0.86], "gsd": 3.0},
}


class StubClayPatchEncoder(nn.Module):
    """Small stand-in so model tests do not need the external Clay checkpoint."""

    def __init__(self, dim: int = 32, patch_size: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.projection = nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)

    def forward(
        self,
        pixels: torch.Tensor,
        time: torch.Tensor,
        latlon: torch.Tensor,
        waves: torch.Tensor,
        gsd: float,
    ) -> torch.Tensor:
        del time, latlon, waves, gsd
        return self.projection(pixels.mean(dim=1, keepdim=True))


class ModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1)
        config = {
            "clay_checkpoint": "unused-in-tests.ckpt",
            "clay_model_size": "large",
            "freeze_clay": True,
            "patch_size": 8,
            "feature_channels": 32,
            "feature_size": 4,
            "fusion_heads": 4,
            "max_random_drop": 2,
            "drop_probability": 1.0,
            "output_size": 32,
        }
        with patch(
            "floodclay.models.clay_encoder.ClayPatchEncoder",
            return_value=StubClayPatchEncoder(),
        ):
            self.model = TeacherStudentFloodModel(config, SENSORS)

    def _batch(self) -> dict[str, object]:
        return {
            "images": {
                "s1": torch.randn(2, 2, 32, 32),
                "s2": torch.randn(2, 3, 32, 32),
                "ps": torch.randn(2, 4, 32, 32),
            },
            "availability": torch.ones(2, 3, dtype=torch.bool),
            "time": torch.zeros(2, 4),
            "latlon": torch.zeros(2, 4),
        }

    def test_forward_and_losses(self) -> None:
        batch = self._batch()
        mask = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)
        outputs = self.model(batch, student_mask=mask)
        self.assertEqual(tuple(outputs["student"]["logits"].shape), (2, 1, 32, 32))
        self.assertTrue(torch.equal(outputs["student_mask"], mask))
        target = (torch.rand(2, 1, 32, 32) > 0.7).float()
        valid = torch.ones_like(target, dtype=torch.bool)
        weights = {
            "teacher_seg": 1,
            "student_seg": 1,
            "boundary_supervision": 0.25,
            "shared_kd": 0.5,
            "boundary_kd": 0.25,
            "prediction_kd": 0.5,
            "reconstruction": 0.5,
        }
        loss, _ = MultiLevelDistillationLoss(weights)(outputs, target, valid)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertIsNotNone(self.model.generator.missing_tokens.grad)

    def test_natural_missing_inference(self) -> None:
        batch = self._batch()
        batch["availability"] = torch.tensor([[1, 0, 1], [0, 0, 1]], dtype=torch.bool)
        output = self.model.predict(batch)
        self.assertEqual(tuple(output["logits"].shape), (2, 1, 32, 32))

    def test_native_sizes_are_aligned_after_encoding(self) -> None:
        batch = self._batch()
        batch["images"]["s1"] = torch.randn(2, 2, 30, 38)
        batch["images"]["s2"] = torch.randn(2, 3, 40, 48)
        batch["images"]["ps"] = torch.randn(2, 4, 64, 64)
        with torch.no_grad():
            features = self.model.encoder(batch["images"], batch["time"], batch["latlon"])
            self.assertEqual(tuple(features["s1"].shape), (2, 32, 4, 5))
            self.assertEqual(tuple(features["s2"].shape), (2, 32, 5, 6))
            self.assertEqual(tuple(features["ps"].shape), (2, 32, 8, 8))
            aligned = self.model.aligner(features)
            self.assertEqual(tuple(aligned.shape), (2, 3, 32, 4, 4))
            output = self.model(batch)
            self.assertEqual(tuple(output["student"]["logits"].shape), (2, 1, 32, 32))


if __name__ == "__main__":
    unittest.main()
