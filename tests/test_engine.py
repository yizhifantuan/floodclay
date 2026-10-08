import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.utils.data import DataLoader

from floodclay.engine import load_checkpoint, run_epoch, save_checkpoint


class RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.calls = []

    def forward(self, batch, student_mask=None):
        self.calls.append((self.training, torch.is_grad_enabled(), student_mask))
        return {"student": {"logits": self.bias.expand_as(batch["target"])}}


class MeanSquaredLoss(nn.Module):
    def forward(self, outputs, target, valid):
        loss = (outputs["student"]["logits"] - target).square().mean()
        return loss, {"total": loss}


def make_loader(values, batch_size):
    return DataLoader([
        {
            "availability": torch.ones(3, dtype=torch.bool),
            "target": torch.full((1, 2, 2), float(value)),
            "valid": torch.ones(1, 2, 2, dtype=torch.bool),
        }
        for value in values
    ], batch_size=batch_size)


class EpochTest(unittest.TestCase):
    def setUp(self):
        self.model = RecordingModel()
        self.device = torch.device("cpu")
        self.scaler = torch.amp.GradScaler("cpu", enabled=False)

    def test_validation_weights_last_batch_and_cycles_masks(self):
        losses, metrics = run_epoch(
            self.model, make_loader([1, 1, 3, 3, 5], 2),
            MeanSquaredLoss(), self.device, self.scaler,
        )
        self.assertAlmostEqual(losses["total"], 9.0)
        self.assertIn("iou", metrics)
        self.assertEqual(self.model.bias.item(), 0.0)
        for step, (training, gradients, mask) in enumerate(self.model.calls):
            self.assertFalse(training)
            self.assertFalse(gradients)
            self.assertTrue(torch.all(~mask[:, step]))
            self.assertTrue(torch.all(mask.sum(dim=1) == 2))

    def test_training_updates_parameters_with_disabled_scaler(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        run_epoch(
            self.model, make_loader([1, 1], 2),
            MeanSquaredLoss(), self.device, self.scaler,
            optimizer=optimizer,
        )
        self.assertGreater(self.model.bias.item(), 0.0)
        self.assertEqual(self.model.calls, [(True, True, None)])

    def test_checkpoint_restores_parameters_and_metadata(self):
        optimizer = torch.optim.AdamW(self.model.parameters())
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            config = {"training": {"epochs": 2}}
            save_checkpoint(path, self.model, optimizer, 2, config, {"iou": 0.7})
            with torch.no_grad():
                self.model.bias.fill_(3)
            payload = load_checkpoint(path, self.model)
        self.assertEqual(self.model.bias.item(), 0.0)
        self.assertEqual(payload["epoch"], 2)
        self.assertEqual(payload["config"], config)
        self.assertEqual(payload["metrics"], {"iou": 0.7})
