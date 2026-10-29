"""用轻量模型和指定的验证 IoU 序列检查真实训练入口的早停行为。"""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from floodclay.config import load_config
from scripts import train


class TrainingEarlyStoppingTest(unittest.TestCase):
    def run_training(self, scores, *, epochs=100, patience=20):
        temp_root = train.PROJECT / "tmp"
        temp_root.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=temp_root) as directory:
            config = load_config()
            config["training"].update(
                output_dir=directory, epochs=epochs, early_stopping_patience=patience,
            )
            model = torch.nn.Linear(1, 1)
            model.alphaearth_enabled = False
            model.encoder = SimpleNamespace(ps_encoder_size=512, ps_detail=None)
            loader = SimpleNamespace(dataset=[0])
            validation_scores = iter(scores)

            def fake_epoch(*args, optimizer=None, **kwargs):
                if optimizer is not None:
                    optimizer.step()
                    return {"total": 1.0}, {"iou": 0.0}
                return {"total": 1.0}, {"iou": next(validation_scores)}

            output = StringIO()
            with (
                patch.object(train, "load_config", return_value=config),
                patch.object(train, "make_loaders", return_value=(loader, loader, {})),
                patch.object(train, "build_model", return_value=model),
                patch.object(train, "run_epoch", side_effect=fake_epoch),
                patch.object(train, "save_checkpoint") as save,
                patch.object(torch.cuda, "is_available", return_value=False),
                redirect_stdout(output),
            ):
                checkpoint = train.main()

            self.assertEqual(checkpoint.name, "best.pt")
            self.assertEqual(checkpoint.parent.parent, Path(directory))
            snapshot = json.loads((checkpoint.parent / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(snapshot["training"]["output_dir"], str(checkpoint.parent))
            self.assertTrue((checkpoint.parent / "split_manifest.json").is_file())
            self.assertTrue(all(call.args[0].parent == checkpoint.parent for call in save.call_args_list))
            history = [
                json.loads(row)
                for row in (checkpoint.parent / "history.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            saved = [(call.args[0].name, call.args[3]) for call in save.call_args_list]
            return history, saved, output.getvalue()

    def test_twenty_equal_or_worse_epochs_stop_and_keep_best(self):
        history, saved, output = self.run_training([0.7] + [0.7, 0.6] * 10)
        self.assertEqual(len(history), 21)
        self.assertEqual([epoch for name, epoch in saved if name == "best.pt"], [1])
        self.assertIn(("last.pt", 21), saved)
        self.assertIn("提前停止训练", output)
        self.assertTrue(all(a["lr"] > b["lr"] for a, b in zip(history, history[1:])))

    def test_new_best_resets_patience(self):
        history, saved, _ = self.run_training([0.5] * 19 + [0.6] + [0.55] * 20)
        self.assertEqual(len(history), 40)
        self.assertEqual([epoch for name, epoch in saved if name == "best.pt"], [1, 20])
        self.assertIn(("last.pt", 40), saved)

    def test_epoch_limit_still_applies(self):
        history, _, output = self.run_training([0.5] * 10, epochs=10)
        self.assertEqual(len(history), 10)
        self.assertNotIn("提前停止训练", output)

    def test_patience_is_configurable(self):
        history, _, _ = self.run_training([0.5] * 3, patience=2)
        self.assertEqual(len(history), 3)

    def test_invalid_patience_is_rejected_before_training(self):
        for patience in (0, -1, True, 1.5):
            with self.subTest(patience=patience):
                with self.assertRaisesRegex(ValueError, "early_stopping_patience"):
                    self.run_training([], patience=patience)


if __name__ == "__main__":
    unittest.main()
