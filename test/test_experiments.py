"""检查实验隔离、默认权重选择和训练后评估的路径衔接。"""

from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from floodclay.config import load_config
from floodclay.engine import read_checkpoint
from floodclay.engine.experiments import create_experiment_dir, latest_best_checkpoint
from scripts import train as train_module, train_and_evaluate


class ExperimentsTest(unittest.TestCase):
    def setUp(self):
        temp_root = train_and_evaluate.PROJECT / "tmp"
        temp_root.mkdir(exist_ok=True)
        self.directory = TemporaryDirectory(dir=temp_root)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_same_timestamp_creates_new_directory_and_preserves_previous_files(self):
        with patch("floodclay.engine.experiments.datetime") as clock:
            clock.now.return_value = datetime(2026, 10, 28, 15, 30)
            first = create_experiment_dir(self.root)
            (first / "best.pt").write_bytes(b"previous weights")
            (first / "history.jsonl").write_text("previous history", encoding="utf-8")
            second = create_experiment_dir(self.root)
        self.assertNotEqual(first, second)
        self.assertEqual(first.parent, second.parent)
        self.assertEqual(list(second.iterdir()), [])
        self.assertEqual((first / "best.pt").read_bytes(), b"previous weights")
        self.assertEqual((first / "history.jsonl").read_text(encoding="utf-8"), "previous history")

    def test_default_checkpoint_prefers_latest_experiment_with_weights(self):
        (self.root / "best.pt").write_bytes(b"legacy weights")
        with patch("floodclay.engine.experiments.datetime") as clock:
            clock.now.return_value = datetime(2026, 10, 28, 15, 30)
            first = create_experiment_dir(self.root)
            (first / "best.pt").touch()
            second = create_experiment_dir(self.root)
            (second / "best.pt").touch()
            create_experiment_dir(self.root)  # 模拟尚未生成权重的新实验。
        self.assertEqual(latest_best_checkpoint(self.root), second / "best.pt")

    def test_legacy_checkpoint_remains_usable(self):
        legacy = self.root / "best.pt"
        legacy.touch()
        self.assertEqual(latest_best_checkpoint(self.root), legacy)

    def test_default_root_is_runs_alongside_existing_default(self):
        self.assertEqual(Path(load_config()["training"]["output_dir"]), train_module.PROJECT / "runs")

    def test_legacy_default_checkpoint_is_found_from_runs_root(self):
        legacy = self.root / "default" / "best.pt"
        legacy.parent.mkdir()
        legacy.write_bytes(b"old default weights")
        self.assertEqual(latest_best_checkpoint(self.root), legacy)

    def test_old_nested_run_remains_usable_but_newer_sibling_run_is_selected(self):
        old = self.root / "default" / "run_20261027_153000_000000" / "best.pt"
        old.parent.mkdir(parents=True)
        old.write_bytes(b"old nested weights")
        self.assertEqual(latest_best_checkpoint(self.root), old)
        new = self.root / "run_20261028_153000_000000" / "best.pt"
        new.parent.mkdir()
        new.write_bytes(b"new weights")
        self.assertEqual(latest_best_checkpoint(self.root), new)
        self.assertEqual(old.read_bytes(), b"old nested weights")

    def test_all_epochs_and_final_evaluation_share_one_directory_per_experiment(self):
        legacy = self.root / "default"
        legacy.mkdir()
        for name in ("best.pt", "last.pt", "history.jsonl", "split_manifest.json", "evaluation.json"):
            (legacy / name).write_bytes(b"keep existing default files")
        old_files = {path.name: path.read_bytes() for path in legacy.iterdir()}
        config = load_config()
        config["training"].update(output_dir=str(self.root), epochs=3)
        loader = SimpleNamespace(dataset=[0])
        model = torch.nn.Linear(1, 1)
        model.alphaearth_enabled = False
        model.encoder = SimpleNamespace(ps_encoder_size=512, ps_detail=None)
        validation_calls = 0

        def fake_epoch(*args, optimizer=None, **kwargs):
            nonlocal validation_calls
            if optimizer is not None:
                optimizer.step()
                return {"total": 1.0}, {"iou": 0.0}
            score = (0.5, 0.6, 0.55)[validation_calls % 3]
            validation_calls += 1
            return {"total": 1.0}, {"iou": score}

        def fake_evaluate(checkpoint):
            payload = read_checkpoint(checkpoint)
            self.assertEqual(payload["epoch"], 2)
            checkpoint.with_name("evaluation.json").write_text(
                json.dumps({"checkpoint_epoch": payload["epoch"]}), encoding="utf-8",
            )

        with (
            patch.object(train_module, "load_config", side_effect=lambda *args: deepcopy(config)),
            patch.object(train_and_evaluate, "load_config", side_effect=lambda *args: deepcopy(config)),
            patch.object(train_module, "make_loaders", return_value=(loader, loader, {})),
            patch.object(train_module, "build_model", return_value=model),
            patch.object(train_module, "run_epoch", side_effect=fake_epoch),
            patch.object(train_module, "create_experiment_dir", wraps=create_experiment_dir) as create,
            patch.object(train_and_evaluate, "evaluate_main", side_effect=fake_evaluate) as evaluate,
            patch.object(torch.cuda, "is_available", return_value=False),
            patch("floodclay.engine.experiments.datetime") as clock,
            redirect_stdout(StringIO()),
        ):
            clock.now.return_value = datetime(2026, 10, 28, 15, 30)
            train_and_evaluate.main()
            first_runs = list(self.root.glob("run_*"))
            self.assertEqual(len(first_runs), 1)
            first_run = first_runs[0]
            first_files = {path.name: path.read_bytes() for path in first_run.iterdir()}
            train_and_evaluate.main()
            self.assertEqual(create.call_count, 2)  # 两次完整实验，而非六个 epoch 各建目录。
            self.assertEqual(evaluate.call_count, 2)

        runs = sorted(self.root.glob("run_*"))
        self.assertEqual(len(runs), 2)
        for run in runs:
            self.assertTrue(all(path.is_file() for path in run.iterdir()))
            self.assertEqual({path.name for path in run.iterdir()}, {
                "config.json", "split_manifest.json", "history.jsonl",
                "last.pt", "best.pt", "evaluation.json",
            })
            rows = (run / "history.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual([json.loads(row)["epoch"] for row in rows], [1, 2, 3])
            self.assertEqual(read_checkpoint(run / "last.pt")["epoch"], 3)
            self.assertEqual(read_checkpoint(run / "best.pt")["epoch"], 2)
            snapshot = json.loads((run / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(Path(snapshot["training"]["output_dir"]), run)
        self.assertEqual({path.name: path.read_bytes() for path in legacy.iterdir()}, old_files)
        self.assertEqual({path.name: path.read_bytes() for path in first_run.iterdir()}, first_files)

    def test_no_checkpoint_reports_missing_weights(self):
        create_experiment_dir(self.root)
        with self.assertRaises(FileNotFoundError):
            latest_best_checkpoint(self.root)

    def test_combined_entry_evaluates_exact_checkpoint_returned_by_training(self):
        (self.root / "best.pt").write_bytes(b"old weights")
        checkpoint = create_experiment_dir(self.root) / "best.pt"
        checkpoint.write_bytes(b"new weights")
        with (
            patch.object(train_and_evaluate, "load_config", return_value={
                "training": {"epochs": 100, "output_dir": str(self.root)},
            }),
            patch.object(train_and_evaluate, "train_main", return_value=checkpoint),
            patch.object(train_and_evaluate, "evaluate_main") as evaluate,
            patch.object(train_and_evaluate.torch.cuda, "is_available", return_value=False),
            redirect_stdout(StringIO()) as output,
        ):
            train_and_evaluate.main()
        evaluate.assert_called_once_with(checkpoint)
        self.assertIn(str(checkpoint.with_name("evaluation.json")), output.getvalue())
        self.assertEqual((self.root / "best.pt").read_bytes(), b"old weights")

    def test_combined_entry_rejects_missing_new_checkpoint(self):
        (self.root / "best.pt").write_bytes(b"old weights")
        checkpoint = create_experiment_dir(self.root) / "best.pt"
        with (
            patch.object(train_and_evaluate, "load_config", return_value={"training": {"epochs": 100}}),
            patch.object(train_and_evaluate, "train_main", return_value=checkpoint),
            patch.object(train_and_evaluate, "evaluate_main") as evaluate,
            patch.object(train_and_evaluate.torch.cuda, "is_available", return_value=False),
            redirect_stdout(StringIO()),
        ):
            with self.assertRaisesRegex(RuntimeError, "训练未生成最优权重"):
                train_and_evaluate.main()
        evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
