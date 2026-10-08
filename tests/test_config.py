import unittest
from pathlib import Path

from floodclay.config import PROJECT, load_config


class ConfigTest(unittest.TestCase):
    def test_paths_and_command_line_overrides_share_project_base(self):
        config = load_config(
            PROJECT / "configs" / "default.json",
            data_root="datasets/FloodPlanet",
            clay_checkpoint="checkpoints/custom.ckpt",
        )
        self.assertEqual(
            Path(config["data"]["root"]), (PROJECT / "datasets/FloodPlanet").resolve()
        )
        self.assertEqual(
            Path(config["model"]["clay_checkpoint"]), (PROJECT / "checkpoints/custom.ckpt").resolve()
        )
        self.assertTrue(Path(config["training"]["output_dir"]).is_absolute())
        self.assertNotIn("_config_path", config)

    def test_absolute_paths_are_preserved(self):
        config = load_config(
            PROJECT / "configs" / "default.json",
            data_root=PROJECT / "datasets",
            clay_checkpoint=PROJECT / "checkpoints/custom.ckpt",
        )
        self.assertEqual(Path(config["data"]["root"]), PROJECT / "datasets")
        self.assertEqual(Path(config["model"]["clay_checkpoint"]), PROJECT / "checkpoints/custom.ckpt")
