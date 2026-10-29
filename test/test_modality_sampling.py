"""学生七种输入组合的采样分布，以及完整/缺失输入下的训练回归检查。"""

import random
import unittest
from unittest.mock import patch

import torch
from torch import nn

from floodclay.config import load_config
from floodclay.losses import MultiLevelDistillationLoss
from floodclay.models.masking import ModalityMaskSampler
from floodclay.models.network import TeacherStudentFloodModel


SCENARIOS = (
    (True, True, True),
    (False, True, True),
    (True, False, True),
    (True, True, False),
    (True, False, False),
    (False, True, False),
    (False, False, True),
)


class TinyEncoder(nn.Module):
    """只替换大型主干，保留真实学生融合、解码器和全部训练损失。"""

    def __init__(self, *args):
        super().__init__()
        self.output_dim = 8
        self.projection = nn.Conv2d(1, self.output_dim, 1)

    def forward(self, images, time, latlon, availability):
        features = {
            name: self.projection(images[name]) * availability[:, index, None, None, None]
            for index, name in enumerate(("s1", "s2", "ps"))
        }
        count = availability.sum(dim=1).clamp_min(1)[:, None, None, None]
        features["shared"] = torch.stack(list(features.values())).sum(dim=0) / count
        return features


class ModalitySamplingTest(unittest.TestCase):
    def setUp(self):
        self.random_state = random.getstate()
        self.torch_state = torch.get_rng_state()
        self.num_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        random.seed(42)
        torch.manual_seed(42)

    def tearDown(self):
        random.setstate(self.random_state)
        torch.set_rng_state(self.torch_state)
        torch.set_num_threads(self.num_threads)

    def test_default_config_enables_all_seven_combinations(self):
        config = load_config()["model"]
        self.assertEqual(config["max_random_drop"], 2)
        self.assertAlmostEqual(config["drop_probability"], 6 / 7)

    def test_default_sampling_is_approximately_uniform_over_all_seven_combinations(self):
        available = torch.ones(7000, 3, dtype=torch.bool)
        masks = ModalityMaskSampler()(available)
        combinations, counts = torch.unique(masks, dim=0, return_counts=True)
        self.assertEqual({tuple(row) for row in combinations.tolist()}, set(SCENARIOS))
        # 7000 次中每种期望 1000 次；容差可检测漏采完整输入或按删除数量不均衡采样。
        self.assertTrue(bool(((counts > 800) & (counts < 1200)).all()), counts.tolist())
        self.assertTrue(bool(available.all()))

    def test_seed_reproduces_the_same_masks(self):
        available = torch.ones(40, 3, dtype=torch.bool)
        sampler = ModalityMaskSampler()
        first = sampler(available)
        random.seed(42)
        torch.manual_seed(42)
        torch.testing.assert_close(first, sampler(available))

    def test_sampling_never_restores_absent_modalities_or_removes_all_observations(self):
        available = torch.tensor(SCENARIOS).repeat(20, 1)
        masks = ModalityMaskSampler(drop_probability=1.0)(available)
        self.assertFalse(bool((masks & ~available).any()))
        self.assertTrue(bool(masks.any(dim=1).all()))
        self.assertEqual(masks.dtype, torch.bool)

    def test_probability_zero_always_preserves_complete_input(self):
        available = torch.ones(10, 3, dtype=torch.bool)
        with patch("floodclay.models.masking.random.random", return_value=0.0):
            masks = ModalityMaskSampler(drop_probability=0.0)(available)
        torch.testing.assert_close(masks, available)

    def test_explicit_probability_one_preserves_missing_only_mode(self):
        masks = ModalityMaskSampler(drop_probability=1.0)(torch.ones(30, 3, dtype=torch.bool))
        self.assertTrue(bool((masks.sum(dim=1) < 3).all()))
        self.assertTrue(bool(masks.any(dim=1).all()))

    def test_configured_maximum_drop_is_respected(self):
        masks = ModalityMaskSampler(max_drop=1)(torch.ones(30, 3, dtype=torch.bool))
        self.assertTrue(bool((masks.sum(dim=1) >= 2).all()))

    def make_training_case(self):
        config = load_config()
        model_config = dict(config["model"], feature_channels=8, feature_size=2, fusion_heads=2, output_size=16)
        with patch("floodclay.models.network.SharedClayEncoder", TinyEncoder):
            model = TeacherStudentFloodModel(model_config, config["sensors"])
        batch = {
            "images": {name: torch.randn(1, 1, 4, 4) for name in ("s1", "s2", "ps")},
            "time": torch.zeros(1, 4),
            "latlon": torch.zeros(1, 4),
            "availability": torch.ones(1, 3, dtype=torch.bool),
        }
        target = torch.randint(2, (1, 1, 16, 16)).float()
        valid = torch.ones_like(target, dtype=torch.bool)
        return model, MultiLevelDistillationLoss(config["loss"]["weights"]), batch, target, valid

    def test_sampled_complete_input_trains_the_student_with_zero_reconstruction_loss(self):
        model, criterion, batch, target, valid = self.make_training_case()
        # 走真实默认采样入口，强制进入完整输入分支。
        with patch("floodclay.models.masking.random.random", return_value=0.99):
            outputs = model(batch)
        self.assertTrue(bool(outputs["student_mask"].all()))
        loss, parts = criterion(outputs, target, valid)
        self.assertEqual(parts["reconstruction"].item(), 0.0)
        self.assertTrue(bool(torch.isfinite(loss)))
        before = model.student_decoder.segmentation_head.weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss.backward()
        gradient = model.student_decoder.segmentation_head.weight.grad
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(gradient.abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.student_decoder.segmentation_head.weight))

    def test_all_six_missing_combinations_have_finite_training_gradients(self):
        model, criterion, batch, target, valid = self.make_training_case()
        for scenario in SCENARIOS[1:]:
            with self.subTest(scenario=scenario):
                model.zero_grad(set_to_none=True)
                outputs = model(batch, student_mask=torch.tensor([scenario]))
                loss, parts = criterion(outputs, target, valid)
                self.assertTrue(all(bool(torch.isfinite(value)) for value in parts.values()))
                loss.backward()
                gradient = model.student_decoder.segmentation_head.weight.grad
                self.assertTrue(bool(torch.isfinite(gradient).all()))
                self.assertGreater(gradient.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
