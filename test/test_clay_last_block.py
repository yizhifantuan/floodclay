"""用缩小的真实 Clay 编码器验证仅最后一层训练及检查点恢复。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch import nn

from floodclay.config import PROJECT, load_config, load_evaluation_config
from floodclay.engine import load_checkpoint, read_checkpoint, save_checkpoint
from floodclay.losses import MultiLevelDistillationLoss
from floodclay.models import build_model
from floodclay.models.clay_encoder import ClayPatchEncoder, MODEL_SIZES


SMALL_SIZE = {"dim": 32, "depth": 3, "heads": 2, "dim_head": 16, "mlp_ratio": 2}


class ClayLastBlockTest(unittest.TestCase):
    def setUp(self):
        self.num_threads = torch.get_num_threads()
        self.random_state = torch.get_rng_state()
        torch.set_num_threads(1)
        torch.manual_seed(42)
        sizes = patch.dict(MODEL_SIZES, {"test": SMALL_SIZE})
        sizes.start()
        self.addCleanup(sizes.stop)
        weights = patch.object(ClayPatchEncoder, "_load_encoder_weights")
        weights.start()
        self.addCleanup(weights.stop)

    def tearDown(self):
        torch.set_num_threads(self.num_threads)
        torch.set_rng_state(self.random_state)

    def make_encoder(self, **kwargs):
        return ClayPatchEncoder("unused.ckpt", model_size="test", patch_size=2, **kwargs)

    def make_model(self, config=None):
        if config is None:
            config = load_config()
            config["model"].update(
                clay_model_size="test", patch_size=2, ps_encoder_size=8,
                feature_channels=8, feature_size=2, fusion_heads=2,
                adapter_bottleneck_dim=4, adapter_grid_size=2,
                output_size=16,
            )
            config["data"]["label_size"] = 16
        return build_model(config), config

    def make_batch(self):
        return {
            "images": {
                "s1": torch.randn(1, 2, 8, 8),
                "s2": torch.randn(1, 10, 8, 8),
                "ps": torch.randn(1, 4, 16, 16),
            },
            "time": torch.zeros(1, 4),
            "latlon": torch.zeros(1, 4),
            "availability": torch.ones(1, 3, dtype=torch.bool),
        }

    def test_current_default_opens_last_block(self):
        config = load_config()["model"]
        self.assertTrue(config["freeze_clay"])
        self.assertTrue(config["unfreeze_clay_last_block"])

    def test_only_last_block_parameters_are_trainable(self):
        backbone = self.make_encoder(freeze=True, unfreeze_last_block=True)
        for name, parameter in backbone.encoder.named_parameters():
            self.assertEqual(parameter.requires_grad, name.startswith("transformer.layers.2."), name)

    def test_partial_training_modes_survive_train_eval_switches(self):
        backbone = self.make_encoder(freeze=True, unfreeze_last_block=True)
        for mode in (True, False, True):
            backbone.train(mode)
            self.assertFalse(backbone.encoder.training)
            self.assertFalse(backbone.encoder.patch_embedding.training)
            self.assertFalse(backbone.encoder.transformer.layers[0].training)
            self.assertFalse(backbone.encoder.transformer.norm.training)
            self.assertEqual(backbone.encoder.transformer.layers[-1].training, mode)
            self.assertTrue(all(p.requires_grad for p in backbone.encoder.transformer.layers[-1].parameters()))

    def test_old_configs_preserve_fully_frozen_backbone(self):
        _, config = self.make_model()
        config["model"].pop("unfreeze_clay_last_block")
        model, _ = self.make_model(load_evaluation_config({"config": config}))
        self.assertFalse(any(p.requires_grad for p in model.encoder.backbone.parameters()))

    def test_disabling_freeze_still_trains_entire_backbone(self):
        backbone = self.make_encoder(freeze=False, unfreeze_last_block=True)
        self.assertTrue(all(p.requires_grad for p in backbone.encoder.parameters()))
        backbone.train()
        self.assertTrue(all(module.training for module in backbone.encoder.modules()))

    def test_standalone_forward_updates_last_block_but_not_frozen_weights(self):
        backbone = self.make_encoder(freeze=True, unfreeze_last_block=True)
        inputs = (torch.randn(1, 4, 8, 8), torch.zeros(1, 4), torch.zeros(1, 4), torch.ones(4), 3.0)
        tokens, _ = backbone.prepare_tokens(*inputs)
        self.assertFalse(tokens.requires_grad)
        before = {name: parameter.detach().clone() for name, parameter in backbone.encoder.named_parameters()}
        optimizer = torch.optim.AdamW(p for p in backbone.parameters() if p.requires_grad)
        output = backbone(*inputs)
        nn.functional.mse_loss(output, torch.randn_like(output)).backward()
        last = backbone.encoder.transformer.layers[-1][0].to_qkv.weight
        self.assertIsNotNone(last.grad)
        self.assertTrue(bool(torch.isfinite(last.grad).all()))
        self.assertGreater(last.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before["transformer.layers.2.0.to_qkv.weight"], last))
        for name, parameter in backbone.encoder.named_parameters():
            if not parameter.requires_grad:
                self.assertIsNone(parameter.grad, name)
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)

    def test_real_teacher_student_training_updates_shared_last_block_and_restores_checkpoint(self):
        model, config = self.make_model()
        batch = self.make_batch()
        student_mask = torch.tensor([[True, False, True]])
        backbone = model.encoder.backbone.encoder
        before = {name: parameter.detach().clone() for name, parameter in backbone.named_parameters()}
        optimizer = torch.optim.AdamW(p for p in model.parameters() if p.requires_grad)
        criterion = MultiLevelDistillationLoss(config["loss"]["weights"])
        outputs = model(batch, student_mask=student_mask)
        target = torch.randint(2, (1, 1, 16, 16)).float()
        loss, _ = criterion(outputs, target, torch.ones_like(target, dtype=torch.bool))
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        last = backbone.transformer.layers[-1][0].to_qkv.weight
        self.assertGreater(last.grad.abs().sum().item(), 0)
        self.assertTrue(bool(torch.isfinite(last.grad).all()))
        self.assertTrue(any(
            adapter.up.weight.grad is not None and adapter.up.weight.grad.abs().sum().item() > 0
            for adapter in model.encoder.adapters.values()
        ))
        optimizer.step()
        self.assertFalse(torch.equal(before["transformer.layers.2.0.to_qkv.weight"], last))
        for name, parameter in backbone.named_parameters():
            if not parameter.requires_grad:
                self.assertIsNone(parameter.grad, name)
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)

        temp_root = PROJECT / "tmp"
        temp_root.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=temp_root) as directory:
            checkpoint = Path(directory) / "best.pt"
            save_checkpoint(checkpoint, model, optimizer, 1, config, {"iou": 0.5}, split_manifest={})
            payload = read_checkpoint(checkpoint)
            restored_config = load_evaluation_config(payload)
            self.assertTrue(restored_config["model"]["unfreeze_clay_last_block"])
            restored, _ = self.make_model(restored_config)
            load_checkpoint(checkpoint, restored, payload=payload)
        model.eval()
        restored.eval()
        with torch.no_grad():
            expected = model.predict(batch, availability=student_mask)["logits"]
            actual = restored.predict(batch, availability=student_mask)["logits"]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(actual.requires_grad)
        self.assertFalse(restored.encoder.backbone.encoder.transformer.layers[-1].training)


if __name__ == "__main__":
    unittest.main()
