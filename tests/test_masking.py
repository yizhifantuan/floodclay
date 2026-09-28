import unittest

import torch

from floodclay.models.modules import ModalityMaskSampler


class MaskSamplerTest(unittest.TestCase):
    def test_drops_one_or_two_but_keeps_one(self) -> None:
        torch.manual_seed(3)
        sampler = ModalityMaskSampler(max_drop=2, drop_probability=1.0)
        availability = torch.ones(64, 3, dtype=torch.bool)
        result = sampler(availability)
        remaining = result.sum(1)
        self.assertTrue(bool(torch.all(remaining >= 1)))
        self.assertTrue(bool(torch.all(remaining <= 2)))

    def test_real_single_modality_is_never_removed(self) -> None:
        sampler = ModalityMaskSampler(max_drop=2, drop_probability=1.0)
        availability = torch.tensor([[False, False, True]])
        self.assertTrue(torch.equal(sampler(availability), availability))


if __name__ == "__main__":
    unittest.main()

