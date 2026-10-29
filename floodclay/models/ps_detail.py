"""原尺寸 PS 的轻量 CNN 分支，在 Clay 编码后补充局部特征。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..masks import modality_mask, zero_invalid
from .blocks import LayerNorm2d


class PSDetailFusion(nn.Module):
    def __init__(self, in_channels: int, feature_channels: int) -> None:
        super().__init__()
        # stages=[] 用来保存各个 CNN 模块
        stages = []
        # 原图为 1024 时：512 → 256 → 128 → 64，通道为 16/32/64/64。
        for out_channels in (16, 32, 64, 64):
            stages.append(nn.Sequential(
                nn.Conv2d(
                    # stride=2	卷积每次移动两个像素，使宽高约减半
                    # 卷积核为3
                    in_channels, in_channels, 3, stride=2, padding=1,
                    groups=in_channels, bias=False,
                ),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                LayerNorm2d(out_channels),
                nn.GELU(),
            ))
            in_channels = out_channels
        self.cnn = nn.Sequential(*stages)
        self.projection = nn.Conv2d(in_channels, feature_channels, 1, bias=False)
        # 第一轮反向先学习融合系数，之后梯度进入 CNN。
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(
        self, native_ps: torch.Tensor, ps_feature: torch.Tensor,
        available: torch.Tensor,
    ) -> torch.Tensor:
        available = available.to(device=ps_feature.device, dtype=torch.bool)
        feature_mask = modality_mask(available, ps_feature)
        if not bool(available.any()):
            return zero_invalid(ps_feature, feature_mask)

        # CNN 前屏蔽原图，融合后再次屏蔽，避免学生读取被隐藏的 PS。
        pixels = zero_invalid(native_ps, modality_mask(available, native_ps))
        detail = self.cnn(pixels)
        # 在低通道阶段对齐空间尺寸，避免保留更大的 1024 通道特征图。
        if detail.shape[-2:] != ps_feature.shape[-2:]:
            detail = F.interpolate(
                detail, size=ps_feature.shape[-2:],
                mode="bilinear", align_corners=False,
            )
        detail = self.projection(detail).to(ps_feature.dtype)
        detail = zero_invalid(detail, feature_mask)
        return zero_invalid(ps_feature + self.scale * detail, feature_mask)
