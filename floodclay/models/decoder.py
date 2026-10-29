"""逐级上采样及洪水分割、边界双输出头。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import ConvBlock


# 分割解码器
class SegmentationDecoder(nn.Module):
    def __init__(self, channels: int, output_size: int) -> None:
        super().__init__()
        self.output_size = output_size
        # 规定各阶段的通道数。
        widths = [channels, max(channels // 2, 32), max(channels // 4, 32), 32]
        # 逐步减少通道数
        self.stages = nn.ModuleList(
            [ConvBlock(widths[index], widths[index + 1]) for index in range(len(widths) - 1)]
        )
        # 建立两个独立的输出层。每个都是 1×1 卷积，把 32 通道变成 1 通道
        self.segmentation_head = nn.Conv2d(widths[-1], 1, 1)
        self.boundary_head = nn.Conv2d(widths[-1], 1, 1)

    def forward(self, feature: torch.Tensor) -> dict[str, torch.Tensor]:
        x = feature
        # 对上采样的各个阶段进行下面的操作
        for stage in self.stages:
            # interpolate把特征图的高和宽各放大 2 倍
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            # 用当前的 ConvBlock 处理放大后的特征，并改变通道数
            x = stage(x)
        # 先执行逐位置 1×1 输出头，再上采样单通道 logits。
        # 避免为 1024 输出保留 32 通道的大特征图；损失梯度仍经过插值回传。
        logits = self.segmentation_head(x)
        boundary_logits = self.boundary_head(x)
        size = (self.output_size, self.output_size)
        if logits.shape[-2:] != size:
            logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
            boundary_logits = F.interpolate(boundary_logits, size=size, mode="bilinear", align_corners=False)
        return {
            # 洪水分割的原始预测分数
            "logits": logits,
            # 边界的原始预测分数
            "boundary_logits": boundary_logits,
        }
