"""按通道进行 LayerNorm 的二维特征归一化和残差卷积块。"""

from __future__ import annotations

import torch
from torch import nn


class LayerNorm2d(nn.LayerNorm):
    """对每个空间位置的通道归一化，输入输出均为 [B,C,H,W]。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)  # [B,H,W,C]
        x = super().forward(x)
        return x.permute(0, 3, 1, 2)  # [B,C,H,W]


# 一个模块
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            # 让神经网络里的数据更加稳定，方便训练
            LayerNorm2d(out_channels),
            # 激活函数
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU(),
        )
        # 进行残差连接
        self.skip = (
            nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        )
    # 前向传播
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)
