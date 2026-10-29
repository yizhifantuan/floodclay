"""模态独立的通道投影与统一空间尺度。"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn

from .blocks import LayerNorm2d

# 把不同模态特征统一成相同通道数和相同 H×W
class ScaleAligner(nn.Module):
    # 初始化函数
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modalities,
        output_size: int,
    ) -> None:
        # 继承父类
        super().__init__()
        # 模态赋值；tuple是元组
        self.modalities = tuple(modalities)
        self.output_size = output_size
        # 给每种模态准备一个自己的小网络
        # 实际上创建一个字典
        self.adapters = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    # 每个空间位置的通道分别进行 LayerNorm
                    LayerNorm2d(out_channels),
                    nn.GELU(),
                )
                for name in self.modalities
            }
        )
    # 函数运行流程
    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        # 创建一个空列表
        aligned = []
        size = (self.output_size, self.output_size)
        # 逐个读取
        for modality in self.modalities:
            # 用不同模态各自的小网络来统一模态特征图
            feature = self.adapters[modality](features[modality])
            # 调整图片/特征图大小
            # F.interpolate()用于调整特征图尺寸的函数
            if feature.shape[-2:] != size:
                feature = F.interpolate(feature, size=size, mode="bilinear", align_corners=False)
            # 把每个模态处理结果放进列表
            aligned.append(feature)
            # B = 一批有多少个样本
            # M = 有多少种模态
            # C = 每个位置多少个特征
            # H = 高
            # W = 宽
        return torch.stack(aligned, dim=1)  # [B, M, C, H, W]
