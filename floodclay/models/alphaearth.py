"""AlphaEarth 辅助分支：掩码池化、64 维投影、门控残差融合。"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ..masks import zero_invalid

class AlphaEarthFusion(nn.Module):
    def __init__(self, channels: int, hidden_channels: int = 64,
                 residual_scale: float = 0.1) -> None:
        super().__init__()
        # 建立了一个通道投影网络；定义两个卷积层
        self.projection = nn.Sequential(
            nn.Conv2d(64, hidden_channels, 1, bias=False), nn.GELU(),
            nn.Conv2d(hidden_channels, channels, 1, bias=False),
        )
        # 决定融合多少 AlphaEarth 信息
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2 + 1, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, 1),
            nn.Sigmoid(),
        )
        # 可学习缩放系数
        self.scale = nn.Parameter(torch.tensor(float(residual_scale)))

    #
    def forward(self, fused: torch.Tensor, embedding: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if (embedding.ndim != 4 or embedding.shape[1] != 64
                or embedding.shape[0] != fused.shape[0]
                or valid.shape != (embedding.shape[0], 1, *embedding.shape[-2:])):
            raise ValueError("AlphaEarth 形状应为 [B,64,H,W]，掩码应为 [B,1,H,W]")
        # 再次检查有效像素
        valid = valid.bool() & torch.isfinite(embedding).all(dim=1, keepdim=True)
        # 将无效位置设置为 0
        safe = zero_invalid(embedding.float(), valid)
        # 将 AlphaEarth 特征对齐到 fused 的空间尺寸
        # valid.float()True 转成 1，False 转成 0
        # size=fused.shape[-2:]表示将有效性掩码调整到 fused 特征图的高度和宽度
        support = F.interpolate(valid.float(), size=fused.shape[-2:], mode="area")
        # 先把特征图缩放到和fused相同的大小，并且只计算有效像素的平均值
        pooled = F.interpolate(safe, size=fused.shape[-2:], mode="area") / support.clamp_min(1e-6)
        # 计算 pooled 特征图中，每个空间位置上 64 个通道组成的特征向量的 L2 范数
        norm = torch.linalg.vector_norm(pooled, dim=1, keepdim=True)
        # 判断 AlphaEarth 特征图中哪些空间位置是有效且可以使用的，生成一个布尔类型的有效性掩码 usable
        usable = (support > 0) & (norm > 1e-6)
        pooled = zero_invalid(pooled / norm.clamp_min(1e-6), usable)
        projected = self.projection(pooled.to(fused.dtype))
        mask = usable.to(fused.dtype)
        gate = self.gate(torch.cat((fused, projected, mask), dim=1))
        # 无效位置的 pooled 已为零，且 projection 无偏置，projected 仍为零，无需再次乘掩码。
        return fused + self.scale.to(fused.dtype) * gate * projected
