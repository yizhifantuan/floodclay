# 逐空间位置的跨模态自注意力与直接使用缺失编码的门控融合。

from __future__ import annotations

import torch
from torch import nn

from .blocks import ConvBlock

# 对特征进行融合
class CrossModalAttentionFusion(nn.Module):

    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("Feature channels must be divisible by attention heads")
        # 让同一像素位置的各模态特征互相交换信息
        # batch_first=True 输入、输出张量的第一维是批次维度
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        # 创建一个归一化层，对特征进行归一化处理
        self.norm = nn.LayerNorm(channels)
        # 每个模态的 C 维特征拼接 1 位缺失编码，再计算融合分数
        # 创建了一个门控评分网络，给每个空间位置上的每个模态计算一个融合评分
        self.gate = nn.Sequential(
            # 全连接层特征从 channels + 1 维映射到 channels // 2 维
            nn.Linear(channels + 1, channels // 2),
            nn.GELU(),
            nn.Linear(channels // 2, 1)
        )
        # 融合
        self.output = ConvBlock(channels, channels)

    def forward(
        self, features: torch.Tensor, original_availability: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, modalities, channels, height, width = features.shape
        # 根据模态可用性生成与 1 个模态缺失编码，此时0为存在，1为缺失
        # 先保持布尔编码，拼接前再转换为 tokens 的类型。
        # 取反把模态是否可用改成模态是否缺失
        missing_code = ~original_availability.bool()
        # 每个空间位置使用同一组缺失编码，形状为 [B×H×W, M, 1]。
        # 首先.view增加维度，然后.expand扩展空间维度，最后合并维度为.reshape
        missing_code = missing_code.view(batch, 1, 1, modalities, 1).expand(
            -1, height, width, -1, -1
        ).reshape(-1, modalities, 1)
        # 把形状改成 [B×H×W, M, C]。可以理解为：把每个像素当成一个小组，每组有 M 个模态特征
        # [B,M,C,H,W]->[B,H,W,M,C] 也进行合并
        tokens = features.permute(0, 3, 4, 1, 2).reshape(-1, modalities, channels)
        # 对每个像素小组做模态间的自注意力
        # need_weights=False：不要求返回注意力权重
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        # 把注意力结果加回原特征，再归一化。这一步保留了原来的信息，同时加入了其他模态的信息
        tokens = self.norm(tokens + attended)
        # 将缺失编码直接拼接到特征后，真实特征和生成特征都参与融合。
        gate_input = torch.cat((tokens, missing_code.to(tokens.dtype)), dim=-1)
        # gate 给每个模态算一个分数，形状从 [B×H×W, M, 1] 经 squeeze 变成 [B×H×W, M]
        # softmax(dim=1) 把同一像素的 M 个分数变成权重，这些权重相加等于 1。
        weights = torch.softmax(self.gate(gate_input).squeeze(-1), dim=1)
        # 按权重对模态特征求和
        fused = (tokens * weights.unsqueeze(-1)).sum(dim=1)
        # 把像素重新排回图像，得到 [B, C, H, W]
        fused = fused.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        # 也把权重排回图像，得到 [B, M, H, W]。它能显示每个像素对各模态分配了多少权重。
        weight_map = weights.reshape(batch, height, width, modalities).permute(0, 3, 1, 2)
        return self.output(fused), weight_map
