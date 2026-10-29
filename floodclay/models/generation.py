"""根据可用观测和共享表示生成缺失模态特征。"""

from __future__ import annotations

import torch
from torch import nn

from ..masks import modality_mask, zero_invalid
from .blocks import ConvBlock


# 缺失模态生成
class MissingModalityGenerator(nn.Module):

    def __init__(self, modalities: int, channels: int) -> None:
        super().__init__()
        self.modalities = modalities
        # 创建缺失模态标记；* 0.02把随机数缩小，方便训练
        # nn.Parameter 将它注册为模型参数，参与梯度计算和优化器更新
        # randn生成一个指定行数、列数的张量里面生成随机数
        self.missing_tokens = nn.Parameter(torch.randn(modalities, channels) * 0.02)
        # 为每个模态创建生成器；n.ModuleList 是 PyTorch 专门用于保存多个神经网络模块的列表
        self.generators = nn.ModuleList(
            [ConvBlock(channels * 3, channels) for _ in range(modalities)]
        )

    def forward(
        self,
        features: torch.Tensor,
        available: torch.Tensor,
        shared: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, channels, height, width = features.shape
        # 模态状态只广播，不展开成完整空间掩码。
        observed_mask = modality_mask(available, features)
        weights = observed_mask.to(features.dtype)
        # 计算已有模态的平均特征
        # 首先屏蔽缺失模态；然后对模态求和；sum(1)表示沿着第一个维度求和
        # clamp_min(1.0)如果数值小于1，就强制设置为1
        observed = zero_invalid(features, observed_mask).sum(1) / weights.sum(1).clamp_min(1.0)
        # 创建结果列表
        # 保存每个模态的生成结果
        generated = []
        # 保存最终使用的特征
        completed = []
        # 遍历每一个模态
        for index in range(self.modalities):
            # 取出当前模态的token
            # .view修改形状
            # .expand将token扩展成特征图
            token = self.missing_tokens[index].view(1, channels, 1, 1).expand(
                batch, -1, height, width
            )
            # torch.cat将三种信息进行拼接
            # dim=1表示再通道维度上面的拼接
            # 生成当前模态的候选特征；generators当前模态专属的生成器
            proposal = self.generators[index](torch.cat((observed, shared, token), dim=1))
            # 加入generated列表中
            generated.append(proposal)
            # : 表示取所有样本，index 表示当前模态
            mask = observed_mask[:, index]
            # 选择使用真实特征或者生成特征
            completed.append(torch.where(mask, features[:, index], proposal))
            # [
            #     [B, C, H, W],
            #     [B, C, H, W],
            #     ...
            # ]
            #torch.stack(completed, dim=1)为他增加一个模态维度
            # 变成[B, M, C, H, W]
        return torch.stack(completed, dim=1), torch.stack(generated, dim=1)
