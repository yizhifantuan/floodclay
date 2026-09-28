from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

# 给 GroupNorm 找一个合适的分组数量
def _groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1

# 一个模块
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            # 让神经网络里的数据更加稳定，方便训练
            nn.GroupNorm(_groups(out_channels), out_channels),
            # 激活函数
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
        )
        # 进行残差连接
        self.skip = (
            nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        )
    # 前向传播
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)

# 随机让模态“消失”
class ModalityMaskSampler(nn.Module):
    # max_drop：最多删除的模态；drop_probability多少概率进行一次数据丢失
    def __init__(self, max_drop: int = 2, drop_probability: float = 1.0) -> None:
        super().__init__()
        self.max_drop = max_drop
        self.drop_probability = drop_probability

    # 下面这个函数不需要计算梯度
    @torch.no_grad()
    def forward(self, availability: torch.Tensor) -> torch.Tensor:
        # 获取原来的模态缺失编码；.clone()为转为文本
        result = availability.bool().clone()
        # 逐个进行处理
        for batch_index in range(result.shape[0]):
            # 首先找到当前样本哪些模态存在
            # [0]是把里面的答案例如 [0, 2] 取出来
            present = torch.where(result[batch_index])[0]
            # 如果只有一个模态就不删除，后面是在随机决定：这次到底删不删
            # torch.rand(())：随机抽一个 0 到 1 之间的数
            # self.drop_probability：设定的隐藏概率
            # device=result.device：让随机数和 result 放在同一设备上
            if present.numel() <= 1 or torch.rand((), device=result.device) > self.drop_probability:
                continue
            # 决定最多删除几个
            maximum = min(self.max_drop, int(present.numel()) - 1)
            # 随机决定删除几个
            count = int(torch.randint(1, maximum + 1, (), device=result.device).item())
            # 随机决定删除谁；present.numel()计算present里面有多少数字
            dropped = present[torch.randperm(present.numel(), device=result.device)[:count]]
            # 模拟缺失
            result[batch_index, dropped] = False
        return result

# 把不同模态特征统一成相同通道数和相同 H×W
class ScaleAligner(nn.Module):
    # 初始化函数
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modalities: Iterable[str],
        output_size: int,
    ) -> None:
        # 继承父类
        super().__init__()
        # 模态赋值
        self.modalities = tuple(modalities)
        self.output_size = output_size
        # 给每种模态准备一个自己的小网络
        self.adapters = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.GroupNorm(_groups(out_channels), out_channels),
                    nn.GELU(),
                )
                for name in self.modalities
            }
        )
    # 函数运行流程
    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        # 创建一个空列表
        aligned = []
        # 逐个读取
        for modality in self.modalities:
            # 用不同模态各自的小网络来统一模态特征图
            # input_feature = features["s1"]       # 取出 S1 的原始特征
            # adapter = self.adapters["s1"]        # 取出专门处理 S1 的小网络
            # feature = adapter(input_feature)     # 把特征送进小网络，得到处理后的特征
            feature = self.adapters[modality](features[modality])
            # 调整图片/特征图大小
            feature = F.interpolate(
                feature,
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            )
            # 把每个模态处理结果放进列表
            aligned.append(feature)
            # B = 一批有多少个样本
            # M = 有多少种模态
            # C = 每个位置多少个特征
            # H = 高
            # W = 宽
        return torch.stack(aligned, dim=1)  # [B, M, C, H, W]

# 提取共享信息代码
class SharedRepresentation(nn.Module):
    def __init__(self, modalities: int, channels: int) -> None:
        super().__init__()
        # 创建一张可以通过训练学会的状态提示表
        # 三个模态，每个模态两种状态，每种情况对应channels个数字
        # torch.randn(...)先随机生成这些数字；* 0.02让数字足够小；nn.Parameter(...)告诉pytorgh这些参数是需要学习的
        # state_embeddings给不同特征加上身份编码，让网络分清楚特征来自哪个模态
        self.state_embeddings = nn.Parameter(torch.randn(modalities, 2, channels) * 0.02)
        # 用于进一步处理最后得到的共享特征
        self.refine = ConvBlock(channels, channels)
    # 输入features和缺失编码
    def forward(self, features: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        # 获取批次；模态；通道
        batch, modalities, channels, _, _ = features.shape
        # 进行转置操作
        # [
        #  [1,0,1],
        #  [1,1,0]
        # ]
        # 变成
        # [
        #  [1,1],   ← S1
        #  [0,1],   ← S2
        #  [1,0]    ← PS
        # ]
        state = available.long().transpose(0, 1)
        # 根据状态取 embedding；state_embeddings里面包括了每个模态的两种状态的身份编码
        embeddings = torch.stack(
            [self.state_embeddings[m, state[m]] for m in range(modalities)], dim=1
        ).view(batch, modalities, channels, 1, 1)
        # 将特诊和身份编码相加
        conditioned = features + embeddings
        # available 变成权重
        weights = available.to(features.dtype).view(batch, modalities, 1, 1, 1)
        # 只把当前存在的模态取平均。
        pooled = (conditioned * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        # 最后再经过 ConvBlock
        return self.refine(pooled)

# 缺失模态生成
class MissingModalityGenerator(nn.Module):

    def __init__(self, modalities: int, channels: int) -> None:
        super().__init__()
        self.modalities = modalities
        # 创建缺失模态标记；* 0.02把随机数缩小，方便训练
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
        batch, modalities, channels, height, width = features.shape
        # 将可用性矩阵转换成权重
        # 把available转换成和features一样的数据类型
        # .view改变形状
        weights = available.to(features.dtype).view(batch, modalities, 1, 1, 1)
        # 计算已有模态的平均特征
        # 首先屏蔽缺失模态；然后对模态求和；sum(1)表示沿着第一个维度求和
        # clamp_min(1.0)如果数值小于1，就强制设置为1
        observed = (features * weights).sum(1) / weights.sum(1).clamp_min(1.0)
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
            mask = available[:, index].view(batch, 1, 1, 1)
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


class CrossModalAttentionFusion(nn.Module):
    """Pixel-wise attention across the three sensor tokens plus dynamic gating."""

    def __init__(self, modalities: int, channels: int, heads: int = 4) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("Feature channels must be divisible by attention heads")
        self.modalities = modalities
        self.channels = channels
        self.state_embeddings = nn.Parameter(torch.randn(modalities, 2, channels) * 0.02)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.gate = nn.Sequential(nn.Linear(channels, channels // 2), nn.GELU(), nn.Linear(channels // 2, 1))
        self.output = ConvBlock(channels, channels)

    def forward(
        self, features: torch.Tensor, original_availability: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, modalities, channels, height, width = features.shape
        state = original_availability.long().transpose(0, 1)
        embeddings = torch.stack(
            [self.state_embeddings[m, state[m]] for m in range(modalities)], dim=1
        ).view(batch, modalities, channels, 1, 1)
        conditioned = features + embeddings
        tokens = conditioned.permute(0, 3, 4, 1, 2).reshape(-1, modalities, channels)
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm(tokens + attended)
        weights = torch.softmax(self.gate(tokens).squeeze(-1), dim=1)
        fused = (tokens * weights.unsqueeze(-1)).sum(dim=1)
        fused = fused.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        weight_map = weights.reshape(batch, height, width, modalities).permute(0, 3, 1, 2)
        return self.output(fused), weight_map


class SegmentationDecoder(nn.Module):
    def __init__(self, channels: int, output_size: int) -> None:
        super().__init__()
        self.output_size = output_size
        widths = [channels, max(channels // 2, 32), max(channels // 4, 32), 32]
        self.stages = nn.ModuleList(
            [ConvBlock(widths[index], widths[index + 1]) for index in range(len(widths) - 1)]
        )
        self.segmentation_head = nn.Conv2d(widths[-1], 1, 1)
        self.boundary_head = nn.Conv2d(widths[-1], 1, 1)

    def forward(self, feature: torch.Tensor) -> dict[str, torch.Tensor]:
        x = feature
        pyramid = [x]
        for stage in self.stages:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = stage(x)
            pyramid.append(x)
        if x.shape[-2:] != (self.output_size, self.output_size):
            x = F.interpolate(
                x, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False
            )
        return {
            "logits": self.segmentation_head(x),
            "boundary_logits": self.boundary_head(x),
            "pyramid": pyramid,
        }

