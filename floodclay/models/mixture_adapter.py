# 三模态共享瓶颈、异构双专家和位置级软路由。
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..masks import masked_softmax, modality_mask, zero_invalid


class MixtureAdapter(nn.Module):
    # grid_size 决定 Adapter 内部统一使用的空间网格大小
    # gate_hidden_dim 门控网络隐藏层的特征维度
    # residual_scale Adapter 残差缩放系数的初始值
    def __init__(
        self, channels: int, bottleneck_dim: int, grid_size: int,
        modalities, gate_hidden_dim: int = 16,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.modalities = tuple(modalities)
        self.grid_size = grid_size
        # 进行层归一化
        # 创建一个字典，每个模态对应着一个归一化层
        self.norms = nn.ModuleDict({name: nn.LayerNorm(channels) for name in self.modalities})
        # 首先进行降维，降到adapter里面的bottleneck_dim
        self.down = nn.Linear(channels, bottleneck_dim, bias=False)
        # 将瓶颈通道数恢复为输入通道数
        self.up = nn.Linear(bottleneck_dim, channels, bias=False)
        # 创建一个局部特征提取专家
        self.local_expert = nn.Sequential(
            # 普通 3×3 卷积同时提取空间特征并融合通道信息，保持通道数和空间尺寸不变。
            nn.Conv2d(bottleneck_dim, bottleneck_dim, 3, padding=1),
            nn.GELU(),
        )
        # 创建一个跨通道特征变换专家学习不同特征通道之间的关系
        self.cross_expert = nn.Sequential(
            # 经过一层全连接层
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.GELU()
        )
        # 使用一个全连接层把一个bottleneck_dim维度的向量转换为一个数值评分
        self.source_scorer = nn.Linear(bottleneck_dim, 1, bias=False)
        # 门控路由网络 两层全连接层，得到两个专家的分数
        self.router = nn.Sequential(
            # 第一次负责组合和学习不同来源特征得到新的特征
            nn.Linear(3 * bottleneck_dim, gate_hidden_dim),
            nn.GELU(),
            # 第二次负责得到分数
            nn.Linear(gate_hidden_dim, 2),
        )
        # 创建一个可训练的参数gamma用于控制残差分支对最终特征的影响程度
        # 控制 Adapter 对原始特征的修改程度
        self.gamma = nn.Parameter(torch.tensor(float(residual_scale)))
        # 将升维线性层的权重初始化为 0，使初始残差为 0
        nn.init.zeros_(self.up.weight)

    def forward(
        self, features: dict[str, torch.Tensor], available: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        # available 只表示整路模态可用性；当前数据流未提供像素级观测掩码。
        reference = features[self.modalities[0]]
        available = available.to(device=reference.device, dtype=torch.bool)
        # 创建一个空列表
        projected = []
        size = (self.grid_size, self.grid_size)
        for index, name in enumerate(self.modalities):
            feature = features[name]
            # 当前模态可用，就保留它的特征；缺失，就把它的特征全部置为 0
            # available[:, index]取出每个模态的可用状态
            # modality_mask 将可用状态广播到整张特征图
            clean = zero_invalid(feature, modality_mask(available[:, index], feature))
            # 首先调整维度顺序[B,C,H,W]->[B,H,W,C]
            # 归一化和线性层都是对最后一维进行作用
            # 然后进行层归一化和通道降维，最后变换回[B,C,H,W]
            local = self.down(self.norms[name](clean.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
            # 使用双线性插值，将 local 特征图的高度和宽度统一调整为 grid_size × grid_size
            if local.shape[-2:] != size:
                local = F.interpolate(local, size, mode="bilinear", align_corners=False)
            # 保存降维并对齐到统一网格的特征
            projected.append(local)

        # 把三个模态拼接然后改变维度位置
        # [B,M,C,H,W]->[B,M,H,W,C] 因为下面要进行线性层变换
        z = torch.stack(projected, dim=1).permute(0, 1, 3, 4, 2)
        # 根据模态可用性生成掩码
        # masks [B, 3, 1, 1, 1]
        masks = modality_mask(available, z)
        # 得到三个模态的三个分数
        # z.shape  # [B, 3, 16, 16, 32]
        # scores [B, 3, 16, 16, 1]
        scores = self.source_scorer(z)
        # 创建空字典保存增强后的各模态特征和空列表保存各模态融合之后的特征
        enhanced, mixed_features = {}, []
        # 遍历每个模态并生成编码
        for index, name in enumerate(self.modalities):
            # 复制一份掩码
            # clone() 创建独立副本，保留原始可用状态
            # other_mask [B, 3, 1, 1, 1]
            other_mask = masks.clone()
            # 将当前正在处理的模态设置为False排除当前的模态，只保留其他可用的模态
            other_mask[:, index] = False
            # 通过masked_softmax获取每个模态的权重
            # scores：各模态在每个空间位置的评分，形状为 [B, 3, 16, 16, 1]
            # weights.shape  # [B, 3, 16, 16, 1]
            weights = masked_softmax(scores, other_mask, dim=1).to(z.dtype)
            # 得到权重之后对各模态进行融合得到融合后的权重
            #  complement [B, 16, 16, 32]
            complement = (weights * z).sum(dim=1)
            # 获取当前正在处理的模态
            own = z[:, index]
            # 首先将[B,H,W,C]->[B,C,H,W]然后经过local_expert局部专家因为接下来要进行卷积
            # 再从[B,C,H,W]->[B,H,W,C]
            local = self.local_expert(own.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            # 将另外的模态互补融合后放入到cross_expert专家中
            # Cross  # [B, 16, 16, 32]
            cross = self.cross_expert(complement)
            # 用当前的模态，其他模态提供的特征，两者差值的绝对值输入路由网络
            # 用于计算两个专家的得分
            logits = self.router(torch.cat((own, complement, (own - complement).abs()), dim=-1))
            # 只有一路观测时禁用跨模态专家，防止其偏置伪造互补信息。
            # any判断是否至少有一个 True
            # has_cross  # [B,    1, 1, 1]
            has_cross = other_mask.any(dim=1)
            # 生成两个专家的可用性掩码，局部专家始终可用，跨模态专家只有存在其他模态才可用
            # 沿最后一维拼接
            expert_mask = torch.cat((torch.ones_like(has_cross), has_cross), dim=-1)
            # 根据专家分数和可用性掩码，计算局部专家与跨模态专家的融合权重
            # routing [B, 16, 16, 2]
            routing = masked_softmax(logits, expert_mask, dim=-1).to(z.dtype)
            mixed = zero_invalid(routing[..., :1] * local + routing[..., 1:] * cross, masks[:, index])
            mixed_features.append(mixed)
            # 先通过线性层恢复通道数，再从[B,H,W,C]转换为[B,C,H,W]
            residual = self.up(mixed).permute(0, 3, 1, 2)
            # 使用双线性插值，将空间尺寸调整到与当前模态原始特征一致
            size = features[name].shape[-2:]
            if residual.shape[-2:] != size:
                residual = F.interpolate(residual, size, mode="bilinear", align_corners=False)
            # mixed 已清零，且 up 无偏置；整路缺失的 residual 仍为零，无需再屏蔽。
            # 残差链接
            enhanced[name] = features[name] + self.gamma * residual
        # 统计有效模态的数量，保证最小值为 1
        count = masks.sum(dim=1).to(z.dtype).clamp_min(1)
        # 将 S1、S2、PlanetScope 三个模态经过专家融合后的特征进行堆叠、求和，
        # 再除以每个空间位置的有效模态数量，得到多模态共享特征 shared
        # shared [B, 16, 16, 32]
        shared = torch.stack(mixed_features, dim=1).sum(dim=1) / count
        # 返回增强和共享特征
        return enhanced, shared.permute(0, 3, 1, 2)
