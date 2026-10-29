from __future__ import annotations

import torch


# 根据每个样本的模态可用情况，生成一个可以用于屏蔽缺失模态特征的掩码
def modality_mask(
    availability: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    # 与特征使用相同设备，并转换为布尔类型
    mask = availability.to(device=values.device, dtype=torch.bool)
    # 在末尾补充维度，直到与特征的维度数量相同
    # ndim指维度数量
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    return mask


def zero_invalid(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    # 根据有效性掩码 valid，保留有效位置的数据，将无效位置的数据设置为 0
    # where(条件，条件为true，条件为false)
    return torch.where(valid.to(device=values.device, dtype=torch.bool), values, 0.0)

# 只给允许参与计算的模态分配权重，并将权重归一化
def masked_softmax(scores: torch.Tensor, valid: torch.Tensor, dim: int) -> torch.Tensor:
    # 首先将有效性编码转换为布尔类型
    valid = valid.to(device=scores.device, dtype=torch.bool)
    # 屏蔽无效位置的评分；~valid取反原来[False, True, False]转换为[True, False, True]
    # masked_fill(mask, value)把掩码为true的位置替换为指定数值
    scores = scores.float().masked_fill(~valid, torch.finfo(torch.float32).min)
    # 先通过 Softmax 将各模态的评分转换为权重，然后利用vaild将无效模态置为 0
    weights = scores.softmax(dim=dim) * valid
    # weights.sum 沿指定维度计算权重之和；clamp_min() 表示将数值的最小值限制为指定值
    # 同一个位置的32个通道共用一个权重，这里主要是让权重和为1
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-12)


def masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """有效元素的全局均值；全部无效时返回零，支持广播掩码。"""
    valid = valid.to(device=values.device, dtype=torch.bool).expand_as(values)
    accumulation_dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
    count = valid.sum(dtype=accumulation_dtype).clamp_min(1.0)
    # valid 已统一设备和类型，直接选择有效值，避免再次转换掩码。
    return torch.where(valid, values, 0.0).sum(dtype=accumulation_dtype) / count
