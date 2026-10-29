"""有效像素上的分割损失与形态学边界目标。"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..masks import masked_mean


# 分割 Loss
# 比较模型预测的洪水区域和真实洪水区域
#
def segmentation_loss(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    # 1024² 像素的半精度求和可能超过 65504；在 float32 中计算 BCE/Dice。
    logits, target = logits.float(), target.float()
    # binary_cross_entropy_with_logits就是二分类 BCE Loss
    # BCE 就是在检查：每一个像素预测得对不对
    #  reduction="none", # 暂时不求平均
    bce = masked_mean(F.binary_cross_entropy_with_logits(logits, target, reduction="none"), valid)
    # valid里面包含着无效和有效区域
    # 计算概率
    # 把预测分数转换为洪水概率，再把无效像素的概率清零
    probabilities = torch.sigmoid(logits) * valid
    # 真实标签也去掉无效区域
    target = target * valid
    # 计算交集 intersection
    # 对每一张图片，把所有通道和所有像素加起来sum(dim=(1, 2, 3)
    intersection = (probabilities * target).sum(dim=(1, 2, 3))
    # 预测洪水区域大小 + 真实洪水区域大小
    denominator = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    # 计算dice
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    # 最终分割 Loss
    return bce + dice


# 获得洪水边界；从真实洪水 mask 中自动提取洪水边界
def boundary_target(target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    # 膨胀
    dilated = F.max_pool2d(target, kernel_size=3, stride=1, padding=1)
    # 腐蚀
    eroded = -F.max_pool2d(-target, kernel_size=3, stride=1, padding=1)
    # 若一个 3×3 区域同时含有 0 和 1，膨胀结果为 1、腐蚀结果为 0，当前位置就被标为边界。
    # 得到的是边界附近的一圈区域，并非精确到单像素的轮廓线
    # 得到洪水边界
    return ((dilated - eroded) > 0).to(target.dtype) * valid
