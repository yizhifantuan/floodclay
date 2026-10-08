from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

# 只计算有效像素的平均 Loss
def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    # 转换数据格式
    valid = valid.to(values.dtype)
    # 计算有效像素之后的平均loss值
    return (values * valid).sum() / valid.sum().clamp_min(1.0)

# 分割 Loss
# 比较模型预测的洪水区域和真实洪水区域
#
def segmentation_loss(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    # binary_cross_entropy_with_logits就是二分类 BCE Loss
    # BCE 就是在检查：每一个像素预测得对不对
    bce = _masked_mean(F.binary_cross_entropy_with_logits(logits, target, reduction="none"), valid)
    # valid里面包含着无效和有效区域
    # 计算概率
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

# 真正的知识蒸馏 Loss
class MultiLevelDistillationLoss(nn.Module):
    # 监督 Loss + shared 特征蒸馏 + boundary 边界蒸馏 + prediction 预测蒸馏 + reconstruction 重建
    def __init__(self, weights: dict[str, float], temperature: float = 2.0) -> None:
        super().__init__()
        # 保存每个 Loss 的权重
        self.weights = weights
        # 给prediction KD使用
        self.temperature = temperature

    # outputs模型输出；target真实洪水标签；valid有效像素
    def forward(
        self,
        outputs: dict[str, Any],
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # 拿出 Teacher 和 Student
        teacher = outputs["teacher"]
        student = outputs["student"]
        # Teacher 分割 Loss
        teacher_seg = segmentation_loss(teacher["logits"], target, valid)
        # Student 分割 Loss
        student_seg = segmentation_loss(student["logits"], target, valid)
        # 生成真实的洪水边界
        target_boundary = boundary_target(target, valid)
        # 比较由 teacher 生成的洪水边界和真实的洪水边界
        teacher_boundary_sup = _masked_mean(
            F.binary_cross_entropy_with_logits(
                teacher["boundary_logits"], target_boundary, reduction="none"
            ),
            valid,
        )
        # 比较由 student 生成的洪水边界和真实的洪水边界
        student_boundary_sup = _masked_mean(
            F.binary_cross_entropy_with_logits(
                student["boundary_logits"], target_boundary, reduction="none"
            ),
            valid,
        )

        # 特征知识蒸馏；smooth_l1_loss计算两个特征之间的差异
        # 计算 student 和 teacher 之间共享特征的损失
        # .detach()在这里作为“答案”，不要通过这个 Loss 去修改 Teacher
        # 仅更新 student
        shared_kd = F.smooth_l1_loss(student["shared"], teacher["shared"].detach())
        # 边界知识蒸馏
        # Teacher预测的洪水边界
        #           ↓
        #        教 Student
        #           ↓
        # Student预测的洪水边界
        boundary_kd = _masked_mean(
            F.smooth_l1_loss(
                torch.sigmoid(student["boundary_logits"]),
                torch.sigmoid(teacher["boundary_logits"].detach()),
                reduction="none",
            ),
            valid,
        )
        # 获取 temperature
        temperature = self.temperature
        # 生成 Teacher 的软标签
        # 除以 temperature 会让 Teacher 的概率变得更加“柔和”
        soft_teacher = torch.sigmoid(teacher["logits"].detach() / temperature)
        # Student 学 Teacher 的最终预测
        prediction_kd = _masked_mean(
            F.binary_cross_entropy_with_logits(
                student["logits"] / temperature,
                soft_teacher,
                reduction="none",
            ),
            valid,
        # temperature**2 这是知识蒸馏中常见的温度缩放补偿，
        # 用来避免 temperature 改变以后梯度尺度变化过大
        ) * (temperature**2)

        # 找出人为隐藏的模态
        hidden = (~outputs["student_mask"]) & outputs["actual_availability"]
        # 计算生成特征与真实特征的误差
        # student["generated"]为缺失模态生成出来的特征；
        # outputs["aligned"]是原始真实模态经过编码、对齐后的真实特征
        # .abs()取绝对值；.mean()取平均值
        per_modality = (student["generated"] - outputs["aligned"].detach()).abs().mean(
            dim=(2, 3, 4)
        )
        # 只计算人为隐藏模态
        reconstruction = (per_modality * hidden.to(per_modality.dtype)).sum() / hidden.sum().clamp_min(1)
        # 把所有 Loss 放进字典
        # | 名称 | 简单理解 | 主要训练谁 |
        # | teacher_seg | Teacher 洪水分割得对不对 | Teacher |
        # | student_seg | Student 洪水分割得对不对 | Student |
        # | boundary_supervision | 洪水边界找得对不对 | Teacher + Student |
        # | shared_kd | Student 中间特征像不像 Teacher | Student |
        # | boundary_kd | Student 边界预测像不像 Teacher | Student |
        # | prediction_kd | Student 最终预测像不像 Teacher | Student |
        # | reconstruction | Student 生成的缺失模态特征像不像真实特征 | 缺失模态生成器 |
        components = {
            "teacher_seg": teacher_seg,
            "student_seg": student_seg,
            "boundary_supervision": 0.5 * (teacher_boundary_sup + student_boundary_sup),
            "shared_kd": shared_kd,
            "boundary_kd": boundary_kd,
            "prediction_kd": prediction_kd,
            "reconstruction": reconstruction,
        }
        # 计算最终 Loss
        total = sum(self.weights[name] * loss for name, loss in components.items())
        # 保存 Total
        components["total"] = total
        return total, components
