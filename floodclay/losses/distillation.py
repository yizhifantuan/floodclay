"""教师—学生分割监督、共享/边界/预测蒸馏及缺失特征重建。"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..masks import masked_mean
from .segmentation import boundary_target, segmentation_loss


# 真正的知识蒸馏 Loss
class MultiLevelDistillationLoss(nn.Module):
    # 监督 Loss + shared 特征蒸馏 + boundary 边界蒸馏 + prediction 预测蒸馏 + reconstruction 重建
    def __init__(self, weights: dict[str, float]) -> None:
        super().__init__()
        # 保存每个 Loss 的权重
        self.weights = weights

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
        teacher_boundary_sup = masked_mean(
            F.binary_cross_entropy_with_logits(
                teacher["boundary_logits"], target_boundary, reduction="none"
            ),
            valid,
        )
        # 比较由 student 生成的洪水边界和真实的洪水边界
        student_boundary_sup = masked_mean(
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
        # 只对有效像素的损失求平均
        boundary_kd = masked_mean(
            F.smooth_l1_loss(
                # 学生预测每个像素属于边界的概率
                torch.sigmoid(student["boundary_logits"]),
                torch.sigmoid(teacher["boundary_logits"].detach()),
                reduction="none",
            ),
            valid,
        )
        # 直接使用 Teacher 的预测概率作为软标签。
        soft_teacher = torch.sigmoid(teacher["logits"].detach())
        # Student 学 Teacher 的最终预测
        prediction_kd = masked_mean(
            F.binary_cross_entropy_with_logits(
                student["logits"],
                soft_teacher,
                reduction="none",
            ),
            valid,
        )

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
        reconstruction = masked_mean(per_modality, hidden)
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
