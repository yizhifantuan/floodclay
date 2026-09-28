from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    valid = valid.to(values.dtype)
    return (values * valid).sum() / valid.sum().clamp_min(1.0)


def segmentation_loss(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    bce = _masked_mean(F.binary_cross_entropy_with_logits(logits, target, reduction="none"), valid)
    probabilities = torch.sigmoid(logits) * valid
    target = target * valid
    intersection = (probabilities * target).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return bce + dice


def boundary_target(target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    dilated = F.max_pool2d(target, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-target, kernel_size=3, stride=1, padding=1)
    return ((dilated - eroded) > 0).to(target.dtype) * valid


class MultiLevelDistillationLoss(nn.Module):
    """Supervision plus shared, boundary, prediction, and reconstruction KD."""

    def __init__(self, weights: dict[str, float], temperature: float = 2.0) -> None:
        super().__init__()
        self.weights = weights
        self.temperature = temperature

    def forward(
        self,
        outputs: dict[str, Any],
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        teacher = outputs["teacher"]
        student = outputs["student"]
        teacher_seg = segmentation_loss(teacher["logits"], target, valid)
        student_seg = segmentation_loss(student["logits"], target, valid)

        target_boundary = boundary_target(target, valid)
        teacher_boundary_sup = _masked_mean(
            F.binary_cross_entropy_with_logits(
                teacher["boundary_logits"], target_boundary, reduction="none"
            ),
            valid,
        )
        student_boundary_sup = _masked_mean(
            F.binary_cross_entropy_with_logits(
                student["boundary_logits"], target_boundary, reduction="none"
            ),
            valid,
        )

        shared_kd = F.smooth_l1_loss(student["shared"], teacher["shared"].detach())
        boundary_kd = _masked_mean(
            F.smooth_l1_loss(
                torch.sigmoid(student["boundary_logits"]),
                torch.sigmoid(teacher["boundary_logits"].detach()),
                reduction="none",
            ),
            valid,
        )

        temperature = self.temperature
        soft_teacher = torch.sigmoid(teacher["logits"].detach() / temperature)
        prediction_kd = _masked_mean(
            F.binary_cross_entropy_with_logits(
                student["logits"] / temperature,
                soft_teacher,
                reduction="none",
            ),
            valid,
        ) * (temperature**2)

        # Only synthetically hidden modalities have a real target feature. This term
        # directly trains the generator instead of relying on segmentation gradients alone.
        hidden = (~outputs["student_mask"]) & outputs["actual_availability"]
        per_modality = (student["generated"] - outputs["aligned"].detach()).abs().mean(
            dim=(2, 3, 4)
        )
        reconstruction = (per_modality * hidden.to(per_modality.dtype)).sum() / hidden.sum().clamp_min(1)

        components = {
            "teacher_seg": teacher_seg,
            "student_seg": student_seg,
            "boundary_supervision": 0.5 * (teacher_boundary_sup + student_boundary_sup),
            "shared_kd": shared_kd,
            "boundary_kd": boundary_kd,
            "prediction_kd": prediction_kd,
            "reconstruction": reconstruction,
        }
        total = sum(self.weights.get(name, 0.0) * loss for name, loss in components.items())
        components["total"] = total
        return total, components

