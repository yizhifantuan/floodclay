"""分割监督、边界目标和多层蒸馏损失的公共接口。"""

from .distillation import MultiLevelDistillationLoss
from .segmentation import boundary_target, segmentation_loss

__all__ = ["MultiLevelDistillationLoss", "boundary_target", "segmentation_loss"]
