"""按成功优化步骤推进的余弦热重启学习率调度。"""

from __future__ import annotations

from typing import Any

import torch


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    training: dict[str, Any],
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.CosineAnnealingWarmRestarts:
    """把配置中的轮数换算为优化步骤数，调用方每次参数更新后 step。"""
    restart_epochs = training.get("scheduler_t0_epochs", 10)
    if isinstance(restart_epochs, bool) or not isinstance(restart_epochs, int) or restart_epochs < 1:
        raise ValueError("training.scheduler_t0_epochs 必须为正整数")
    if steps_per_epoch < 1:
        raise ValueError("训练集必须至少包含一个 batch，才能创建学习率调度器")
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=restart_epochs * steps_per_epoch,
        T_mult=training.get("scheduler_t_mult", 1),
        eta_min=training.get("min_learning_rate", 0.0),
    )
