"""教师—学生模型共用的训练与验证循环。"""

from __future__ import annotations

from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from ..losses import MultiLevelDistillationLoss
from ..metrics import BinarySegmentationMetrics
from .device import move_to_device


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: MultiLevelDistillationLoss,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip: float = 1.0,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """有优化器时训练，否则验证；禁用的 scaler 执行普通精度更新。"""
    training = optimizer is not None
    model.train(training)
    totals = defaultdict(float)
    metrics = BinarySegmentationMetrics()
    samples = 0

    for step, batch in enumerate(loader):
        batch = move_to_device(batch, device)
        student_mask = None
        if training:
            optimizer.zero_grad(set_to_none=True)
        else:
            # 验证按批次轮流隐藏 S1、S2、PS，保持每轮验证可重复。
            student_mask = batch["availability"].clone()
            student_mask[:, step % 3] = False

        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()
        ):
            outputs = model(batch, student_mask=student_mask)
            loss, parts = criterion(outputs, batch["target"], batch["valid"])

        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            previous_scale = scaler.get_scale() if scheduler is not None and scaler.is_enabled() else None
            scaler.step(optimizer)
            scaler.update()
            # AMP 溢出时跳过参数更新并降低 scale，此时调度器也不推进。
            if scheduler is not None and (previous_scale is None or scaler.get_scale() >= previous_scale):
                scheduler.step()

        batch_size = batch["target"].shape[0]
        samples += batch_size
        for name, value in parts.items():
            totals[name] += value.item() * batch_size
        metrics.update(outputs["student"]["logits"], batch["target"], batch["valid"])

    losses = {name: value / samples for name, value in totals.items()}
    return losses, metrics.compute()
