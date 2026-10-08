# ① 固定随机种子 → seed_everything()
# ② 把数据移动到 GPU → move_to_device()
# ③ 保存训练模型 → save_checkpoint()
# ④ 读取训练模型 → load_checkpoint()
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .losses import MultiLevelDistillationLoss
from .metrics import BinarySegmentationMetrics


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: MultiLevelDistillationLoss,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip: float = 1.0,
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
        if optimizer is not None:
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

        if optimizer is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()

        batch_size = batch["target"].shape[0]
        samples += batch_size
        for name, value in parts.items():
            totals[name] += value.detach().item() * batch_size
        metrics.update(outputs["student"]["logits"], batch["target"], batch["valid"])

    losses = {name: value / samples for name, value in totals.items()}
    return losses, metrics.compute()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "metrics": metrics,
        },
        path,
    )

# 加载权重
def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    return payload
