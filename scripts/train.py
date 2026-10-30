from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[1]

from floodclay.config import load_config
from floodclay.data import FloodPlanetDataset, scan_floodplanet, split_by_event
from floodclay.engine import build_lr_scheduler, run_epoch, save_checkpoint, seed_everything
from floodclay.engine.experiments import create_experiment_dir
from floodclay.losses import MultiLevelDistillationLoss
from floodclay.models import build_model

# 传入config字典；输出两个DataLoader一个字典
def make_loaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader, dict[str, object]]:
    # 获取
    data = config["data"]
    splits = split_by_event(
        scan_floodplanet(data["root"]),
        data["train_events"], data["val_events"], data["test_events"],
    )
    # 公共数据
    dataset_args = {
        "sensor_config": config["sensors"],
        "label_size": data["label_size"],
        "require_complete": True,
        "use_geo_metadata": data["use_geo_metadata"],
        "alphaearth_config": config.get("alphaearth"),
    }
    # augment 开启数据增强
    # 分别创建训练集和验证集；**dataset_args把字典解包，作为参数传进去
    train_set = FloodPlanetDataset(splits["train"], augment=True, **dataset_args)
    val_set = FloodPlanetDataset(splits["val"], **dataset_args)
    # 准备加载器的公共数据集
    loader_args = {
        "batch_size": config["training"]["batch_size"],
        "num_workers": data["num_workers"],
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, **loader_args)
    # 创建数据清单
    manifest = {
        name: [record.to_dict() for record in items]
        for name, items in splits.items()
    }
    return train_loader, val_loader, manifest


def main(config_path: str | Path = PROJECT / "configs" / "default.json") -> Path:
    config = load_config(config_path)
    # 从config中获取training字典
    training = config["training"]
    epochs = training["epochs"]
    patience = training.get("early_stopping_patience", 20)
    if not isinstance(patience, int) or isinstance(patience, bool) or patience < 1:
        raise ValueError("training.early_stopping_patience 必须为正整数")
    # 设置随机种子
    seed_everything(training["seed"])
    # 定义设备对象
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 整次实验仅在这里创建一次目录；全部 epoch、验证和最终测试共用该目录。
    # 默认保存在 runs/result/run_*，图像另存到 runs/image/run_*。
    output_dir = create_experiment_dir(training["output_dir"])
    training["output_dir"] = str(output_dir)
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"本次实验输出目录：{output_dir}", flush=True)
    train_loader, val_loader, manifest = make_loaders(config)
    # 将数据清单转换为json格式
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 首先创建一个模型
    model = build_model(config).to(device)
    clay_trainable_parameters = sum(
        parameter.numel() for name, parameter in model.named_parameters()
        if name.startswith("encoder.backbone.") and parameter.requires_grad
    )
    print(f"Clay 主干可训练参数量：{clay_trainable_parameters:,}", flush=True)
    # 定义损失函数
    criterion = MultiLevelDistillationLoss(config["loss"]["weights"])
    optimizer = torch.optim.AdamW(
        # 遍历模型参数，只保留需要训练得参数
        # model.parameters()       # 遍历模型的全部参数
        # parameter.requires_grad  # 这个参数是否需要计算梯度
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=training["learning_rate"],
        # 设置权重衰减
        weight_decay=training["weight_decay"],
    )
    # 重启周期按轮数配置，内部换算为成功优化步骤数，在 run_epoch 内逐步更新。
    scheduler = build_lr_scheduler(optimizer, training, len(train_loader))
    # 摄制混合精读训练
    scaler = torch.amp.GradScaler(
        device.type, enabled=training["amp"] and device.type == "cuda"
    )
    print(
        f"device={device} train_complete={len(train_loader.dataset)} "
        f"val_complete={len(val_loader.dataset)} encoder=clay alphaearth={model.alphaearth_enabled} "
        f"ps_encoder_size={model.encoder.ps_encoder_size} "
        f"ps_detail={model.encoder.ps_detail is not None}"
    )
    print(
        f"scheduler=CosineAnnealingWarmRestarts 每次成功优化后更新 "
        f"initial_lr={training['learning_rate']:.2e} T_0={scheduler.T_0} steps "
        f"T_mult={scheduler.T_mult} eta_min={scheduler.eta_min:.2e}",
        flush=True,
    )

    best_iou = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history_path = output_dir / "history.jsonl"
    for epoch in range(1, epochs + 1):
        # 读取学习率
        learning_rate = optimizer.param_groups[0]["lr"]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        train_started_at = perf_counter()
        train_loss, train_metrics = run_epoch(
            model, train_loader, criterion, device, scaler,
            optimizer=optimizer, gradient_clip=training["gradient_clip"],
            scheduler=scheduler,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = perf_counter() - train_started_at
        train_gpu_memory = None
        if device.type == "cuda":
            train_gpu_memory = {
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
            }
            print(
                f"epoch={epoch:03d} 训练峰值显存：张量占用 "
                f"{train_gpu_memory['peak_allocated_gib']:.2f} GiB，"
                f"PyTorch 保留（含缓存）{train_gpu_memory['peak_reserved_gib']:.2f} GiB",
                flush=True,
            )
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, device, scaler)
        row = {
            "epoch": epoch,
            "lr": learning_rate,
            "lr_end": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_metrics": train_metrics,
            "train_seconds": train_seconds,
            "train_gpu_memory": train_gpu_memory,
            "val_loss": val_loss,
            "val_metrics": val_metrics,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"epoch={epoch:03d} train_loss={train_loss['total']:.4f} "
            f"val_loss={val_loss['total']:.4f} val_iou={val_metrics['iou']:.4f} "
            f"val_miou={val_metrics['miou']:.4f} "
            f"lr={learning_rate:.2e}->{optimizer.param_groups[0]['lr']:.2e} "
            f"train_time={train_seconds:.2f}s ({train_seconds / 60:.2f}min)"
        )
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, config, val_metrics, split_manifest=manifest, scheduler=scheduler)
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, config, val_metrics, split_manifest=manifest, scheduler=scheduler)
        else:
            epochs_without_improvement += 1
        print(f"验证 IoU 连续未提升：{epochs_without_improvement}/{patience} 轮", flush=True)
        if epochs_without_improvement >= patience:
            print(
                f"提前停止训练：验证 IoU 连续 {patience} 轮未超过历史最佳值；"
                f"最佳轮次={best_epoch}，最佳 IoU={best_iou:.4f}。",
                flush=True,
            )
            break
    print(f"Training complete. Best validation IoU={best_iou:.4f} at epoch={best_epoch}")
    print(f"最佳权重：{output_dir / 'best.pt'}", flush=True)
    return output_dir / "best.pt"


if __name__ == "__main__":
    main()
