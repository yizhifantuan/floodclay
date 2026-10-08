from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from floodclay.config import load_config  # noqa: E402
from floodclay.data import FloodPlanetDataset, scan_floodplanet, split_by_event  # noqa: E402
from floodclay.engine import run_epoch, save_checkpoint, seed_everything  # noqa: E402
from floodclay.losses import MultiLevelDistillationLoss  # noqa: E402
from floodclay.models import build_model  # noqa: E402


def make_loaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader, dict[str, object]]:
    data = config["data"]
    splits = split_by_event(
        scan_floodplanet(data["root"]),
        data["val_fraction"], data["test_fraction"], data["split_seed"],
    )
    dataset_args = {
        "sensor_config": config["sensors"],
        "label_size": data["label_size"],
        "require_complete": True,
        "use_geo_metadata": data["use_geo_metadata"],
    }
    train_set = FloodPlanetDataset(splits["train"], augment=True, **dataset_args)
    val_set = FloodPlanetDataset(splits["val"], **dataset_args)
    loader_args = {
        "batch_size": config["training"]["batch_size"],
        "num_workers": data["num_workers"],
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, **loader_args)
    manifest = {
        name: [record.to_dict() for record in items]
        for name, items in splits.items()
    }
    return train_loader, val_loader, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the complete-teacher/missing-student model")
    parser.add_argument("--config", default=str(PROJECT / "configs" / "default.json"))
    parser.add_argument("--data-root")
    parser.add_argument("--clay-checkpoint")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()

    config = load_config(
        args.config, data_root=args.data_root, clay_checkpoint=args.clay_checkpoint
    )
    training = config["training"]
    if args.epochs is not None:
        training["epochs"] = args.epochs

    seed_everything(training["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(training["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, manifest = make_loaders(config)
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    model = build_model(config).to(device)
    criterion = MultiLevelDistillationLoss(
        config["loss"]["weights"], config["loss"]["temperature"]
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training["epochs"])
    scaler = torch.amp.GradScaler(
        device.type, enabled=training["amp"] and device.type == "cuda"
    )
    print(
        f"device={device} train_complete={len(train_loader.dataset)} "
        f"val_complete={len(val_loader.dataset)} encoder=clay"
    )

    best_iou = -1.0
    history_path = output_dir / "history.jsonl"
    for epoch in range(1, training["epochs"] + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        train_loss, train_metrics = run_epoch(
            model, train_loader, criterion, device, scaler,
            optimizer=optimizer, gradient_clip=training["gradient_clip"],
        )
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, device, scaler)
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": learning_rate,
            "train_loss": train_loss,
            "train_metrics": train_metrics,
            "val_loss": val_loss,
            "val_metrics": val_metrics,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"epoch={epoch:03d} train_loss={train_loss['total']:.4f} "
            f"val_loss={val_loss['total']:.4f} val_iou={val_metrics['iou']:.4f}"
        )
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, config, val_metrics)
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, config, val_metrics)
    print(f"Training complete. Best validation IoU={best_iou:.4f}")


if __name__ == "__main__":
    main()
