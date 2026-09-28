from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from floodclay.config import load_config  # noqa: E402
from floodclay.data import FloodPlanetDataset, scan_floodplanet, split_by_event  # noqa: E402
from floodclay.engine import move_to_device, save_checkpoint, seed_everything  # noqa: E402
from floodclay.losses import MultiLevelDistillationLoss  # noqa: E402
from floodclay.metrics import BinarySegmentationMetrics  # noqa: E402
from floodclay.models import build_model  # noqa: E402


def resolve_paths(config: dict[str, Any], data_root: str | None, clay_checkpoint: str | None) -> None:
    if data_root:
        config["data"]["root"] = data_root
    if clay_checkpoint:
        config["model"]["clay_checkpoint"] = clay_checkpoint
    elif not Path(config["model"]["clay_checkpoint"]).is_absolute():
        config["model"]["clay_checkpoint"] = str(PROJECT / config["model"]["clay_checkpoint"])
    if not Path(config["training"]["output_dir"]).is_absolute():
        config["training"]["output_dir"] = str(PROJECT / config["training"]["output_dir"])


def make_loaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader, dict[str, object]]:
    data = config["data"]
    records = scan_floodplanet(data["root"])
    splits = split_by_event(
        records, data["val_fraction"], data["test_fraction"], data["split_seed"]
    )
    common = {
        "sensor_config": config["sensors"],
        "label_size": data["label_size"],
        "require_complete": True,
        "use_geo_metadata": data.get("use_geo_metadata", True),
    }
    train_set = FloodPlanetDataset(splits["train"], augment=True, **common)
    val_set = FloodPlanetDataset(splits["val"], augment=False, **common)
    loader_args = {
        "batch_size": config["training"]["batch_size"],
        "num_workers": data["num_workers"],
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_set, shuffle=True, drop_last=False, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)
    manifest = {
        name: [record.to_dict() for record in items]
        for name, items in splits.items()
    }
    return train_loader, val_loader, manifest


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: MultiLevelDistillationLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    gradient_clip: float,
    amp: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    metrics = BinarySegmentationMetrics()
    for step, batch in enumerate(loader):
        batch = move_to_device(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
            student_mask = None
        else:
            # Deterministic validation cycles through each one-missing scenario.
            student_mask = batch["availability"].clone()
            student_mask[:, step % 3] = False

        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp and device.type == "cuda"
            else nullcontext()
        )
        grad_context = nullcontext() if training else torch.no_grad()
        with grad_context, autocast:
            outputs = model(batch, student_mask=student_mask)
            loss, parts = criterion(outputs, batch["target"], batch["valid"])

        if training:
            assert optimizer is not None
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()

        for name, value in parts.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().item())
        metrics.update(outputs["student"]["logits"], batch["target"], batch["valid"])

    losses = {name: value / len(loader) for name, value in totals.items()}
    return losses, metrics.compute()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the complete-teacher/missing-student model")
    parser.add_argument("--config", default=str(PROJECT / "configs" / "default.json"))
    parser.add_argument("--data-root")
    parser.add_argument("--clay-checkpoint")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    resolve_paths(config, args.data_root, args.clay_checkpoint)
    if args.epochs:
        config["training"]["epochs"] = args.epochs

    seed_everything(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, manifest = make_loaders(config)
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    model = build_model(config).to(device)
    criterion = MultiLevelDistillationLoss(
        config["loss"]["weights"], config["loss"]["temperature"]
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"]
    )
    use_amp = bool(config["training"]["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
    print(
        f"device={device} train_complete={len(train_loader.dataset)} "
        f"val_complete={len(val_loader.dataset)} encoder=clay"
    )

    history_path = output_dir / "history.jsonl"
    best_iou = -1.0
    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            config["training"]["gradient_clip"],
            use_amp,
        )
        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            None,
            None,
            config["training"]["gradient_clip"],
            use_amp,
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
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
