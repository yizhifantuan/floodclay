"""Run one real teacher/student training step and one validation step.

From the project root: python test/train_smoke.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from floodclay.config import load_config  # noqa: E402
from floodclay.data import FloodPlanetDataset, scan_floodplanet, split_by_event  # noqa: E402
from floodclay.engine import run_epoch, seed_everything  # noqa: E402
from floodclay.losses import MultiLevelDistillationLoss  # noqa: E402
from floodclay.models import build_model  # noqa: E402


class SmallInputDataset(Dataset):
    """Keep the real labels and metadata, but shrink the three image inputs."""

    def __init__(self, dataset: FloodPlanetDataset, image_size: int) -> None:
        self.dataset = dataset
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        sample = self.dataset[index]
        sample["images"] = {
            name: F.interpolate(
                image.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            for name, image in sample["images"].items()
        }
        return sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs" / "default.json")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--clay-checkpoint", type=Path)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(
        args.config, data_root=args.data_root, clay_checkpoint=args.clay_checkpoint
    )
    patch_size = config["model"]["patch_size"]
    if args.image_size < patch_size or args.image_size % patch_size:
        parser.error(f"--image-size must be a positive multiple of {patch_size}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")

    config["model"]["freeze_clay"] = True
    seed_everything(config["training"]["seed"])

    splits = split_by_event(
        scan_floodplanet(config["data"]["root"]),
        config["data"]["val_fraction"],
        config["data"]["test_fraction"],
        config["data"]["split_seed"],
    )
    selected = {}
    for name in ("train", "val"):
        selected[name] = next((record for record in splits[name] if record.is_complete), None)
        if selected[name] is None:
            parser.error(f"No complete S1/S2/PS sample in the {name} split")

    loaders = {}
    for name in ("train", "val"):
        dataset = FloodPlanetDataset(
            [selected[name]],
            config["sensors"],
            label_size=config["data"]["label_size"],
            augment=name == "train",
            require_complete=True,
            use_geo_metadata=config["data"]["use_geo_metadata"],
        )
        loaders[name] = DataLoader(SmallInputDataset(dataset, args.image_size), batch_size=1, num_workers=0)

    device = torch.device(args.device)
    model = build_model(config).to(device)
    criterion = MultiLevelDistillationLoss(config["loss"]["weights"], config["loss"]["temperature"])
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scaler = torch.amp.GradScaler(
        device.type, enabled=config["training"]["amp"] and device.type == "cuda"
    )
    train_loss, _ = run_epoch(
        model, loaders["train"], criterion, device, scaler,
        optimizer=optimizer, gradient_clip=config["training"]["gradient_clip"],
    )
    val_loss, val_metrics = run_epoch(
        model, loaders["val"], criterion, device, scaler,
    )
    if not all(math.isfinite(value) for value in (train_loss["total"], val_loss["total"], val_metrics["iou"])):
        raise RuntimeError("Smoke test produced NaN or Inf")
    print(
        f"PASS train={selected['train'].sample_id} val={selected['val'].sample_id} "
        f"device={device} input={args.image_size}x{args.image_size} "
        f"train_loss={train_loss['total']:.4f} val_loss={val_loss['total']:.4f} "
        f"val_iou={val_metrics['iou']:.4f} backward=ok optimizer_step=ok"
    )


if __name__ == "__main__":
    main()
