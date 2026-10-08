"""Lightweight smoke test for the real three-modality teacher branch.

Run from the project root:
    python test/test_teacher_smoke.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from floodclay.config import load_config  # noqa: E402
from floodclay.data import FloodPlanetDataset, SampleRecord, scan_floodplanet  # noqa: E402
from floodclay.engine import move_to_device, seed_everything  # noqa: E402
from floodclay.losses import boundary_target, segmentation_loss  # noqa: E402
from floodclay.metrics import BinarySegmentationMetrics  # noqa: E402
from floodclay.models import build_model  # noqa: E402


def make_batch(
    record: SampleRecord, config: dict[str, Any], image_size: int, device: torch.device
) -> dict[str, Any]:
    dataset = FloodPlanetDataset(
        [record],
        config["sensors"],
        label_size=config["data"]["label_size"],
        require_complete=True,
        use_geo_metadata=config["data"]["use_geo_metadata"],
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
    if batch["availability"].shape != (1, 3) or not bool(batch["availability"].all()):
        raise RuntimeError(f"Sample {record.sample_id} is not complete")
    batch["images"] = {
        name: F.interpolate(image, size=(image_size, image_size), mode="bilinear", align_corners=False)
        for name, image in batch["images"].items()
    }
    return move_to_device(batch, device)


def teacher_loss(
    output: dict[str, torch.Tensor], batch: dict[str, Any], boundary_weight: float
) -> torch.Tensor:
    target, valid = batch["target"], batch["valid"]
    seg_loss = segmentation_loss(output["logits"], target, valid)
    boundary_loss = F.binary_cross_entropy_with_logits(
        output["boundary_logits"], boundary_target(target, valid), reduction="none"
    )
    boundary_loss = (boundary_loss * valid).sum() / valid.sum().clamp_min(1)
    loss = seg_loss + boundary_weight * boundary_loss
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Teacher loss contains NaN or Inf")
    return loss


def check_output(output: dict[str, torch.Tensor], output_size: int) -> None:
    expected_shape = (1, 1, output_size, output_size)
    if tuple(output["logits"].shape) != expected_shape:
        raise RuntimeError(f"Unexpected teacher output shape: {tuple(output['logits'].shape)}")
    if not all(bool(torch.isfinite(output[name]).all()) for name in ("logits", "boundary_logits")):
        raise RuntimeError("Teacher output contains NaN or Inf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs" / "default.json")
    parser.add_argument("--data-root", type=Path, help="Override the data directory")
    parser.add_argument("--checkpoint", type=Path, help="Override the Clay checkpoint")
    parser.add_argument("--sample-id", help="Use a specific complete sample")
    parser.add_argument("--val-sample-id", help="Use a different complete sample for validation")
    parser.add_argument("--image-size", type=int, default=32, help="Small square input size; default: 32")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(
        args.config, data_root=args.data_root, clay_checkpoint=args.checkpoint
    )
    patch_size = config["model"]["patch_size"]
    if args.image_size < patch_size or args.image_size % patch_size:
        parser.error(f"--image-size must be a positive multiple of {patch_size}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --device cpu")
    if config["data"]["label_size"] != config["model"]["output_size"]:
        parser.error("data.label_size must equal model.output_size")

    records = scan_floodplanet(config["data"]["root"])
    complete = [r for r in records if r.is_complete]
    train_record = next((r for r in complete if args.sample_id is None or r.sample_id == args.sample_id), None)
    if train_record is None:
        parser.error("No complete S1/S2/PS sample matched --sample-id")
    if args.val_sample_id is None:
        val_record = next((r for r in complete if r.event != train_record.event), None)
    else:
        val_record = next((r for r in complete if r.sample_id == args.val_sample_id), None)
    if val_record is None or val_record.sample_id == train_record.sample_id:
        parser.error("Validation requires a second, different complete S1/S2/PS sample")

    config["model"]["freeze_clay"] = True  # Keep the smoke test light.
    seed_everything(config["training"]["seed"])

    device = torch.device(args.device)
    train_batch = make_batch(train_record, config, args.image_size, device)
    val_batch = make_batch(val_record, config, args.image_size, device)
    model = build_model(config).to(device).train()
    teacher_modules = (
        model.encoder,
        model.aligner,
        model.shared_representation,
        model.teacher_fusion,
        model.teacher_decoder,
    )
    parameters = [p for module in teacher_modules for p in module.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["training"]["learning_rate"])
    boundary_weight = config["loss"]["weights"]["boundary_supervision"]

    optimizer.zero_grad(set_to_none=True)
    output = model._teacher_branch(model.encode(train_batch))
    check_output(output, config["model"]["output_size"])
    loss = teacher_loss(output, train_batch, boundary_weight)
    loss.backward()
    if not any(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.teacher_decoder.parameters()):
        raise RuntimeError("Teacher decoder did not receive a finite gradient")
    if any(p.grad is not None for p in model.student_decoder.parameters()):
        raise RuntimeError("Student branch unexpectedly received gradients")
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        val_output = model._teacher_branch(model.encode(val_batch))
        check_output(val_output, config["model"]["output_size"])
        val_loss = teacher_loss(val_output, val_batch, boundary_weight)
        metrics = BinarySegmentationMetrics()
        metrics.update(val_output["logits"], val_batch["target"], val_batch["valid"])
        val_iou = metrics.compute()["iou"]

    print(
        f"PASS train={train_record.sample_id} val={val_record.sample_id} "
        f"modalities=S1/S2/PS device={device} input={args.image_size}x{args.image_size} "
        f"output={tuple(output['logits'].shape)} train_loss={loss.item():.4f} "
        f"val_loss={val_loss.item():.4f} val_iou={val_iou:.4f} "
        "backward=ok optimizer_step=ok"
    )


if __name__ == "__main__":
    main()
