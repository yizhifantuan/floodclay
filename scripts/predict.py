from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from floodclay.config import load_config  # noqa: E402
from floodclay.data import FloodPlanetDataset, scan_floodplanet  # noqa: E402
from floodclay.engine import load_checkpoint, move_to_device  # noqa: E402
from floodclay.models import build_model  # noqa: E402


def write_geotiff(reference: str, output: Path, array: np.ndarray, dtype: str) -> None:
    with rasterio.open(reference) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width
    # Prediction value 0 is valid background/probability, not no-data.
    profile.update(count=1, dtype=dtype, nodata=None)
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(array.reshape(height, width).astype(dtype), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict one chip using its real modality availability")
    parser.add_argument("--config", default=str(PROJECT / "configs" / "default.json"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--clay-checkpoint")
    parser.add_argument("--output-dir", default=str(PROJECT / "predictions"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_root:
        config["data"]["root"] = args.data_root
    if args.clay_checkpoint:
        config["model"]["clay_checkpoint"] = args.clay_checkpoint
    elif not Path(config["model"]["clay_checkpoint"]).is_absolute():
        config["model"]["clay_checkpoint"] = str(PROJECT / config["model"]["clay_checkpoint"])

    matching = [r for r in scan_floodplanet(config["data"]["root"]) if r.sample_id == args.sample_id]
    if not matching:
        raise KeyError(f"Unknown sample id: {args.sample_id}")
    record = matching[0]
    dataset = FloodPlanetDataset(
        matching,
        config["sensors"],
        label_size=config["data"]["label_size"],
        use_geo_metadata=config["data"].get("use_geo_metadata", True),
    )
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, model, device)
    model.eval()
    batch = move_to_device(batch, device)
    with torch.no_grad():
        output = model.predict(batch)
        probability = torch.sigmoid(output["logits"])

    with rasterio.open(record.ps) as reference:
        output_shape = (reference.height, reference.width)
    probability = F.interpolate(
        probability, size=output_shape, mode="bilinear", align_corners=False
    )[0, 0]
    binary = (probability >= args.threshold).to(torch.uint8)
    output_dir = Path(args.output_dir)
    probability_path = output_dir / f"{record.sample_id}_probability.tif"
    mask_path = output_dir / f"{record.sample_id}_mask.tif"
    write_geotiff(record.ps, probability_path, probability.cpu().numpy(), "float32")
    write_geotiff(record.ps, mask_path, binary.cpu().numpy(), "uint8")
    print(f"availability [S1,S2,PS] = {list(record.availability)}")
    print(f"probability: {probability_path}")
    print(f"mask: {mask_path}")


if __name__ == "__main__":
    main()
