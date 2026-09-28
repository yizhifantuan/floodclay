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
from floodclay.engine import load_checkpoint, move_to_device  # noqa: E402
from floodclay.metrics import BinarySegmentationMetrics  # noqa: E402
from floodclay.models import build_model  # noqa: E402


SCENARIOS = {
    "complete": (1, 1, 1),
    "missing_s1": (0, 1, 1),
    "missing_s2": (1, 0, 1),
    "missing_ps": (1, 1, 0),
    "only_s1": (1, 0, 0),
    "only_s2": (0, 1, 0),
    "only_ps": (0, 0, 1),
}


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    pattern: tuple[int, int, int] | None,
    amp: bool,
) -> dict[str, float]:
    metrics = BinarySegmentationMetrics()
    model.eval()
    for batch in loader:
        batch = move_to_device(batch, device)
        selected = None
        if pattern is not None:
            selected = torch.tensor(pattern, device=device, dtype=torch.bool).view(1, 3)
            selected = selected.expand(batch["availability"].shape[0], -1)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp and device.type == "cuda"
            else nullcontext()
        )
        with context:
            output = model.predict(batch, availability=selected)
        metrics.update(output["logits"], batch["target"], batch["valid"])
    return metrics.compute()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate every missing-modality scenario")
    parser.add_argument("--config", default=str(PROJECT / "configs" / "default.json"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--clay-checkpoint")
    parser.add_argument("--data-root")
    parser.add_argument("--output")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_root:
        config["data"]["root"] = args.data_root
    if args.clay_checkpoint:
        config["model"]["clay_checkpoint"] = args.clay_checkpoint
    elif not Path(config["model"]["clay_checkpoint"]).is_absolute():
        config["model"]["clay_checkpoint"] = str(PROJECT / config["model"]["clay_checkpoint"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = scan_floodplanet(config["data"]["root"])
    splits = split_by_event(
        records,
        config["data"]["val_fraction"],
        config["data"]["test_fraction"],
        config["data"]["split_seed"],
    )
    common: dict[str, Any] = {
        "sensor_config": config["sensors"],
        "label_size": config["data"]["label_size"],
        "augment": False,
        "use_geo_metadata": config["data"].get("use_geo_metadata", True),
    }
    complete_set = FloodPlanetDataset(splits["test"], require_complete=True, **common)
    natural_set = FloodPlanetDataset(splits["test"], require_complete=False, **common)
    loader_args = {
        "batch_size": config["training"]["batch_size"],
        "num_workers": config["data"]["num_workers"],
        "shuffle": False,
    }
    complete_loader = DataLoader(complete_set, **loader_args)
    natural_loader = DataLoader(natural_set, **loader_args)

    model = build_model(config).to(device)
    payload = load_checkpoint(args.checkpoint, model, device)
    amp = bool(config["training"].get("amp", True))
    results = {
        name: evaluate_loader(model, complete_loader, device, pattern, amp)
        for name, pattern in SCENARIOS.items()
    }
    results["natural_missingness"] = evaluate_loader(model, natural_loader, device, None, amp)
    report = {
        "checkpoint_epoch": payload.get("epoch"),
        "complete_test_samples": len(complete_set),
        "natural_test_samples": len(natural_set),
        "metrics": results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    output = Path(args.output) if args.output else Path(args.checkpoint).with_name("evaluation.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Evaluation saved to {output}")


if __name__ == "__main__":
    main()
