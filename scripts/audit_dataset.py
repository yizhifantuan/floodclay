from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from floodclay.config import load_config  # noqa: E402
from floodclay.data.index import scan_floodplanet, split_by_event, summarize  # noqa: E402


def mask_signature(path: Path) -> tuple[object, ...]:
    with rasterio.open(path) as src:
        array = src.read()
    values = np.unique(array)
    valid_binary = array.shape[0] == 1 and set(values.tolist()).issubset({0, 1, 2})
    return tuple(array.shape), str(array.dtype), int(values.min()), int(values.max()), valid_binary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit FloodPlanet pairing and label integrity")
    parser.add_argument("--config", default=str(PROJECT / "configs" / "default.json"))
    parser.add_argument("--output", default=str(PROJECT / "runs" / "dataset_audit.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(config["data"]["root"])
    records = scan_floodplanet(root)
    splits = split_by_event(
        records,
        config["data"]["val_fraction"],
        config["data"]["test_fraction"],
        config["data"]["split_seed"],
    )

    report: dict[str, object] = {
        "root": str(root),
        "all": summarize(records),
        "splits": {name: summarize(items) for name, items in splits.items()},
    }
    mask_report: dict[str, object] = {}
    for modality in ("S1", "S2"):
        directory = root / modality / "masks"
        signatures: Counter[tuple[object, ...]] = Counter()
        invalid: list[str] = []
        for path in sorted(directory.glob("*.tif")):
            signature = mask_signature(path)
            signatures[signature] += 1
            if not signature[-1]:
                invalid.append(path.name)
        mask_report[modality] = {
            "patterns": {str(key): value for key, value in signatures.items()},
            "invalid_files": invalid,
        }
    report["masks"] = mask_report

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nAudit saved to {output}")


if __name__ == "__main__":
    main()

