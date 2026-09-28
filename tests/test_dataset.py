import json
import unittest
import rasterio
from pathlib import Path

from floodclay.config import load_config
from floodclay.data import FloodPlanetDataset, scan_floodplanet


PROJECT = Path(__file__).resolve().parents[1]


class DatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(PROJECT / "configs" / "default.json")
        root = Path(cls.config["data"]["root"])
        if not root.is_dir():
            raise unittest.SkipTest(f"FloodPlanet is not mounted at {root}")

    def test_inventory_matches_supplied_copy(self) -> None:
        records = scan_floodplanet(self.config["data"]["root"])
        self.assertEqual(len(records), 366)
        self.assertEqual(sum(r.s1 is not None for r in records), 362)
        self.assertEqual(sum(r.s2 is not None for r in records), 298)
        self.assertEqual(sum(r.is_complete for r in records), 294)

    def test_real_missing_modality_is_zero_filled_and_marked(self) -> None:
        records = scan_floodplanet(self.config["data"]["root"])
        record = next(r for r in records if r.s2 is None)
        dataset = FloodPlanetDataset(
            [record],
            self.config["sensors"],
            label_size=32,
        )
        item = dataset[0]
        self.assertEqual(item["availability"].tolist(), [True, False, True])
        self.assertEqual(float(item["images"]["s2"].abs().sum()), 0.0)
        self.assertEqual(tuple(item["target"].shape), (1, 32, 32))
        for modality in ("s1", "ps"):
            with rasterio.open(getattr(record, modality)) as src:
                self.assertEqual(tuple(item["images"][modality].shape),
                                 (src.count, src.height, src.width))


if __name__ == "__main__":
    unittest.main()
