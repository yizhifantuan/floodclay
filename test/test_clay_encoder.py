from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", message=r".*Torch was not compiled with flash attention", category=UserWarning)

from floodclay.config import load_config
from floodclay.data import FloodPlanetDataset, scan_floodplanet
from floodclay.models.clay_encoder import MODEL_SIZES, SharedClayEncoder


def make_smoke_checkpoint(path: Path, patch_size: int) -> None:
    from claymodel.model import Encoder

    encoder = Encoder(
        mask_ratio=0.0,
        patch_size=patch_size,
        shuffle=False,
        **MODEL_SIZES["tiny"],
    )
    torch.save(
        {"state_dict": {f"model.encoder.{key}": value for key, value in encoder.state_dict().items()}},
        path,
    )


def run(args: argparse.Namespace, checkpoint: Path, model_size: str) -> None:
    config = load_config(args.config, clay_checkpoint=checkpoint)
    records = scan_floodplanet(config["data"]["root"])

    index = next((i for i, r in enumerate(records) if r.sample_id == args.sample_id), None)
    if index is None:
        raise ValueError(f"找不到样本 {args.sample_id!r}")

    dataset = FloodPlanetDataset(
        records,
        config["sensors"],
        label_size=64,
        augment=False,
        validate_geography=False,
    )
    sample = dataset[index]
    if not sample["availability"][0]:
        raise ValueError(f"样本 {args.sample_id} 缺少真实 S1 影像，无法展示 S1 特征")

    device = torch.device(args.device)
    model_config = dict(config["model"])
    model_config["clay_model_size"] = model_size
    encoder = SharedClayEncoder(model_config, config["sensors"]).to(device).eval()

    # 官方 Clay 的位置编码按正方形 patch 网格构造；缩小影像以便 CPU 也能完成测试。
    images = {
        modality: F.interpolate(
            image.unsqueeze(0), size=(args.image_size, args.image_size),
            mode="bilinear", align_corners=False,
        ).to(device)
        for modality, image in sample["images"].items()
    }
    time = sample["time"].unsqueeze(0).to(device)
    latlon = sample["latlon"].unsqueeze(0).to(device)
    with torch.inference_mode():
        features = encoder(images, time, latlon)

    patch_size = encoder.backbone.patch_size
    expected_shape = (1, encoder.output_dim, args.image_size // patch_size, args.image_size // patch_size)
    for modality in encoder.modalities:
        feature = features[modality]
        if tuple(feature.shape) != expected_shape or not bool(torch.isfinite(feature).all()):
            raise RuntimeError(f"{modality} 特征异常：形状={tuple(feature.shape)}")
        print(f"{modality.upper()} 特征形状：{list(feature.shape)}")

    label = "S1 第1通道特征图（左上角最多 4×4）"
    if args.smoke:
        label += "，随机权重"
    print(f"{label}：")
    for row in features["s1"][0, 0, :4, :4].cpu().tolist():
        print(" ".join(f"{value:8.4f}" for value in row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs" / "default.json")
    parser.add_argument("--sample-id", default="BGD_40_119")
    parser.add_argument("--checkpoint", type=Path, help="预训练 Clay checkpoint 路径")
    parser.add_argument("--model-size", choices=MODEL_SIZES, help="权重对应的 Clay 模型尺寸")
    parser.add_argument("--image-size", type=int, default=32, help="测试时缩放到的正方形边长")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true", help="使用官方 Encoder 的临时随机 tiny 权重")
    args = parser.parse_args()

    config = load_config(args.config, clay_checkpoint=args.checkpoint)
    patch_size = config["model"]["patch_size"]
    if args.image_size < patch_size or args.image_size % patch_size:
        parser.error(f"--image-size 必须是 {patch_size} 的正整数倍")
    if args.smoke:
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "clay-tiny-random.ckpt"
            make_smoke_checkpoint(checkpoint, patch_size)
            run(args, checkpoint, "tiny")
    else:
        checkpoint = Path(config["model"]["clay_checkpoint"])
        run(args, checkpoint, args.model_size or config["model"]["clay_model_size"])


if __name__ == "__main__":
    main()
