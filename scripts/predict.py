from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# 找到项目根目录，并让 Python 能从这个目录导入项目代码
# __file__指当前文件的路径
# .resolve()：得到绝对路径
# .parents[1]：取上两级目录。
PROJECT = Path(__file__).resolve().parents[1]
# sys.path Python 导入模块时搜索的目录列表
# str(PROJECT)：把路径对象转换成字符串
# insert(0, ...) 把项目根目录放到搜索列表的最前面，优先搜索它
sys.path.insert(0, str(PROJECT))

from floodclay.config import load_config, load_evaluation_config  # noqa: E402
from floodclay.data import FloodPlanetDataset, scan_floodplanet  # noqa: E402
from floodclay.engine import load_checkpoint, move_to_device, read_checkpoint  # noqa: E402
from floodclay.engine.experiments import latest_best_checkpoint  # noqa: E402
from floodclay.image.prediction import experiment_image_dir, write_geotiff  # noqa: E402
from floodclay.models import build_model  # noqa: E402

def main() -> None:
    # 运行设置：修改样本编号后直接运行，图像保存到本次实验的 image 目录。
    config_path = PROJECT / "configs" / "default.json"
    checkpoint = None  # None 选择输出根目录下最新一个已有 best.pt 的实验。
    sample_id = "BGD_40_119"
    threshold = 0.5
    # 加载配置
    config = load_config(config_path)
    if checkpoint is None:
        checkpoint = latest_best_checkpoint(config["training"]["output_dir"])
    output_dir = experiment_image_dir(checkpoint) / "predictions"
    # 预测恢复训练配置，避免默认配置变化后误用 PS 尺寸或 CNN 开关。
    payload = read_checkpoint(checkpoint, map_location="cpu")
    config = load_evaluation_config(payload)
    # 找到需要预测的样本
    matching = [
        # 扫描数据目录，生成样本记录。每条记录包含样本编号、各模态路径、标签路径和模态可用情况
        r for r in scan_floodplanet(config["data"]["root"], require_labels=False)
        # 只保留编号与用户指定编号相同的记录
        if r.sample_id == sample_id
    ]
    if not matching:
        raise KeyError(f"Unknown sample id: {sample_id}")
    # 取匹配记录，后面使用其参考影像、样本编号和实际模态状态。
    record = matching[0]
    # 创建预测数据集
    dataset = FloodPlanetDataset(
        matching,
        config["sensors"],
        label_size=config["data"]["label_size"],
        # 是否提供地理位置编码
        use_geo_metadata=config["data"]["use_geo_metadata"],
        alphaearth_config=config.get("alphaearth"),
    )
    # 创建数据加载器，每个批次包含一个样本
    # iter(...)获得可以逐批取数据的迭代器
    # next(...)取出第一个批次
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    load_checkpoint(checkpoint, model, map_location="cpu", payload=payload)
    del payload
    model.eval()
    batch = move_to_device(batch, device)
    with torch.no_grad():
        output = model.predict(batch)
        probability = torch.sigmoid(output["logits"])

    probability = probability[0, 0]
    binary = (probability >= threshold).to(torch.uint8)
    probability_path = output_dir / f"{record.sample_id}_probability.tif"
    mask_path = output_dir / f"{record.sample_id}_mask.tif"
    write_geotiff(record.reference_path, probability_path, probability.cpu().numpy(), "float32")
    write_geotiff(record.reference_path, mask_path, binary.cpu().numpy(), "uint8")
    print(f"availability [S1,S2,PS] = {list(record.availability)}")
    print(f"reference: {record.reference_modality.upper()} ({record.reference_path})")
    print(f"probability: {probability_path}")
    print(f"mask: {mask_path}")


if __name__ == "__main__":
    main()
