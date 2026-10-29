"""直接运行 main()：训练和逐轮验证完成后，加载本次最优权重测试。"""

from __future__ import annotations

import gc
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import torch  # noqa: E402

from floodclay.config import load_config  # noqa: E402
from scripts.train import main as train_main  # noqa: E402
from scripts.evaluate import main as evaluate_main  # noqa: E402


def main() -> None:
    # 运行设置：训练轮数、数据路径和 AlphaEarth 开关均从配置文件读取。
    config_path = PROJECT / "configs" / "default.json"
    config = load_config(config_path)
    epochs = config["training"]["epochs"]
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("training.epochs 必须为正整数")

    print(f"[1/2] 开始训练和验证，最多 {epochs} 轮，验证 IoU 持续未提升时提前停止。", flush=True)
    checkpoint = train_main(config_path)

    # 训练函数返回后清理模型与优化器，再开始评估，避免同时保留两份模型。
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not checkpoint.is_file():
        raise RuntimeError(f"训练未生成最优权重，无法测试：{checkpoint}")

    # 评估入口从检查点恢复配置和数据划分，确保测试对应本次训练。
    print(f"[2/2] 加载最优权重进行测试：{checkpoint}", flush=True)
    evaluate_main(checkpoint)
    print(f"训练、验证和测试完成。测试报告：{checkpoint.with_name('evaluation.json')}", flush=True)


if __name__ == "__main__":
    main()
