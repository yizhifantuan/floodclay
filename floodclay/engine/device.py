# 随机种子与批次设备移动。
from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

def seed_everything(seed: int) -> None:
    # 随机选择、列表打乱、随机整数
    random.seed(seed)
    # NumPy 随机数组、随机采样
    np.random.seed(seed)
    # PyTorch 模型权重初始化、Dropout 和随机张量，同时设置 CPU/CUDA 等设备。
    torch.manual_seed(seed)
# 把张量移动到gpu上
def move_to_device(value: Any, device: torch.device) -> Any:
    # isinstance() 用于判断变量是否属于指定类型
    # 判断是否为张量
    if isinstance(value, torch.Tensor):
        return value.to(device)
    # 判断是否为字典
    if isinstance(value, dict):
        # value.items()获取键和值
        # 把获取到的键和值都放到GPU上
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value
