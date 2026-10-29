"""学生训练时采样完整输入，以及保留至少一路观测的缺失组合。"""

from __future__ import annotations

import random

import torch
from torch import nn


# 随机让模态“消失”
class ModalityMaskSampler(nn.Module):
    # max_drop：最多删除的模态；drop_probability多少概率进行一次数据丢失
    # 三路齐全且 max_drop=2 时：完整输入概率为 1-p，每种缺失组合概率为 p/6。
    # 默认 p=6/7，使完整输入和六种缺失组合各以 1/7 的概率参与学生训练。
    def __init__(self, max_drop: int = 2, drop_probability: float = 6 / 7) -> None:
        super().__init__()
        self.max_drop = max_drop
        self.drop_probability = drop_probability

    # 下面这个函数不需要计算梯度
    @torch.no_grad()
    def forward(self, availability: torch.Tensor) -> torch.Tensor:
        # 获取原来的模态缺失编码；.clone()复制一个新的张量，避免修改复制后的数据时影响原始数据
        result = availability.bool().clone()
        # 逐个进行处理
        for batch_index in range(result.shape[0]):
            # 首先找到当前样本哪些模态存在
            # torch.where() 会找出所有值为 True 的位置
            # 这里的 [0] 是取出返回元组中的第一个元素
            present = torch.where(result[batch_index])[0]
            # 如果只有一个模态就不删除，后面是在随机决定：这次到底删不删
            # random.random() 直接生成 0 到 1 之间的 Python 浮点数
            # self.drop_probability：设定的隐藏概率
            if present.numel() <= 1 or random.random() >= self.drop_probability:
                continue
            # 决定最多删除几个
            maximum = min(self.max_drop, present.numel() - 1)
            # 生成 Python 整数，范围为 1 到 maximum（包含两端）
            count = random.randint(1, maximum)
            # 随机决定删除谁；present.numel()计算present里面有多少数字
            # torch.randperm(3) 会把 0、1、2 随机打乱顺序
            # [:count]取前count个元素
            # 得到一个张量记住屏蔽哪些模态
            dropped = present[torch.randperm(present.numel(), device=result.device)[:count]]
            # 模拟缺失
            result[batch_index, dropped] = False
        # result = torch.tensor([
        #     [True, True, True]
        # ])
        return result
