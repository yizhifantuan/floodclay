# 比较预测与真实标签，计算洪水 IoU、两类 mIoU、F1、Precision、Recall、Accuracy。
from __future__ import annotations

import torch


class BinarySegmentationMetrics:
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> None:
        # 先选出有效标签，四项统计复用同一有效区域。
        valid = valid.bool()
        prediction = torch.sigmoid(logits[valid]) >= self.threshold
        truth = target[valid] >= 0.5
        self.tp += (prediction & truth).sum().item()
        self.fp += (prediction & ~truth).sum().item()
        self.fn += (~prediction & truth).sum().item()
        self.tn += (~prediction & ~truth).sum().item()

    def compute(self) -> dict[str, float]:
        eps = 1e-9
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        background_iou = self.tn / (self.tn + self.fp + self.fn + eps)
        # 累计所有有效像素后，对洪水与非洪水两类取平均；并集为空的类计为 0。
        miou = (iou + background_iou) / 2
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn + eps)
        return {
            "iou": iou,
            "miou": miou,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
        }

