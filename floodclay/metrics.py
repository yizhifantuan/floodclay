# 把模型预测的洪水区域和真实洪水区域进行比较，然后计算 IoU、F1、Precision、Recall、Accuracy。
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
        prediction = torch.sigmoid(logits) >= self.threshold
        truth = target >= 0.5
        mask = valid.bool()
        self.tp += int((prediction & truth & mask).sum().item())
        self.fp += int((prediction & ~truth & mask).sum().item())
        self.fn += int((~prediction & truth & mask).sum().item())
        self.tn += int((~prediction & ~truth & mask).sum().item())

    def compute(self) -> dict[str, float]:
        eps = 1e-9
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn + eps)
        return {
            "iou": iou,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
        }

