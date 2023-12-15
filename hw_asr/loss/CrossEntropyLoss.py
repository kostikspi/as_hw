import torch
from torch import Tensor
import torch.nn as nn


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
        self.loss = nn.BCEWithLogitsLoss(weight=torch.tensor(weights))

    def forward(self, logits, label, **batch) -> Tensor:
        loss = self.loss(logits, label.float())
        return loss

