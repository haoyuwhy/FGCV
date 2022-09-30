import math
from turtle import forward
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class CrossentropyLoss(nn.Module):
    def __init__(self, loss_weight=1.0) -> None:
        super(CrossentropyLoss,self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.loss_weight = loss_weight

    def forward(self, x, label):
        return self.loss_weight * self.loss(x, label)
