from turtle import forward
import torch
from torch import nn as nn
from torch.nn import functional as F
import torchvision
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class resnet(nn.Module):
    def __init__(self, name="resnet50", pretrained=True, lables=1000):
        super(resnet, self).__init__()

        if name == "resnet34" or name == "resnet50" or name == "resnet101" or name == "resnet152":
            self.net = torchvision.models.resnet.__dict__[
                name](pretrained=pretrained)
        else:
            self.net = torch.hub.load(
                'XingangPan/IBN-Net', name, pretrained=pretrained)

        # freeze layers
        self.net.conv1.weight.requires_grad_(False)
        self.net.bn1.weight.requires_grad_(False)
        self.net.bn1.bias.requires_grad_(False)
        in_channel = self.net.fc.in_features
        # net.fc = nn.Linear(in_channel, 5)
        self.net.fc = nn.Linear(in_channel, lables)

    def forward(self, x):
        return self.net(x)
