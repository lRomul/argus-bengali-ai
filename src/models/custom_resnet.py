from functools import partial

import torch
import torch.nn as nn
from torchvision.models.resnet import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152
)
from timm import create_model

from src.models.classifiers import Classifier


ENCODERS = {
    "resnet18": (resnet18, 512),
    "resnet34": (resnet34, 512),
    "resnet50": (resnet50, 2048),
    "resnet101": (resnet101, 2048),
    "resnet152": (resnet152, 2048),
    "gluon_resnet34_v1b": (partial(create_model, 'gluon_resnet34_v1b'), 512),
    "gluon_resnet50_v1d": (partial(create_model, 'gluon_resnet50_v1d'), 2048),
}


class CustomResnet(nn.Module):
    def __init__(self,
                 encoder="resnet34",
                 pretrained=True):
        super().__init__()

        resnet, num_bottleneck_filters = ENCODERS[encoder]
        resnet = resnet(pretrained=pretrained)

        self.first_layers = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = Classifier(num_bottleneck_filters, None)

    def forward(self, x):
        x = self.first_layers(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
