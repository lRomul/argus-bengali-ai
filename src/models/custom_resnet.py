import torch
import torch.nn as nn
from torchvision.models.resnet import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152
)

from src.models.dropblock import DropBlock2D, LinearScheduler
from src.models.classifiers import Classifier


ENCODERS = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
}


class CustomResnet(nn.Module):
    def __init__(self,
                 encoder="resnet34",
                 pretrained=True,
                 dropblock_prob=0.,
                 dropblock_size=5,
                 dropblock_nr_steps=5000):
        super().__init__()

        if encoder in ["resnet18", "resnet34"]:
            self.filters = [64, 128, 256, 512]
        else:
            self.filters = [256, 512, 1024, 2048]

        resnet = ENCODERS[encoder](pretrained=pretrained)

        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=dropblock_prob,
                        block_size=dropblock_size),
            start_value=0.,
            stop_value=dropblock_prob,
            nr_steps=dropblock_nr_steps
        )

        self.first_layers = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = Classifier(self.filters[-1], None)

    def forward(self, x):
        self.dropblock.step()
        x = self.first_layers(x)
        x = self.layer1(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.dropblock(x)
        x = self.layer2(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
