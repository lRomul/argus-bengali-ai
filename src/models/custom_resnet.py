from functools import partial

import torch.nn as nn
from torchvision.models.resnet import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152
)
from timm import create_model
from src.models.resnext_wsl import resnext101_32x8d_wsl

from src.models.classifiers import Classifier, ConvClassifier
from src.models.cbam import CBAM


ENCODERS = {
    "resnet18": (resnet18, 512),
    "resnet34": (resnet34, 512),
    "resnet50": (resnet50, 2048),
    "resnet101": (resnet101, 2048),
    "resnet152": (resnet152, 2048),
    "gluon_resnet34_v1b": (partial(create_model, 'gluon_resnet34_v1b'), 512),
    "gluon_resnet50_v1d": (partial(create_model, 'gluon_resnet50_v1d'), 2048),
    "gluon_seresnext50_32x4d": (partial(create_model, 'gluon_seresnext50_32x4d'), 2048),
    "resnext101_32x8d_wsl": (resnext101_32x8d_wsl, 2048),
    "seresnext26t_32x4d": (partial(create_model, 'seresnext26t_32x4d'), 2048),
    "resnet50_jsd": (partial(create_model, 'resnet50'), 2048),
}


class CustomResnet(nn.Module):
    def __init__(self,
                 encoder="resnet34",
                 pretrained=True,
                 classifier=None,
                 cbam=None,
                 aux=None):
        super().__init__()
        resnet, num_bottleneck_filters = ENCODERS[encoder]
        resnet = resnet(pretrained=pretrained)

        if hasattr(resnet, 'relu'):
            act = resnet.relu
        elif hasattr(resnet, 'act1'):
            act = resnet.act1
        else:
            raise Exception

        self.first_layers = nn.Sequential(
            resnet.conv1, resnet.bn1, act, resnet.maxpool
        )

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        if cbam is not None:
            for layer in [self.layer2, self.layer3, self.layer4]:
                block = layer[-1]
                block.se = CBAM(
                    block.bn3.num_features,
                    reduction_ratio=cbam['reduction_ratio'],
                    pool_types=cbam['pool_types'],
                    no_spatial=cbam['no_spatial']
                )

        self.aux2, self.aux3 = None, None
        if aux is not None:
            self.aux2 = self._make_classifier(self.layer2[-1].bn3.num_features, classifier)
            self.aux3 = self._make_classifier(self.layer3[-1].bn3.num_features, classifier)

        self.classifier = self._make_classifier(num_bottleneck_filters, classifier)

    def _make_classifier(self, num_filters, classifier):
        if classifier is None:
            classifier = 'fc', {'pooler': 'avgpool'}

        if classifier[0] == 'fc':
            classifier = Classifier(num_filters, None, **classifier[1])
        elif classifier[0] == 'conv':
            classifier = ConvClassifier(num_filters, None, **classifier[1])
        else:
            raise NotImplementedError

        return classifier

    def forward(self, x):
        x = self.first_layers(x)

        x = self.layer1(x)
        x = self.layer2(x)
        if self.aux2 is not None:
            aux2 = self.aux2(x)
        x = self.layer3(x)
        if self.aux3 is not None:
            aux3 = self.aux3(x)
        x = self.layer4(x)

        x = self.classifier(x)

        if self.aux2 is None:
            return x
        else:
            return x, aux3, aux2
