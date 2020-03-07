from functools import partial

import torch.nn as nn
from timm import create_model

from src.models.classifiers import Classifier, ConvClassifier


ENCODERS = {
    "tf_efficientnet_b1": (partial(create_model, 'tf_efficientnet_b1'), 1280),
    "efficientnet_b0": (partial(create_model, 'efficientnet_b0'), 1280),
    "tf_efficientnet_b3_ns": (partial(create_model, 'tf_efficientnet_b3_ns'), 1536),
}


class CustomEfficient(nn.Module):
    def __init__(self,
                 encoder="tf_efficientnet_b1",
                 pretrained=True,
                 classifier=None):
        super().__init__()
        if classifier is None:
            classifier = 'fc', {'pooler': 'none'}

        efficient, num_bottleneck_filters = ENCODERS[encoder]
        self.efficient = efficient(pretrained=pretrained)

        if classifier[0] == 'fc':
            self.efficient.classifier = Classifier(num_bottleneck_filters,
                                                   None, **classifier[1])
        elif classifier[0] == 'conv':
            self.efficient.classifier = ConvClassifier(num_bottleneck_filters,
                                                       None, **classifier[1])
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.efficient(x)
