import torch.nn as nn
import torch.nn.functional as F

import timm


class N01z3Model(nn.Module):
    def __init__(self, filename, drop_rate=0.0):
        super().__init__()

        if "ig_resnext101" in filename:
            model = timm.create_model("ig_resnext101_32x8d", pretrained=False)
        elif "tf_efficientnet_b5" in filename:
            model = timm.create_model("tf_efficientnet_b5_ns", pretrained=False)
        elif "tf_efficientnet_b3" in filename:
            model = timm.create_model("tf_efficientnet_b3_ns", pretrained=False)
        else:
            model = timm.create_model("gluon_resnet50_v1d", pretrained=False)

        self.model = model
        self.drop_rate = drop_rate
        # grapheme_root
        self.fc1 = self.build_new_linear(168)
        # vowel_diacritic
        self.fc2 = self.build_new_linear(11)
        # consonant_diacritic
        self.fc3 = self.build_new_linear(7)

    def build_new_linear(self, num_classes):
        old_fc = self.model.get_classifier()
        new_last_linear = nn.Linear(old_fc.in_features, num_classes)
        return new_last_linear

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.model.global_pool(x).flatten(1)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        return x1, x2, x3
