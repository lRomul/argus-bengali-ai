import torch
from torch import nn
import torch.nn.functional as F


class SoftBCELoss(nn.Module):
    def __init__(self, smooth_factor=0.0, ohem_rate=1.0):
        super().__init__()
        self.smooth_factor = float(smooth_factor)
        self.ohem_rate = ohem_rate

    def forward(self, label_input, label_target, training=False):
        if self.smooth_factor > 0.0:
            label_target = (1 - label_target) * self.smooth_factor \
                           + label_target * (1 - self.smooth_factor)

        loss = F.binary_cross_entropy_with_logits(label_input, label_target, reduction="none")

        if training and self.ohem_rate < 1.0:
            loss = torch.mean(loss, dim=1)
            _, idx = torch.sort(loss, descending=True)
            keep_num = int(label_input.size(0) * self.ohem_rate)
            if keep_num < label_input.size(0):
                keep_idx = idx[:keep_num]
                loss = loss[keep_idx]
                return loss.sum() / keep_num

        return loss.mean()


class BengaliAiCrossEntropy(nn.Module):
    def __init__(self,
                 grapheme_weight=2.0,
                 vowel_weight=1.0,
                 consonant_weight=1.0,
                 smooth_factor=0,
                 ohem_rate=1.0):
        super(BengaliAiCrossEntropy, self).__init__()

        self.grapheme_weight = grapheme_weight
        self.vowel_weight = vowel_weight
        self.consonant_weight = consonant_weight
        self.smooth_factor = smooth_factor
        self.ohem_rate = ohem_rate

        loss = SoftBCELoss(smooth_factor=smooth_factor, ohem_rate=ohem_rate)
        self.grapheme_ce = loss
        self.vowel_ce = loss
        self.consonant_ce = loss

    def __call__(self, pred, target, training=False):
        grapheme_pred, vowel_pred, consonant_pred = pred
        grapheme_target, vowel_target, consonant_target = target

        loss = 0
        if self.grapheme_weight:
            loss += self.grapheme_weight \
                    * self.grapheme_ce(grapheme_pred, grapheme_target,
                                       training=training)

        if self.vowel_weight:
            loss += self.vowel_weight \
                    * self.vowel_ce(vowel_pred, vowel_target,
                                    training=training)

        if self.consonant_weight:
            loss += self.consonant_weight \
                    * self.consonant_ce(consonant_pred, consonant_target,
                                        training=training)

        return loss
