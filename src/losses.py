import torch
from torch import nn
import torch.nn.functional as F


class SoftTargetCrossEntropy(nn.Module):
    def __init__(self, ohem_rate=1.0):
        super(SoftTargetCrossEntropy, self).__init__()
        self.ohem_rate = ohem_rate

    def forward(self, x, target, training=False):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)

        if training and self.ohem_rate < 1.0:
            _, idx = torch.sort(loss, descending=True)
            keep_num = int(x.size(0) * self.ohem_rate)
            if keep_num < x.size(0):
                keep_idx = idx[:keep_num]
                loss = loss[keep_idx]

        return loss.mean()


class BengaliAiCrossEntropy(nn.Module):
    def __init__(self,
                 grapheme_weight=2.0,
                 vowel_weight=1.0,
                 consonant_weight=1.0,
                 ohem_rate=1.0):
        super(BengaliAiCrossEntropy, self).__init__()

        self.grapheme_weight = grapheme_weight
        self.vowel_weight = vowel_weight
        self.consonant_weight = consonant_weight
        self.ohem_rate = ohem_rate

        loss = SoftTargetCrossEntropy(ohem_rate=ohem_rate)
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
