from typing import Optional

from torch import nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    def __init__(self, ignore_index: Optional[int] = -100, reduction="mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, label_input, label_target):
        if self.ignore_index is not None:
            not_ignored_mask = (label_target != self.ignore_index).float()

        loss = F.binary_cross_entropy_with_logits(label_input, label_target, reduction="none")
        if self.ignore_index is not None:
            loss = loss * not_ignored_mask.float()

        if self.reduction == "mean":
            loss = loss.mean()

        if self.reduction == "sum":
            loss = loss.sum()

        return loss


class SoftBCELoss(nn.Module):
    def __init__(self, smooth_factor=0, ignore_index: Optional[int] = -100, reduction="mean"):
        super().__init__()
        self.smooth_factor = float(smooth_factor)
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, label_input, label_target):
        if self.ignore_index is not None:
            not_ignored_mask = (label_target != self.ignore_index).float()

        label_target = (1 - label_target) * self.smooth_factor + label_target * (1 - self.smooth_factor)

        loss = F.binary_cross_entropy_with_logits(label_input, label_target, reduction="none")

        if self.ignore_index is not None:
            loss = loss * not_ignored_mask.float()

        if self.reduction == "mean":
            loss = loss.mean()

        if self.reduction == "sum":
            loss = loss.sum()

        return loss


class BengaliAiCrossEntropy(nn.Module):
    def __init__(self,
                 grapheme_weight=2.0,
                 vowel_weight=1.0,
                 consonant_weight=1.0,
                 binary=True,
                 smooth_factor=0):
        super(BengaliAiCrossEntropy, self).__init__()

        self.grapheme_weight = grapheme_weight
        self.vowel_weight = vowel_weight
        self.consonant_weight = consonant_weight
        self.binary = binary
        self.smooth_factor = smooth_factor

        if self.smooth_factor:
            if binary:
                loss = partial(SoftBCELoss, smooth_factor=smooth_factor)
            else:
                raise NotImplementedError
        else:
            loss = nn.BCEWithLogitsLoss if binary else nn.CrossEntropyLoss

        self.grapheme_ce = loss()
        self.vowel_ce = loss()
        self.consonant_ce = loss()

    def __call__(self, pred, target):
        grapheme_pred, vowel_pred, consonant_pred = pred
        grapheme_target, vowel_target, consonant_target = target

        loss = 0
        if self.grapheme_weight:
            loss += self.grapheme_weight \
                    * self.grapheme_ce(grapheme_pred, grapheme_target)

        if self.vowel_weight:
            loss += self.vowel_weight \
                    * self.vowel_ce(vowel_pred, vowel_target)

        if self.consonant_weight:
            loss += self.consonant_weight \
                    * self.consonant_ce(consonant_pred, consonant_target)

        return loss
