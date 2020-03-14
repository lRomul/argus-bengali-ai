import torch
from torch import nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss


class SmoothingOhemCrossEntropy(nn.Module):
    def __init__(self, smooth_factor=0.0, ohem_rate=1.0):
        super().__init__()
        self.smooth_factor = float(smooth_factor)
        self.ohem_rate = ohem_rate
        self.ce = LabelSmoothingCrossEntropy(smoothing=self.smooth_factor)

    def forward(self, label_input, label_target, training=False):
        if isinstance(label_target, (tuple, list)):
            y1, y2, lam = label_target
            loss = self.ce(label_input, y1) * lam + self.ce(label_input, y2) * (1 - lam)

            if training and self.ohem_rate < 1.0:
                _, idx = torch.sort(loss, descending=True)
                keep_num = int(label_input.size(0) * self.ohem_rate)
                if keep_num < label_input.size(0):
                    keep_idx = idx[:keep_num]
                    loss = loss[keep_idx]
                    return loss.sum() / keep_num
        else:
            loss = self.ce(label_input, label_target)

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

        self.loss = SmoothingOhemCrossEntropy(smooth_factor=smooth_factor,
                                              ohem_rate=ohem_rate)
        self.grapheme_ce = self.loss
        self.vowel_ce = self.loss
        self.consonant_ce = self.loss

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
