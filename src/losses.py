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


class JsdCrossEntropy(nn.Module):
    """ Jensen-Shannon Divergence + Cross-Entropy Loss
    Based on impl here: https://github.com/google-research/augmix/blob/master/imagenet.py
    From paper: 'AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty -
    https://arxiv.org/abs/1912.02781
    Hacked together by Ross Wightman
    """
    def __init__(self,
                 num_splits=3,
                 smooth_factor=0.0,
                 ohem_rate=1.0):
        super().__init__()
        self.num_splits = num_splits
        self.iter = 0
        self.loss = SoftBCELoss(smooth_factor=smooth_factor,
                                ohem_rate=ohem_rate)

    def __call__(self, output, target, training=False):
        split_size = output.shape[0] // self.num_splits
        assert split_size * self.num_splits == output.shape[0]
        logits_split = torch.split(output, split_size)

        # Cross-entropy is only computed on clean images
        loss = self.loss(logits_split[0], target[:split_size], training=training)
        probs = [torch.sigmoid(logits) for logits in logits_split]

        # Clamp mixture distribution to avoid exploding KL divergence
        logp_mixture = torch.clamp(torch.stack(probs).mean(axis=0), 1e-7, 1).log()
        kl_loss = sum([F.kl_div(
            logp_mixture, p_split, reduction='batchmean') for p_split in probs]) / len(probs)
        return loss, kl_loss


class BengaliAiCrossEntropy(nn.Module):
    def __init__(self,
                 grapheme_weight=2.0,
                 vowel_weight=1.0,
                 consonant_weight=1.0,
                 smooth_factor=0,
                 ohem_rate=1.0,
                 jsd=None):
        super(BengaliAiCrossEntropy, self).__init__()

        self.grapheme_weight = grapheme_weight
        self.vowel_weight = vowel_weight
        self.consonant_weight = consonant_weight
        self.smooth_factor = smooth_factor
        self.ohem_rate = ohem_rate

        self.loss = SoftBCELoss(smooth_factor=smooth_factor, ohem_rate=ohem_rate)

        self.jsd = jsd
        if jsd is not None:
            self.jsd_loss = JsdCrossEntropy(
                num_splits=jsd['num_splits'],
                smooth_factor=smooth_factor,
                ohem_rate=ohem_rate
            )
            self.grapheme_alpha = self.jsd['alpha']['grapheme']
            self.vowel_alpha = self.jsd['alpha']['vowel']
            self.consonant_alpha = self.jsd['alpha']['consonant']

        self.iter = 0

    def __call__(self, pred, target, training=False):
        self.iter += 1
        grapheme_pred, vowel_pred, consonant_pred = pred
        grapheme_target, vowel_target, consonant_target = target
        loss = 0
        if training and self.jsd is not None:
            ce_loss, kl_loss = self.jsd_loss(grapheme_pred, grapheme_target, training=training)
            loss += self.grapheme_weight * ce_loss + self.grapheme_alpha * kl_loss

            ce_loss, kl_loss = self.jsd_loss(vowel_pred, vowel_target, training=training)
            loss += self.vowel_weight * ce_loss + self.vowel_alpha * kl_loss

            ce_loss, kl_loss = self.jsd_loss(consonant_pred, consonant_target, training=training)
            loss += self.consonant_weight * ce_loss + self.consonant_alpha * kl_loss
        else:
            loss += self.grapheme_weight \
                    * self.loss(grapheme_pred, grapheme_target, training=training)

            loss += self.vowel_weight \
                    * self.loss(vowel_pred, vowel_target, training=training)

            loss += self.consonant_weight \
                    * self.loss(consonant_pred, consonant_target, training=training)

        return loss
