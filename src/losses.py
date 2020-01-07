from torch import nn


class BengaliAiCrossEntropy(nn.Module):
    def __init__(self,
                 grapheme_weight=2.0,
                 vowel_weight=1.0,
                 consonant_weight=1.0):
        super(BengaliAiCrossEntropy, self).__init__()

        self.grapheme_weight = grapheme_weight
        self.vowel_weight = vowel_weight
        self.consonant_weight = consonant_weight

        self.grapheme_bce = nn.CrossEntropyLoss()
        self.vowel_bce = nn.CrossEntropyLoss()
        self.consonant_bce = nn.CrossEntropyLoss()

    def __call__(self, pred, target):
        grapheme_pred, vowel_pred, consonant_pred = pred
        grapheme_target, vowel_target, consonant_target = target

        loss = 0
        if self.grapheme_weight:
            loss += self.grapheme_weight \
                    * self.grapheme_bce(grapheme_pred, grapheme_target)

        if self.vowel_weight:
            loss += self.vowel_weight \
                    * self.vowel_bce(vowel_pred, vowel_target)

        if self.consonant_weight:
            loss += self.consonant_weight \
                    * self.consonant_bce(consonant_pred, consonant_target)

        return loss
