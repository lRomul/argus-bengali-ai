import torch
from torch import nn
import torch.nn.functional as F

from src import config

from torch.nn.parameter import Parameter


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p),
                        (x.size(-2), x.size(-1))).pow(1./p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ \
               + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) \
               + ', ' + 'eps=' + str(self.eps) + ')'


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.tanh(F.softplus(input))


class Classifier(nn.Module):
    def __init__(self, in_features, num_classes, pooler='avgpool'):
        super().__init__()

        if pooler == 'gem':
            self.pooler = GeM(p=3, eps=1e-6)
        elif pooler == 'avgpool':
            self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise NotImplementedError

        self.grapheme_root_fc = nn.Linear(in_features,
                                          config.n_grapheme_roots)
        self.vowel_diacritic_fc = nn.Linear(in_features,
                                            config.n_vowel_diacritics)
        self.consonant_diacritic_fc = nn.Linear(in_features,
                                                config.n_consonant_diacritics)

    def forward(self, x):
        x = self.pooler(x)
        x = torch.flatten(x, 1)

        grapheme = self.grapheme_root_fc(x)
        vowel = self.vowel_diacritic_fc(x)
        consonant = self.consonant_diacritic_fc(x)
        return grapheme, vowel, consonant
