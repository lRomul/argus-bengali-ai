from torch import nn

from src import config


class Classifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super(Classifier, self).__init__()

        self.grapheme_root_fc = nn.Linear(in_features,
                                          config.n_grapheme_roots)
        self.vowel_diacritic_fc = nn.Linear(in_features,
                                            config.n_vowel_diacritics)
        self.consonant_diacritic_fc = nn.Linear(in_features,
                                                config.n_consonant_diacritics)

    def forward(self, x):
        grapheme = self.grapheme_root_fc(x)
        vowel = self.vowel_diacritic_fc(x)
        consonant = self.consonant_diacritic_fc(x)
        return grapheme, vowel, consonant
