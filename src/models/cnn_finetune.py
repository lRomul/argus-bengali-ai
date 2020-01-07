from torch import nn

from cnn_finetune import make_model

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


def get_cnn_finetune_model(model_name,
                           pretrained=True,
                           dropout_p=None):
    model = make_model(
        model_name,
        1,
        pretrained=pretrained,
        dropout_p=dropout_p,
        classifier_factory=Classifier
    )
    return model
