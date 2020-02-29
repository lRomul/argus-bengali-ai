import torch
import numpy as np
from sklearn.metrics import recall_score

from argus.metrics.metric import Metric


class Recall:
    def __init__(self):
        self.predictions = []
        self.targets = []

    def reset(self):
        self.predictions = []
        self.targets = []

    def update(self, pred, target):
        _, pred = torch.max(pred, 1)
        pred = pred.cpu().numpy()
        _, target = torch.max(target, 1)
        target = target.cpu().numpy()

        self.predictions.append(pred)
        self.targets.append(target)

    def compute(self):
        y_true = np.concatenate(self.targets, axis=0)
        y_pred = np.concatenate(self.predictions, axis=0)
        score = recall_score(y_true, y_pred, average='macro')
        return score


class HierarchicalRecall(Metric):
    name = 'hierarchical_recall'
    better = 'max'

    def reset(self):
        self.grapheme_recall = Recall()
        self.vowel_recall = Recall()
        self.consonant_recall = Recall()

    def update(self, step_output: dict):
        grapheme_pred, vowel_pred, consonant_pred = step_output['prediction'][:3]
        grapheme_target, vowel_target, consonant_target = step_output['target'][:3]

        self.grapheme_recall.update(grapheme_pred, grapheme_target)
        self.vowel_recall.update(vowel_pred, vowel_target)
        self.consonant_recall.update(consonant_pred, consonant_target)

    def compute(self):
        grapheme_score = self.grapheme_recall.compute()
        vowel_score = self.vowel_recall.compute()
        consonant_score = self.consonant_recall.compute()
        return grapheme_score, vowel_score, consonant_score

    def epoch_complete(self, state, name_prefix=''):
        grapheme_score, vowel_score, consonant_score = self.compute()
        score = np.average([grapheme_score, vowel_score, consonant_score],
                           weights=[2, 1, 1])
        state.metrics[name_prefix + self.name] = score
        state.metrics[name_prefix + 'grapheme_recall'] = grapheme_score
        state.metrics[name_prefix + 'vowel_recall'] = vowel_score
        state.metrics[name_prefix + 'consonant_recall'] = consonant_score
