import torch
import numpy as np
from torch.utils.data import DataLoader

from argus import load_model


class StackPredictor:
    def __init__(self, model_path,
                 batch_size, device='cuda'):
        self.model = load_model(model_path, device=device)
        self.batch_size = batch_size

    def predict(self, probs):
        probs = probs.copy()
        stack_tensors = [torch.from_numpy(prob.astype(np.float32))
                         for prob in probs]

        loader = DataLoader(stack_tensors, batch_size=self.batch_size)

        grapheme_preds_lst = []
        vowel_preds_lst = []
        consonant_preds_lst = []

        for batch in loader:
            pred_batch = self.model.predict(batch)
            grapheme_pred, vowel_pred, consonant_pred = pred_batch

            grapheme_preds_lst.append(grapheme_pred)
            vowel_preds_lst.append(vowel_pred)
            consonant_preds_lst.append(consonant_pred)

        grapheme_pred = torch.cat(grapheme_preds_lst, dim=0)
        grapheme_pred = grapheme_pred.cpu().numpy()

        vowel_pred = torch.cat(vowel_preds_lst, dim=0)
        vowel_pred = vowel_pred.cpu().numpy()

        consonant_pred = torch.cat(consonant_preds_lst, dim=0)
        consonant_pred = consonant_pred.cpu().numpy()

        return grapheme_pred, vowel_pred, consonant_pred
