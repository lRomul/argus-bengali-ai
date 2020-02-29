import torch
from torch.utils.data import DataLoader

from argus import load_model

from src.datasets import BengaliAiDataset


@torch.no_grad()
def predict_data(data, model, batch_size, transform):

    dataset = BengaliAiDataset(data,
                               target=False,
                               folds=None,
                               transform=transform)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False)

    grapheme_preds_lst = []
    vowel_preds_lst = []
    consonant_preds_lst = []

    for batch in loader:
        pred_batch = model.predict(batch)
        grapheme_pred, vowel_pred, consonant_pred = pred_batch[:3]

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


class Predictor:
    def __init__(self,
                 model_path,
                 batch_size,
                 transform,
                 device='cuda'):
        self.model = load_model(model_path, device=device)
        self.batch_size = batch_size
        self.transform = transform

    def predict(self, data):
        pred = predict_data(data, self.model,
                            self.batch_size, self.transform)
        return pred
