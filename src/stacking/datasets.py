import time
import torch
import random
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from src import config


def load_experiment_predictions(experiment):
    experiment_preds_dir = config.predictions_dir / experiment

    grapheme_pred_lst = []
    vowel_pred_lst = []
    consonant_pred_lst = []
    image_ids_lst = []

    for fold in config.folds:
        preds_path = experiment_preds_dir / f'fold_{fold}' / 'val' / 'preds.npz'
        preds = np.load(preds_path)
        grapheme_pred_lst.append(preds['grapheme_pred'])
        vowel_pred_lst.append(preds['vowel_pred'])
        consonant_pred_lst.append(preds['consonant_pred'])
        image_ids_lst.append(preds['image_ids'])

    grapheme_pred = np.concatenate(grapheme_pred_lst)
    vowel_pred = np.concatenate(vowel_pred_lst)
    consonant_pred = np.concatenate(consonant_pred_lst)
    image_ids = np.concatenate(image_ids_lst)

    preds = grapheme_pred, vowel_pred, consonant_pred
    preds = np.concatenate(preds, axis=1)

    return preds, image_ids


def load_experiments_predictions(experiments):
    preds_lst = []
    prev_image_ids = None
    for experiment in experiments:
        preds, image_ids = load_experiment_predictions(experiment)

        if prev_image_ids is not None:
            assert np.all(prev_image_ids == image_ids)
        prev_image_ids = image_ids

        preds_lst.append(preds)

    image_ids = [str(img_id) for img_id in prev_image_ids]

    preds = np.concatenate(preds_lst, axis=1)
    return preds, image_ids


def get_folds_stacking_data(experiments):
    print("Get folds stacking data")
    train_folds_df = pd.read_csv(config.train_folds_path, index_col=0)
    train_folds_dict = train_folds_df.to_dict('index')

    probs, image_ids = load_experiments_predictions(experiments)

    folds_data = []
    for prob, image_id in zip(probs, image_ids):
        sample = train_folds_dict[image_id]
        sample['prob'] = prob
        sample['image_id'] = image_id
        folds_data.append(sample)

    return folds_data


class StackingDataset(Dataset):
    def __init__(self, data, folds,
                 size=None,
                 target=True,
                 black_list=None):
        super().__init__()
        self.folds = folds
        self.size = size
        self.target = target

        if folds is None:
            self.data = data
        else:
            self.data = [s for s in data if s['fold'] in folds]

        if black_list is not None:
            black_set = set(black_list)
            print(f"Remove {len(black_set)} samples from {len(self.data)} ", end='')
            self.data = [s for s in self.data if s['image_id'] not in black_set]
            print(f"to {len(self.data)}")

    def __len__(self):
        if self.size is None:
            return len(self.data)
        else:
            return self.size

    def get_sample(self, idx):
        sample = self.data[idx]

        probs = sample['prob'].copy()
        probs = torch.from_numpy(probs)

        if not self.target:
            return probs

        grapheme = torch.tensor(sample['grapheme_root'], dtype=torch.int64)
        vowel = torch.tensor(sample['vowel_diacritic'], dtype=torch.int64)
        consonant = torch.tensor(sample['consonant_diacritic'], dtype=torch.int64)
        target = grapheme, vowel, consonant

        return probs, target

    def __getitem__(self, idx):
        if self.size is not None:
            seed = int(time.time() * 1000.0) + idx
            random.seed(seed)
            np.random.seed(seed % (2 ** 31))
            idx = np.random.randint(len(self.data))

        probs, target = self.get_sample(idx)

        return probs, target
