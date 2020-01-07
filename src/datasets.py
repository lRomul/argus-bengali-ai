import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from src import config


def process_raw_image(image):
    image = 255 - image
    image = (image * (255.0 / image.max())).astype(np.uint8)
    image = image.reshape(config.raw_image_shape)
    return image


def get_train_image_date_df():
    data_df_lst = []
    for train_image_data_path in config.train_image_data_paths:
        data_df = pd.read_parquet(train_image_data_path)
        data_df.set_index('image_id', inplace=True)
        data_df_lst.append(data_df)

    train_image_data_df = pd.concat(data_df_lst)
    return train_image_data_df


def get_folds_data():
    train_folds_df = pd.read_csv(config.train_folds_path)
    train_image_date_df = get_train_image_date_df()

    folds_data = []
    for _, row in train_folds_df.iterrows():
        sample = dict(row)

        image = train_image_date_df.loc[row.image_id].values
        image = process_raw_image(image)

        sample['image'] = image
        folds_data.append(sample)

    return folds_data


class BengaliAiDataset(Dataset):
    def __init__(self,
                 data,
                 folds=None,
                 target=True,
                 transform=None):
        self.folds = folds
        self.target = target
        self.transform = transform
        if folds is None:
            self.data = data
        else:
            self.data = [s for s in data if s['fold'] in folds]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        image = sample['image']
        if self.transform is not None:
            image = self.transform(image)

        target = [
            sample['grapheme_root'],
            sample['vowel_diacritic'],
            sample['consonant_diacritic']
        ]
        target = [torch.tensor(t) for t in target]

        return image, target
