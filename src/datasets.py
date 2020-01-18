import pandas as pd
import gc

import torch
from torch.utils.data import Dataset

from src import config


def process_raw_image(image):
    image = 255 - image
    image = image.reshape(config.raw_image_shape)
    return image


def get_image_data_dict(image_data_paths):
    image_data_dict = dict()
    for image_data_path in image_data_paths:
        print("Load parquet", image_data_path)
        data_df = pd.read_parquet(image_data_path)
        data_df.set_index('image_id', inplace=True)
        for image_id, row in data_df.iterrows():
            image = process_raw_image(row.values)
            image_data_dict[image_id] = image

    return image_data_dict


def get_folds_data():
    print("Get folds data")
    train_folds_df = pd.read_csv(config.train_folds_path)
    train_image_data = get_image_data_dict(config.train_image_data_paths)

    folds_data = []
    for _, row in train_folds_df.iterrows():
        sample = dict(row)
        image = train_image_data[row.image_id]
        sample['image'] = image
        folds_data.append(sample)

    return folds_data


def get_test_data_generator(batch=None):
    print("Get test data")
    if batch is None:
        batch = len(config.test_image_data_paths)

    for i in range(0, len(config.test_image_data_paths), batch):
        test_image_data_paths = config.test_image_data_paths[i:i + batch]
        test_image_data = get_image_data_dict(test_image_data_paths)

        test_data = []
        for image_id, image in test_image_data.items():
            sample = {
                "image_id": image_id,
                "image": image
            }
            test_data.append(sample)

        yield test_data

        test_data = []
        gc.collect()


class BengaliAiDataset(Dataset):
    def __init__(self,
                 data,
                 folds=None,
                 target=True,
                 transform=None,
                 mixer=None):
        self.folds = folds
        self.target = target
        self.transform = transform
        self.mixer = mixer
        if folds is None:
            self.data = data
        else:
            self.data = [s for s in data if s['fold'] in folds]

    def __len__(self):
        return len(self.data)

    def get_sample(self, idx):
        sample = self.data[idx]

        image = sample['image']
        if self.transform is not None:
            image = self.transform(image)

        if not self.target:
            return image

        grapheme = torch.zeros(config.n_grapheme_roots, dtype=torch.float32)
        grapheme[sample['grapheme_root']] = 1.0
        vowel = torch.zeros(config.n_vowel_diacritics, dtype=torch.float32)
        vowel[sample['vowel_diacritic']] = 1.0
        consonant = torch.zeros(config.n_consonant_diacritics, dtype=torch.float32)
        consonant[sample['consonant_diacritic']] = 1.0
        target = grapheme, vowel, consonant

        return image, target

    def __getitem__(self, idx):
        if not self.target:
            image = self.get_sample(idx)
            return image
        else:
            image, target = self.get_sample(idx)
            if self.mixer is not None:
                image, target = self.mixer(self, image, target)
            return image, target
