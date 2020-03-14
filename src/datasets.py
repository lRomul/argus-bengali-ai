import numpy as np
import pandas as pd
import random
import time
import gc

import torch
from torch.utils.data import Dataset

from src.draw import BengaliDrawDataset
from src import config


def process_raw_image(image):
    image = 255 - image
    image = image.reshape(config.raw_image_shape)
    return image


def get_image_data_dict(image_data_paths, engine='auto'):
    image_data_dict = dict()
    for image_data_path in image_data_paths:
        print("Load parquet", image_data_path)
        data_df = pd.read_parquet(image_data_path, engine=engine)
        data_df.set_index('image_id', inplace=True)
        for image_id, raw_image in zip(data_df.index, data_df.values):
            image = process_raw_image(raw_image)
            image_data_dict[image_id] = image

    return image_data_dict


def get_folds_data(engine='auto'):
    print("Get folds data")
    train_folds_df = pd.read_csv(config.train_folds_path)
    train_image_data = get_image_data_dict(config.train_image_data_paths,
                                           engine=engine)

    folds_data = []
    for _, row in train_folds_df.iterrows():
        sample = dict(row)
        image = train_image_data[row.image_id]
        sample['image'] = image
        folds_data.append(sample)

    return folds_data


def get_test_data_generator(batch=None, engine='auto'):
    print("Get test data")
    if batch is None:
        batch = len(config.test_image_data_paths)

    for i in range(0, len(config.test_image_data_paths), batch):
        test_image_data_paths = config.test_image_data_paths[i:i + batch]
        test_image_data = get_image_data_dict(test_image_data_paths,
                                              engine=engine)

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
                 mixer=None,
                 black_list=None,
                 use_unseen=True,
                 draw_prob=0.0):
        self.folds = folds
        self.target = target
        self.transform = transform
        self.mixer = mixer
        self.draw_dataset = BengaliDrawDataset(config.fonts_dir,
                                               transform,
                                               mixer)
        self.draw_prob = draw_prob

        self.data = data

        if folds is not None:
            self.data = [s for s in self.data if s['fold'] in folds]

        if not use_unseen:
            print(f"Remove unseen samples from {len(self.data)} ", end='')
            self.data = [s for s in self.data if not s['unseen']]
            print(f"to {len(self.data)}")

        if black_list is not None:
            black_set = set(black_list)
            print(f"Remove {len(black_set)} samples from {len(self.data)} ", end='')
            self.data = [s for s in self.data if s['image_id'] not in black_set]
            print(f"to {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def get_sample(self, idx):
        sample = self.data[idx]

        image = sample['image'].copy()

        if not self.target:
            return image

        grapheme = torch.tensor(sample['grapheme_root'], dtype=torch.int64)
        vowel = torch.tensor(sample['vowel_diacritic'], dtype=torch.int64)
        consonant = torch.tensor(sample['consonant_diacritic'], dtype=torch.int64)
        target = grapheme, vowel, consonant

        return image, target

    def _set_random_seed(self, idx):
        seed = int(time.time() * 1000.0) + idx
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))

    @torch.no_grad()
    def __getitem__(self, idx):
        self._set_random_seed(idx)

        if not self.target:
            image = self.get_sample(idx)
            if self.transform is not None:
                image = self.transform(image)
            return image
        else:
            if self.draw_prob > random.random():
                random_index = np.random.randint(0, len(self.draw_dataset))
                image, target = self.draw_dataset.get_sample(random_index)
            else:
                image, target = self.get_sample(idx)
            if self.mixer is not None:
                image, target = self.mixer(self, image, target)
            if self.transform is not None:
                image = self.transform(image)
            return image, target
