import numpy as np
import pandas as pd
import random
import time
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
        for image_id, raw_image in zip(data_df.index, data_df.values):
            image = process_raw_image(raw_image)
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

    def _set_random_seed(self, idx):
        seed = int(time.time() * 1000.0) + idx
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))

    def __getitem__(self, idx):
        self._set_random_seed(idx)

        if not self.target:
            image = self.get_sample(idx)
            return image
        else:
            image, target = self.get_sample(idx)
            if self.mixer is not None:
                image, target = self.mixer(self, image, target)
            return image, target


def augmix_collate(batch):
    """ A fast collation function optimized for uint8 images (np array or torch) and int64 targets (labels)"""
    assert isinstance(batch[0], tuple)
    assert isinstance(batch[0][0], tuple)
    batch_size = len(batch)
    # This branch 'deinterleaves' and flattens tuples of input tensors into one tensor ordered by position
    # such that all tuple of position n will end up in a torch.split(tensor, batch_size) in nth position
    inner_tuple_size = len(batch[0][0])
    flattened_batch_size = batch_size * inner_tuple_size
    targets0 = torch.zeros((flattened_batch_size, *batch[0][1][0].shape), dtype=torch.float32)
    targets1 = torch.zeros((flattened_batch_size, *batch[0][1][1].shape), dtype=torch.float32)
    targets2 = torch.zeros((flattened_batch_size, *batch[0][1][2].shape), dtype=torch.float32)
    tensor = torch.zeros((flattened_batch_size, *batch[0][0][0].shape), dtype=torch.float32)
    for i in range(batch_size):
        assert len(batch[i][0]) == inner_tuple_size  # all input tensor tuples must be same length
        for j in range(inner_tuple_size):
            targets0[i + j * batch_size] += batch[i][1][0]
            targets1[i + j * batch_size] += batch[i][1][1]
            targets2[i + j * batch_size] += batch[i][1][2]
            tensor[i + j * batch_size] += batch[i][0][j]
    return tensor, (targets0, targets1, targets2)


class AugMixDataset(BengaliAiDataset):
    def __init__(self,
                 data,
                 folds=None,
                 transform=None,
                 mixer=None,
                 num_splits=3,
                 auto_augment=None):
        super().__init__(data, folds=folds, target=True,
                         transform=transform, mixer=mixer)
        self.num_splits = num_splits
        self.auto_augment = auto_augment

    def __getitem__(self, idx):
        self._set_random_seed(idx)

        image, target = self.get_sample(idx)
        if self.mixer is not None:
            image, target = self.mixer(self, image, target)

        images = [image]
        for _ in range(self.num_splits - 1):
            images.append(self.auto_augment(image))

        return tuple(images), target
