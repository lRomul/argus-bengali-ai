import torch
import random
import numpy as np
from PIL import Image

from torchvision import transforms

from timm.data.auto_augment import (
    rand_augment_transform,
    augment_and_mix_transform,
    auto_augment_transform
)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, trg=None):
        if trg is None:
            for t in self.transforms:
                image = t(image)
            return image
        else:
            for t in self.transforms:
                image, trg = t(image, trg)
            return image, trg


class UseWithProb:
    def __init__(self, transform, prob=.5):
        self.transform = transform
        self.prob = prob

    def __call__(self, image, trg=None):
        if trg is None:
            if random.random() < self.prob:
                image = self.transform(image)
            return image
        else:
            if random.random() < self.prob:
                image, trg = self.transform(image, trg)
            return image, trg


class OneOf:
    def __init__(self, transforms, p=None):
        self.transforms = transforms
        self.p = p

    def __call__(self, image, trg=None):
        transform = np.random.choice(self.transforms, p=self.p)
        if trg is None:
            image = transform(image)
            return image
        else:
            image, trg = transform(image, trg)
            return image, trg


class PILImage:
    def __init__(self, size, resample=Image.BICUBIC):
        self.size = tuple(size)
        self.resample = resample

    def __call__(self, image):
        image = Image.fromarray(image)
        image = image.convert("RGB")
        image = image.resize(self.size, resample=self.resample)
        return image


def get_transforms(train, size, auto_augment='', resample=Image.BICUBIC):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    primary_tfl = [PILImage(size, resample=resample)]

    secondary_tfl = []
    if train and auto_augment:
        img_size_min = min(size)

        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )

        if auto_augment.startswith('rand'):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
        elif auto_augment.startswith('augmix'):
            aa_params['translate_pct'] = 0.3
            secondary_tfl += [augment_and_mix_transform(auto_augment, aa_params)]
        else:
            secondary_tfl += [auto_augment_transform(auto_augment, aa_params)]

    final_tfl = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor(mean),
            std=torch.tensor(std)
        )
    ]

    return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)
