import cv2
import torch
import random
import numpy as np

import albumentations as alb

from src import config


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


class ImageToTensor:
    def __call__(self, image):
        image = np.stack([image, image, image], axis=0)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)
        return image


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def crop_resize(image, size, pad=16,
                height=config.raw_image_shape[0],
                width=config.raw_image_shape[1]):
    # crop a box around pixels large than the threshold
    # some images contain line at the sides
    ymin, ymax, xmin, xmax = bbox(image[5:-5, 5:-5] > 80)
    # cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < width - 13) else width
    ymax = ymax + 10 if (ymax < height - 10) else height
    img = image[ymin:ymax, xmin:xmax]
    # remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax - xmin, ymax - ymin
    l = max(lx, ly) + pad
    # make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l - ly) // 2,), ((l - lx) // 2,)], mode='constant')
    return cv2.resize(img, (size, size))


class IafossCrop:
    def __init__(self, size, pad=16):
        self.size = size
        self.pad = pad

    def __call__(self, image):
        # Source: https://www.kaggle.com/iafoss/image-preprocessing-128x128
        image = crop_resize(image, size=self.size, pad=self.pad)
        return image


class Albumentations:
    def __init__(self, p=1.0):
        self.augmentation = alb.Compose([
                    alb.ShiftScaleRotate(
                        shift_limit=0.12,
                        scale_limit=0.12,
                        rotate_limit=20,
                        border_mode=cv2.BORDER_CONSTANT,
                        p=0.75
                    ),
                    alb.OneOf([
                        alb.OpticalDistortion(p=0.25,
                                              border_mode=cv2.BORDER_CONSTANT),
                        alb.GridDistortion(p=0.25,
                                           border_mode=cv2.BORDER_CONSTANT),
                        alb.IAAPiecewiseAffine(p=0.25),
                        alb.IAAPerspective(p=0.25),
                    ], p=0.25),
                ], p=p)

    def __call__(self, image):
        augmented = self.augmentation(image=image)
        image = augmented["image"]
        return image


def get_transforms(train, size):
    if train:
        transforms = Compose([
            IafossCrop(size),
            Albumentations(),
            ImageToTensor()
        ])
    else:
        transforms = Compose([
            IafossCrop(size),
            ImageToTensor()
        ])
    return transforms
