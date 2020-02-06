import cv2
import torch
import random
import numpy as np

import albumentations as alb
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations import functional as F


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


def get_random_kernel():
    structure = np.random.choice([
        cv2.MORPH_RECT,
        cv2.MORPH_ELLIPSE,
        cv2.MORPH_CROSS]
    )
    ksize = tuple(np.random.randint(1, 6, 2))
    kernel = cv2.getStructuringElement(structure, ksize)
    return kernel


class Erosion:
    def __call__(self, image):
        image = cv2.erode(image, get_random_kernel(), iterations=1)
        return image


class Dilation:
    def __call__(self, image):
        image = cv2.dilate(image, get_random_kernel(), iterations=1)
        return image


class Opening:
    def __call__(self, image):
        image = cv2.erode(image, get_random_kernel(), iterations=1)
        image = cv2.dilate(image, get_random_kernel(), iterations=1)
        return image


class Closing:
    def __call__(self, image):
        image = cv2.dilate(image, get_random_kernel(), iterations=1)
        image = cv2.erode(image, get_random_kernel(), iterations=1)
        return image


class GridMask(DualTransform):
    """GridMask augmentation for image classification and object detection.

    Author: Qishen Ha
    Email: haqishen@gmail.com
    2020/01/29

    Args:
        num_grid (int): number of grid in a row or column.
        fill_value (int, float, lisf of int, list of float): value for dropped pixels.
        rotate ((int, int) or int): range from which a random angle is picked. If rotate is a single int
            an angle is picked from (-rotate, rotate). Default: (-90, 90)
        mode (int):
            0 - cropout a quarter of the square of each grid (left top)
            1 - reserve a quarter of the square of each grid (left top)
            2 - cropout 2 quarter of the square of each grid (left top & right bottom)

    Targets:
        image, mask

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/abs/2001.04086
    |  https://github.com/akuxcw/GridMask
    """

    def __init__(self, num_grid=3, fill_value=0, rotate=0, mode=0, always_apply=False, p=0.5):
        super(GridMask, self).__init__(always_apply, p)
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)
        if isinstance(rotate, int):
            rotate = (-rotate, rotate)
        self.num_grid = num_grid
        self.fill_value = fill_value
        self.rotate = rotate
        self.mode = mode
        self.masks = None
        self.rand_h_max = []
        self.rand_w_max = []

    def init_masks(self, height, width):
        if self.masks is None:
            self.masks = []
            n_masks = self.num_grid[1] - self.num_grid[0] + 1
            for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):
                grid_h = height / n_g
                grid_w = width / n_g
                this_mask = np.ones((int((n_g + 1) * grid_h), int((n_g + 1) * grid_w))).astype(np.uint8)
                for i in range(n_g + 1):
                    for j in range(n_g + 1):
                        this_mask[
                        int(i * grid_h): int(i * grid_h + grid_h / 2),
                        int(j * grid_w): int(j * grid_w + grid_w / 2)
                        ] = self.fill_value
                        if self.mode == 2:
                            this_mask[
                            int(i * grid_h + grid_h / 2): int(i * grid_h + grid_h),
                            int(j * grid_w + grid_w / 2): int(j * grid_w + grid_w)
                            ] = self.fill_value

                if self.mode == 1:
                    this_mask = 1 - this_mask

                self.masks.append(this_mask)
                self.rand_h_max.append(grid_h)
                self.rand_w_max.append(grid_w)

    def apply(self, image, mask, rand_h, rand_w, angle, **params):
        h, w = image.shape[:2]
        mask = F.rotate(mask, angle) if self.rotate[1] > 0 else mask
        mask = mask[:, :, np.newaxis] if image.ndim == 3 else mask
        image *= mask[rand_h:rand_h + h, rand_w:rand_w + w].astype(image.dtype)
        return image

    def get_params_dependent_on_targets(self, params):
        img = params['image']
        height, width = img.shape[:2]
        self.init_masks(height, width)

        mid = np.random.randint(len(self.masks))
        mask = self.masks[mid]
        rand_h = np.random.randint(self.rand_h_max[mid])
        rand_w = np.random.randint(self.rand_w_max[mid])
        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0

        return {'mask': mask, 'rand_h': rand_h, 'rand_w': rand_w, 'angle': angle}

    @property
    def targets_as_params(self):
        return ['image']

    def get_transform_init_args_names(self):
        return 'num_grid', 'fill_value', 'rotate', 'mode'


class Albumentations:
    def __init__(self, p=1.0):
        self.augmentation = alb.Compose([
                    alb.ShiftScaleRotate(
                        shift_limit=0.05,
                        scale_limit=0.25,
                        rotate_limit=15,
                        border_mode=cv2.BORDER_CONSTANT,
                        p=0.2
                    ),
                    alb.OneOf([
                        alb.GridDistortion(p=1.0,
                                           border_mode=cv2.BORDER_CONSTANT,
                                           distort_limit=0.25,
                                           num_steps=10)
                    ], p=0.2),
                ], p=p)

    def __call__(self, image):
        augmented = self.augmentation(image=image)
        image = augmented["image"]
        return image


class Resize:
    def __init__(self, size, interpolation=cv2.INTER_CUBIC):
        self.size = tuple(size)
        self.interpolation = interpolation

    def __call__(self, image):
        return cv2.resize(image, self.size, interpolation=self.interpolation)


def get_transforms(train, size):
    if size is None:
        resize = lambda x: x
    else:
        resize = Resize((size, size))

    if train:
        transforms = Compose([
            UseWithProb(OneOf([
                Erosion(),
                Dilation(),
                Opening(),
                Closing()
            ]), prob=0.2),
            resize,
            Albumentations(),
            ImageToTensor()
        ])
    else:
        transforms = Compose([
            resize,
            ImageToTensor()
        ])
    return transforms
