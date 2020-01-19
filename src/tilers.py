import cv2
import torch
import random
import numpy as np

import albumentations as alb


def get_random_mixed_sample(dataset):
    rnd_idx = random.randint(0, len(dataset) - 1)
    rnd_image, rnd_target = dataset.get_mixed_sample(rnd_idx)
    return rnd_image, rnd_target


def axis_min_max(img, axis=0):
    assert axis in [0, 1]
    axis = 0 if axis else 1
    any = np.any(img, axis=axis)
    try:
        amin, amax = np.where(any)[0][[0, -1]]
    except:
        amin, amax = 0, len(any) - 1

    return amin, amax


class StackTiler:
    def __init__(self, size, axis=0, prob=1.0):
        assert axis in [0, 1]
        self.size = size
        self.axis = axis
        self.prob = prob
        self.pad_to_size = alb.Compose([
            alb.PadIfNeeded(min_height=size[0], min_width=size[1],
                            border_mode=cv2.BORDER_CONSTANT),
            alb.RandomCrop(size[0], size[1])
        ])

    def __call__(self, dataset, image, target):
        if random.random() < self.prob:
            rnd_image, rnd_target = get_random_mixed_sample(dataset)

            min1, max1 = axis_min_max(image[0].numpy() > 0.5, axis=self.axis)
            min2, max2 = axis_min_max(rnd_image[0].numpy() > 0.5, axis=self.axis)

            if self.axis == 0:
                tile_image = torch.cat([image[:, :max1],
                                        rnd_image[:, min2:]], dim=1)
            else:
                tile_image = torch.cat([image[:, :, :max1],
                                        rnd_image[:, :, min2:]], dim=2)

            new_target = []
            for trg, rnd_trg in zip(target, rnd_target):
                trg = trg + rnd_trg
                trg = torch.clamp(trg, 0.0, 1.0)
                new_target.append(trg)
        else:
            tile_image = image
            new_target = target

        tile_image = np.transpose(tile_image.numpy(), (1, 2, 0))
        tile_image = self.pad_to_size(image=tile_image)['image']
        tile_image = np.transpose(tile_image, (2, 0, 1))
        tile_image = torch.from_numpy(tile_image)

        return tile_image, new_target
