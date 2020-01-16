import random
import numpy as np


def get_random_sample(dataset):
    rnd_idx = random.randint(0, len(dataset) - 1)
    rnd_image, rnd_target = dataset.get_sample(rnd_idx)
    return rnd_image, rnd_target


def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class MixUp:
    def __init__(self, alpha_dist='beta'):
        assert alpha_dist in ['uniform', 'beta']
        self.alpha_dist = alpha_dist

    def sample_alpha(self):
        if self.alpha_dist == 'uniform':
            return random.uniform(0, 0.5)
        elif self.alpha_dist == 'beta':
            return np.random.beta(0.4, 0.4)

    def __call__(self, dataset, image, target):
        rnd_image, rnd_target = get_random_sample(dataset)

        alpha = self.sample_alpha()
        image = (1 - alpha) * image + alpha * rnd_image
        new_target = []
        for trg, rnd_trg in zip(target, rnd_target):
            trg = (1 - alpha) * trg + alpha * rnd_trg
            new_target.append(trg)
        return image, new_target


class CutMix:
    def __init__(self, num_mix=1, beta=1.0, prob=1.0):
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

    def __call__(self, dataset, image, target):
        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rnd_image, rnd_target = get_random_sample(dataset)

            bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)
            rnd_image_crop = rnd_image[:, bbx1:bbx2, bby1:bby2]

            rnd_crop_sum = rnd_image_crop.sum()
            image_sum = image.sum() - image[:, bbx1:bbx2, bby1:bby2].sum()

            image[:, bbx1:bbx2, bby1:bby2] = rnd_image_crop
            lam = 1 - rnd_crop_sum / (rnd_crop_sum + image_sum)

            new_target = []
            for trg, rnd_trg in zip(target, rnd_target):
                trg = trg * lam + rnd_trg * (1 - lam)
                new_target.append(trg)
            target = new_target

        return image, target


class RandomMixer:
    def __init__(self, mixers, p=None):
        self.mixers = mixers
        self.p = p

    def __call__(self, dataset, image, target):
        mixer = np.random.choice(self.mixers, p=self.p)
        image, target = mixer(dataset, image, target)
        return image, target


class UseMixerWithProb:
    def __init__(self, mixer, prob=.5):
        self.mixer = mixer
        self.prob = prob

    def __call__(self, dataset, image, target):
        if random.random() < self.prob:
            return self.mixer(dataset, image, target)
        return image, target
