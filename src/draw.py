import time
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import torch
from torch.utils.data import Dataset

from src import config


def draw_grapheme(grapheme, font_path, size=(137, 236)):
    height, width = size
    image = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(image)
    font_size = np.random.randint(70, 110)
    font = ImageFont.truetype(str(font_path), font_size)
    w, h = draw.textsize(grapheme, font=font)
    width_ratio = np.random.uniform(1.5, 2.5)
    height_ratio = np.random.uniform(2.5, 3.5)
    fill = np.random.randint(200, 255)
    draw.text(((width - w) / width_ratio, (height - h) / height_ratio),
              grapheme, font=font, fill=fill)
    image = image.filter(ImageFilter.BLUR)
    return np.array(image)[:, :, 0]


def get_draw_data():
    graphemes = []
    for grapheme_root_idx, grapheme_root in config.class_map['grapheme_root'].items():
        for vowel_diacritic_idx, vowel_diacritic in config.class_map['vowel_diacritic'].items():
            for consonant_diacritic_idx, consonant_diacritic in config.class_map['consonant_diacritic'].items():
                consonant_diacritic, grapheme_root, vowel_diacritic = [c if c != '0' else '' for c in
                                                                       [consonant_diacritic, grapheme_root,
                                                                        vowel_diacritic]]

                grapheme = consonant_diacritic + grapheme_root + vowel_diacritic
                graphemes.append({
                    'grapheme': grapheme,
                    'grapheme_root': grapheme_root_idx,
                    'vowel_diacritic': vowel_diacritic_idx,
                    'consonant_diacritic': consonant_diacritic_idx
                })
    return graphemes


class BengaliDrawDataset(Dataset):
    def __init__(self,
                 fonts_dir,
                 transform=None,
                 mixer=None):
        self.fonts_dir = fonts_dir
        self.transform = transform
        self.mixer = mixer
        self.data = get_draw_data()
        self.font_paths = sorted(Path(fonts_dir).glob('*.ttf'))

    def __len__(self):
        return len(self.data)

    def get_sample(self, idx):
        sample = self.data[idx]

        font_path = np.random.choice(self.font_paths)
        image = draw_grapheme(sample['grapheme'], font_path,
                              size=config.raw_image_shape)

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

        image, target = self.get_sample(idx)
        if self.mixer is not None:
            image, target = self.mixer(self, image, target)
        if self.transform is not None:
            image = self.transform(image)
        return image, target
