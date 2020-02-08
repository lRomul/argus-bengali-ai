import os
import json
import argparse
from subprocess import Popen

from argus.callbacks import (
    MonitorCheckpoint,
    EarlyStopping,
    LoggingToFile,
    ReduceLROnPlateau
)

from torch.utils.data import DataLoader

from src.datasets import BengaliAiDataset, get_folds_data
from src.argus_models import BengaliAiModel
from src.transforms import get_transforms
from src.mixers import UseMixerWithProb, CutMix
from src.utils import initialize_amp
from src import config


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, type=str)
parser.add_argument('--fold', required=False, type=int)
args = parser.parse_args()

IMAGE_SIZE = [128, 176, 224]
BATCH_SIZE = [448, 224, 154]
TRAIN_EPOCHS = [40, 40, 120]
BASE_LR = 0.001
NUM_WORKERS = 8
USE_AMP = True
MIX_PROB = 1.0
DEVICES = ['cuda']


def get_lr(base_lr, batch_size):
    return base_lr * (batch_size / 128)


SAVE_DIR = config.experiments_dir / args.experiment
PARAMS = {
    'nn_module': ('CustomResnet', {
        'encoder': 'gluon_resnet50_v1d',
        'pretrained': True,
        'classifier': ('fc', {'pooler': 'avgpool'})
    }),
    'loss': ('BengaliAiCrossEntropy', {
        'grapheme_weight': 9.032258064516129 * 2,
        'vowel_weight': 0.5913978494623656,
        'consonant_weight': 0.3763440860215054,
        'smooth_factor': 0.0,
        'ohem_rate': 1.0
    }),
    'optimizer': ('Over9000', {'lr': get_lr(BASE_LR, BATCH_SIZE[0])}),
    'device': DEVICES[0]
}


def train_fold(save_dir, train_folds, val_folds):
    folds_data = get_folds_data()

    model = BengaliAiModel(PARAMS)
    model.params['nn_module'][1]['pretrained'] = False

    if USE_AMP:
        initialize_amp(model)

    model.set_device(DEVICES)

    for image_size, batch_size, epochs in zip(IMAGE_SIZE, BATCH_SIZE, TRAIN_EPOCHS):
        print(f"Start train step: image_size {image_size}, batch_size {batch_size}, epochs {epochs}")
        model.set_lr(get_lr(BASE_LR, batch_size))

        train_transform = get_transforms(train=True, size=image_size)
        mixer = UseMixerWithProb(CutMix(num_mix=1, beta=1.0, prob=1.0), MIX_PROB)
        test_transform = get_transforms(train=False, size=image_size)

        train_dataset = BengaliAiDataset(folds_data, train_folds,
                                         transform=train_transform,
                                         mixer=mixer)
        val_dataset = BengaliAiDataset(folds_data, val_folds, transform=test_transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, drop_last=True,
                                  num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 2,
                                shuffle=False, num_workers=NUM_WORKERS)

        callbacks = [
            MonitorCheckpoint(save_dir, monitor='val_hierarchical_recall', max_saves=1),
            EarlyStopping(monitor='val_hierarchical_recall', patience=30),
            ReduceLROnPlateau(monitor='val_hierarchical_recall',
                              factor=0.64, patience=7, threshold=1e-7),
            LoggingToFile(save_dir / 'log.txt')
        ]

        model.fit(train_loader,
                  val_loader=val_loader,
                  max_epochs=epochs,
                  callbacks=callbacks,
                  metrics=['hierarchical_recall'])


if __name__ == "__main__":
    if args.fold is None:
        for fold in config.folds:
            command = [
                'python',
                os.path.abspath(__file__),
                '--experiment', args.experiment,
                '--fold', str(fold)
            ]
            pipe = Popen(command)
            pipe.wait()
    elif args.fold in config.folds:
        if not SAVE_DIR.exists():
            SAVE_DIR.mkdir(parents=True, exist_ok=True)

        with open(SAVE_DIR / 'source.py', 'w') as outfile:
            outfile.write(open(__file__).read())

        print("Model params", PARAMS)
        with open(SAVE_DIR / 'params.json', 'w') as outfile:
            json.dump(PARAMS, outfile)

        val_folds = [args.fold]
        train_folds = list(set(config.folds) - set(val_folds))
        save_fold_dir = SAVE_DIR / f'fold_{args.fold}'
        print(f"Val folds: {val_folds}, Train folds: {train_folds}")
        print(f"Fold save dir {save_fold_dir}")
        train_fold(save_fold_dir, train_folds, val_folds)
