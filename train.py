import os
import json
import argparse
from subprocess import Popen

from argus.callbacks import LoggingToFile

from torch.utils.data import DataLoader

from src.ema import EmaMonitorCheckpoint
from src.lr_schedulers import CosineAnnealingLR
from src.datasets import BengaliAiDataset, get_folds_data
from src.argus_models import BengaliAiModel
from src.transforms import get_transforms
from src.mixers import CutMix
from src.utils import initialize_amp, initialize_ema
from src import config


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, type=str)
parser.add_argument('--fold', required=False, type=int)
args = parser.parse_args()

IMAGE_SIZE = [128, None, None]
BATCH_SIZE = [313, 156, 156]
TRAIN_EPOCHS = [40, 120, 20]
COOLDOWN = [False, False, True]
BASE_LR = 0.001
NUM_WORKERS = 8
USE_AMP = True
USE_EMA = True
DEVICES = ['cuda']
BLACKLIST = config.input_data_dir / 'black_list_001.json'


def get_lr(base_lr, batch_size):
    return base_lr * (batch_size / 128)


SAVE_DIR = config.experiments_dir / args.experiment
PARAMS = {
    'nn_module': ('CustomEfficient', {
        'encoder': 'tf_efficientnet_b3_ns',
        'pretrained': True,
        'classifier': ('fc', {'pooler': None})
    }),
    'loss': ('BengaliAiCrossEntropy', {
        'grapheme_weight': 2.0,
        'vowel_weight': 1.0,
        'consonant_weight': 1.0,
        'smooth_factor': 0.1,
        'ohem_rate': 0.4
    }),
    'optimizer': ('AdamW', {'lr': get_lr(BASE_LR, BATCH_SIZE[0])}),
    'device': DEVICES[0],
}


def train_fold(save_dir, train_folds, val_folds):
    folds_data = get_folds_data()
    black_list = None
    if BLACKLIST is not None:
        with open(BLACKLIST) as file:
            black_list = json.load(file)

    model = BengaliAiModel(PARAMS)
    model.params['nn_module'][1]['pretrained'] = False

    if USE_AMP:
        initialize_amp(model)

    model.set_device(DEVICES)

    if USE_EMA:
        initialize_ema(model, decay=0.9999)

    lr_scheduler = CosineAnnealingLR(T_max=sum(TRAIN_EPOCHS), eta_min=1e-5)
    prev_batch = BATCH_SIZE[0]

    for image_size, batch_size, epochs, cooldown in zip(IMAGE_SIZE, BATCH_SIZE,
                                                        TRAIN_EPOCHS, COOLDOWN):
        print(f"Start train step: image_size {image_size}, batch_size {batch_size},"
              f" epochs {epochs}, cooldown {cooldown}")

        batch_lr_scale = batch_size / prev_batch
        model.set_lr(model.get_lr() * batch_lr_scale)

        train_transform = get_transforms(train=True, size=image_size, gridmask_p=0.5)
        if not cooldown:
            mixer = CutMix(beta=1.0)
        else:
            mixer = None
        test_transform = get_transforms(train=False, size=image_size)

        train_dataset = BengaliAiDataset(folds_data, train_folds,
                                         transform=train_transform,
                                         mixer=mixer,
                                         black_list=black_list)
        val_dataset = BengaliAiDataset(folds_data, val_folds, transform=test_transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, drop_last=True,
                                  num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 2,
                                shuffle=False, num_workers=NUM_WORKERS)

        callbacks = [
            EmaMonitorCheckpoint(save_dir, monitor='val_hierarchical_recall', max_saves=1),
            LoggingToFile(save_dir / 'log.txt'),
        ]

        if not cooldown:
            callbacks += [lr_scheduler]

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
