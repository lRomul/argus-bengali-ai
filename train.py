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
from src.mixers import UseMixerWithProb, MixUp
from src.utils import initialize_amp
from src import config


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, type=str)
parser.add_argument('--fold', required=False, type=int)
args = parser.parse_args()

BATCH_SIZE = 256
NUM_WORKERS = 8
USE_AMP = True
MIX_PROB = 0.8

SAVE_DIR = config.experiments_dir / args.experiment
PARAMS = {
    'nn_module': ('cnn_finetune', {
        'model_name': 'se_resnext50_32x4d',
        'pretrained': True,
        'dropout_p': 0.0
    }),
    'loss': ('BengaliAiCrossEntropy', {
        'grapheme_weight': 90.323 * 2,
        'vowel_weight': 5.914,
        'consonant_weight': 3.763,
        'binary': True
    }),
    'optimizer': ('Adam', {'lr': 0.001}),
    'device': 'cuda'
}


def train_fold(save_dir, train_folds, val_folds):
    folds_data = get_folds_data()

    train_transform = get_transforms(train=True)
    mixer = UseMixerWithProb(MixUp(alpha_dist='beta'), MIX_PROB)
    test_transform = get_transforms(train=False)

    train_dataset = BengaliAiDataset(folds_data, train_folds,
                                     transform=train_transform,
                                     mixer=mixer)
    val_dataset = BengaliAiDataset(folds_data, val_folds, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True,
                              num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2,
                            shuffle=False, num_workers=NUM_WORKERS)

    model = BengaliAiModel(PARAMS)
    model.params['nn_module'][1]['pretrained'] = False

    if USE_AMP:
        initialize_amp(model)

    callbacks = [
        MonitorCheckpoint(save_dir, monitor='val_hierarchical_recall', max_saves=1),
        EarlyStopping(monitor='val_hierarchical_recall', patience=10),
        ReduceLROnPlateau(monitor='val_hierarchical_recall', factor=0.64, patience=3),
        LoggingToFile(save_dir / 'log.txt')
    ]

    model.fit(train_loader,
              val_loader=val_loader,
              max_epochs=100,
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
