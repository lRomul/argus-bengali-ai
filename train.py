import os
import json
import argparse
from subprocess import Popen

import argus
from argus.callbacks import (
    MonitorCheckpoint,
    EarlyStopping,
    LoggingToFile,
    ReduceLROnPlateau
)

from torch.utils.data import DataLoader
from torchcontrib.optim import SWA

from src.datasets import BengaliAiDataset, get_folds_data
from src.argus_models import BengaliAiModel
from src.transforms import get_transforms
from src.mixers import UseMixerWithProb, CutMix
from src.swa import SWACallback
from src.utils import initialize_amp, get_best_model_path
from src import config


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, type=str)
parser.add_argument('--fold', required=False, type=int)
args = parser.parse_args()

BATCH_SIZE = 512
NUM_WORKERS = 12
USE_AMP = True
MIX_PROB = 1.0
SWA_EPOCHS = 3
SWA_LR = 0.00003
DEVICES = ['cuda']

SAVE_DIR = config.experiments_dir / args.experiment
PARAMS = {
    'nn_module': ('cnn_finetune', {
        'model_name': 'resnet34',
        'pretrained': True,
        'dropout_p': 0.0
    }),
    'loss': ('BengaliAiCrossEntropy', {
        'grapheme_weight': 9.032258064516129 * 2,
        'vowel_weight': 0.5913978494623656,
        'consonant_weight': 0.3763440860215054,
        'binary': True
    }),
    'optimizer': ('Over9000', {'lr': 0.004}),
    'device': DEVICES[0]
}


def train_fold(save_dir, train_folds, val_folds):
    folds_data = get_folds_data()

    train_transform = get_transforms(train=True)
    mixer = UseMixerWithProb(CutMix(num_mix=1, beta=1.0, prob=1.0), MIX_PROB)
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
    model.set_device(DEVICES)

    callbacks = [
        MonitorCheckpoint(save_dir, monitor='val_hierarchical_recall', max_saves=1),
        EarlyStopping(monitor='val_hierarchical_recall', patience=20),
        ReduceLROnPlateau(monitor='val_hierarchical_recall', factor=0.64, patience=5),
        LoggingToFile(save_dir / 'log.txt')
    ]

    print("Start training")
    model.fit(train_loader,
              val_loader=val_loader,
              max_epochs=500,
              callbacks=callbacks,
              metrics=['hierarchical_recall'])

    model_path = get_best_model_path(save_dir)
    model = argus.load_model(model_path)
    model.set_device(DEVICES)
    model.set_lr(SWA_LR)

    print("Start SWA training", model_path)
    swa_model_path = save_dir / 'swa.pth'
    model.fit(train_loader,
              val_loader=val_loader,
              max_epochs=SWA_EPOCHS,
              callbacks=[
                  SWACallback(swa_start=0,
                              swa_freq=1,
                              swa_save_path=swa_model_path,
                              use_amp=USE_AMP),
                  LoggingToFile(save_dir / 'log.txt')
              ],
              metrics=['hierarchical_recall'])

    model = argus.load_model(swa_model_path)
    SWA.bn_update(train_loader, model.nn_module, device=DEVICES[0])
    model.save(swa_model_path)
    model = argus.load_model(swa_model_path)

    print("Start SWA validation")
    model.validate(val_loader,
                   metrics=['hierarchical_recall'],
                   callbacks=[
                       MonitorCheckpoint(save_dir,
                                         monitor='val_hierarchical_recall',
                                         max_saves=1)
                   ])


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
