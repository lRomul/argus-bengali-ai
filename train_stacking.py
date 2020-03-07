import json

from argus.callbacks import (
    EarlyStopping,
    LoggingToFile,
    MonitorCheckpoint
)

from torch.utils.data import DataLoader

from src.ema import EmaMonitorCheckpoint
from src.lr_schedulers import CosineAnnealingLR
from src.stacking.datasets import get_folds_stacking_data, StackingDataset
from src.stacking.argus_models import StackingModel
from src.utils import initialize_ema
from src import config


STACKING_EXPERIMENT = "stacking_001"

EXPERIMENTS = [
    'cooldown_004_nf',
    'cooldown_005',
]
USE_EMA = False
RS_PARAMS = {"base_size": 256, "reduction_scale": 4, "p_dropout": 0.02446562971778976, "lr": 4.796301375650003e-05,
             "epochs": 114, "eta_min_scale": 0.02442943902352819, "batch_size": 32}
BATCH_SIZE = RS_PARAMS['batch_size']
DATASET_SIZE = 128 * 256
NUM_WORKERS = 2

SAVE_DIR = config.experiments_dir / STACKING_EXPERIMENT
PARAMS = {
    'nn_module': ('FCNet', {
        'in_channels': len(EXPERIMENTS) * (config.n_grapheme_roots
                                           + config.n_vowel_diacritics
                                           + config.n_consonant_diacritics),
        'base_size': RS_PARAMS['base_size'],
        'reduction_scale': RS_PARAMS['reduction_scale'],
        'p_dropout': RS_PARAMS['p_dropout']
    }),
    'loss': ('BengaliAiCrossEntropy', {
        'grapheme_weight': 2.0,
        'vowel_weight': 1.0,
        'consonant_weight': 1.0,
        'smooth_factor': 0.1,
        'ohem_rate': 0.8
    }),
    'optimizer': ('Adam', {'lr': RS_PARAMS['lr']}),
    'device': 'cuda',
}


def train_fold(save_dir, train_folds, val_folds, folds_data):
    train_dataset = StackingDataset(folds_data, train_folds, size=DATASET_SIZE)
    val_dataset = StackingDataset(folds_data, val_folds)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True,
                              num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2,
                            shuffle=False, num_workers=NUM_WORKERS)

    model = StackingModel(PARAMS)

    if USE_EMA:
        initialize_ema(model, decay=0.999)
        checkpointer = EmaMonitorCheckpoint(save_dir, monitor='val_hierarchical_recall', max_saves=1)
    else:
        checkpointer = MonitorCheckpoint(save_dir, monitor='val_hierarchical_recall', max_saves=1)

    callbacks = [
        checkpointer,
        CosineAnnealingLR(T_max=RS_PARAMS['epochs'],
                          eta_min=RS_PARAMS['lr'] * RS_PARAMS['eta_min_scale']),
        EarlyStopping(monitor='val_hierarchical_recall', patience=30),
        LoggingToFile(save_dir / 'log.txt'),
    ]

    model.fit(train_loader,
              val_loader=val_loader,
              max_epochs=RS_PARAMS['epochs'],
              callbacks=callbacks,
              metrics=['hierarchical_recall'])


if __name__ == "__main__":
    if not SAVE_DIR.exists():
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Folder {SAVE_DIR} already exists.")

    with open(SAVE_DIR / 'source.py', 'w') as outfile:
        outfile.write(open(__file__).read())

    print("Model params", PARAMS)
    with open(SAVE_DIR / 'params.json', 'w') as outfile:
        json.dump(PARAMS, outfile)

    folds_data = get_folds_stacking_data(EXPERIMENTS)

    for fold in config.folds:
        val_folds = [fold]
        train_folds = list(set(config.folds) - set(val_folds))
        save_fold_dir = SAVE_DIR / f'fold_{fold}'
        print(f"Val folds: {val_folds}, Train folds: {train_folds}")
        print(f"Fold save dir {save_fold_dir}")
        train_fold(save_fold_dir, train_folds, val_folds, folds_data)
