import argparse
import numpy as np
import pandas as pd

from src.utils import get_best_model_path, blend_predictions
from src.datasets import get_folds_data, get_test_data
from src.predictor import Predictor
from src.transforms import get_transforms
from src import config


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, type=str)
args = parser.parse_args()

EXPERIMENT_DIR = config.experiments_dir / args.experiment
PREDICTION_DIR = config.predictions_dir / args.experiment
DEVICE = 'cuda'
BATCH_SIZE = 256
IMAGE_SIZE = 128


def predict_val_fold(folds_data, predictor, fold):
    fold_prediction_dir = PREDICTION_DIR / f'fold_{fold}' / 'val'
    fold_prediction_dir.mkdir(parents=True, exist_ok=True)

    data = [s for s in folds_data if s['fold'] == fold]
    image_ids = [s['image_id'] for s in data]

    preds = predictor.predict(data)

    for name, pred in zip(config.class_map.keys(), preds):
        probs_df = pd.DataFrame(data=pred,
                                index=image_ids)
        probs_df.index.name = 'image_id'
        probs_df.to_csv(fold_prediction_dir / f'{name}_probs.csv')


def predict_test(test_data, predictor, fold):
    fold_prediction_dir = PREDICTION_DIR / f'fold_{fold}' / 'test'
    fold_prediction_dir.mkdir(parents=True, exist_ok=True)

    image_ids = [s['image_id'] for s in test_data]
    preds = predictor.predict(test_data)

    for name, pred in zip(config.class_map.keys(), preds):
        probs_df = pd.DataFrame(data=pred,
                                index=image_ids)
        probs_df.index.name = 'image_id'
        probs_df.to_csv(fold_prediction_dir / f'{name}_probs.csv')


def blend_test_predictions():
    row_ids = []
    pred_lst = []
    for name in config.class_map.keys():
        probs_df_lst = []
        for fold in config.folds:
            fold_probs_path = PREDICTION_DIR / f'fold_{fold}' / 'test' / f'{name}_probs.csv'
            probs_df = pd.read_csv(fold_probs_path)
            probs_df.set_index('image_id', inplace=True)
            probs_df_lst.append(probs_df)

        blend_df = blend_predictions(probs_df_lst, use_gmean=False)
        blend_pred = np.argmax(blend_df.values, axis=1)

        for image_id, pred in zip(blend_df.index, blend_pred):
            row_ids.append(f"{image_id}_{name}")
            pred_lst.append(pred)

    pred_df = pd.DataFrame({"row_id": row_ids, "target": pred_lst})

    if not config.kernel_mode:
        pred_df.to_csv(PREDICTION_DIR / 'submission.csv', index=False)
    else:
        pred_df.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    transforms = get_transforms(train=False, size=IMAGE_SIZE)
    folds_data = None
    if not config.kernel_mode:
        folds_data = get_folds_data()
    test_data = get_test_data()

    for fold in config.folds:
        print("Predict fold", fold)
        fold_dir = EXPERIMENT_DIR / f'fold_{fold}'
        model_path = get_best_model_path(fold_dir)

        print("Model path", model_path)
        predictor = Predictor(model_path,
                              batch_size=BATCH_SIZE,
                              transform=transforms,
                              device=DEVICE)

        if folds_data is not None:
            print("Val predict")
            predict_val_fold(folds_data, predictor, fold)

        print("Test predict")
        predict_test(test_data, predictor, fold)

    print("Blend folds predictions")
    blend_test_predictions()
