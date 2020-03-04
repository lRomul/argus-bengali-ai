import argparse
import pandas as pd

from src.utils import get_best_model_path
from src.datasets import get_folds_data
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
IMAGE_SIZE = None


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


if __name__ == "__main__":
    transforms = get_transforms(train=False, size=IMAGE_SIZE)
    folds_data = get_folds_data()

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
