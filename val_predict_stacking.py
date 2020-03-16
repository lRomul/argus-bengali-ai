import numpy as np

from src.stacking.datasets import get_folds_stacking_data
from src.utils import get_best_model_path
from src import config

from src.stacking.predictor import StackPredictor


STACKING_EXPERIMENT = "stacking_007"

EXPERIMENT_DIR = config.experiments_dir / STACKING_EXPERIMENT
PREDICTION_DIR = config.predictions_dir / STACKING_EXPERIMENT

EXPERIMENTS = [
    'effb3ns_005',
    'tf_efficientnet_b5_ns_cl_fv2',
]

DEVICE = 'cuda'
BATCH_SIZE = 256


def predict_val_fold(folds_data, predictor, fold):
    fold_prediction_dir = PREDICTION_DIR / f'fold_{fold}' / 'val'
    fold_prediction_dir.mkdir(parents=True, exist_ok=True)

    data = [s for s in folds_data if s['fold'] == fold]
    probs = np.stack([d['prob'] for d in data], axis=0)
    image_ids = [s['image_id'] for s in data]

    preds = predictor.predict(probs)
    grapheme_pred, vowel_pred, consonant_pred = preds

    np.savez(
        fold_prediction_dir / 'preds.npz',
        grapheme_pred=grapheme_pred,
        vowel_pred=vowel_pred,
        consonant_pred=consonant_pred,
        image_ids=image_ids,
    )


if __name__ == "__main__":
    folds_data = get_folds_stacking_data(EXPERIMENTS)

    for fold in config.folds:
        print("Predict fold", fold)
        fold_dir = EXPERIMENT_DIR / f'fold_{fold}'
        model_path = get_best_model_path(fold_dir)

        if model_path is None:
            print("Skip fold", fold)
            continue

        print("Model path", model_path)
        predictor = StackPredictor(model_path,
                                   batch_size=BATCH_SIZE,
                                   device=DEVICE)

        print("Val predict")
        predict_val_fold(folds_data, predictor, fold)
