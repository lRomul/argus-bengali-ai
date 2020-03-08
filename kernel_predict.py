import argparse
import numpy as np
import pandas as pd

from src.utils import get_best_model_path, blend_predictions
from src.datasets import get_test_data_generator
from src.predictor import Predictor
from src.transforms import get_transforms
from src import config


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, type=str)
args = parser.parse_args()

EXPERIMENTS = args.experiment.split(',')
DEVICE = 'cuda'
BATCH_SIZE = 256
IMAGE_SIZE = None


def predict_test(test_data, predictor, experiment, fold, batch_num):
    fold_prediction_dir = config.predictions_dir / experiment / f'fold_{fold}' / 'test'
    fold_prediction_dir.mkdir(parents=True, exist_ok=True)

    image_ids = [s['image_id'] for s in test_data]
    preds = predictor.predict(test_data)

    for class_name, pred in zip(config.class_map.keys(), preds):
        probs_df = pd.DataFrame(data=pred,
                                index=image_ids)
        probs_df.index.name = 'image_id'
        probs_df.to_csv(fold_prediction_dir
                        / f'{batch_num}_{class_name}_probs.csv')


def get_class_prediction_df(class_name):
    probs_df_lst = []
    dtypes = {str(c): np.float32 for c in config.class_map[class_name].keys()}
    for experiment in EXPERIMENTS:
        for fold in config.folds:
            fold_prediction_dir = config.predictions_dir / experiment / f'fold_{fold}' / 'test'
            fold_probs_paths = fold_prediction_dir.glob(f'*_{class_name}_probs.csv')
            fold_probs_paths = sorted(fold_probs_paths)
            if not fold_probs_paths:
                continue

            probs_batch_df_lst = []
            for fold_probs_path in fold_probs_paths:
                probs_batch_df = pd.read_csv(fold_probs_path, dtype=dtypes)
                probs_batch_df_lst.append(probs_batch_df)

            probs_df = pd.concat(probs_batch_df_lst)
            del probs_batch_df_lst
            probs_df.set_index('image_id', inplace=True)
            probs_df.sort_values("image_id", inplace=True)
            probs_df_lst.append(probs_df)

    if len(probs_df_lst) > 1:
        pred_df = blend_predictions(probs_df_lst, blend_type='mean')
    else:
        pred_df = probs_df_lst[0]

    return pred_df


def blend_test_predictions():
    row_ids = []
    pred_lst = []
    for class_name in config.class_map.keys():
        pred_df = get_class_prediction_df(class_name)
        prediction = np.argmax(pred_df.values, axis=1)

        for image_id, pred in zip(pred_df.index, prediction):
            row_ids.append(f"{image_id}_{class_name}")
            pred_lst.append(pred)

    pred_df = pd.DataFrame({"row_id": row_ids, "target": pred_lst})
    pred_df.sort_values("row_id", inplace=True)

    pred_df.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    transforms = get_transforms(train=False, size=IMAGE_SIZE)
    test_data_generator = get_test_data_generator(batch=2)

    for batch_num, test_data in enumerate(test_data_generator):
        print("Predict batch", batch_num)

        for experiment in EXPERIMENTS:
            print("Predict experiment", experiment)
            for fold in config.folds:
                fold_dir = config.experiments_dir / experiment / f'fold_{fold}'
                model_path = get_best_model_path(fold_dir)

                if model_path is None:
                    print("Skip fold", fold)
                    continue

                print("Predict fold", fold)
                print("Model path", model_path)
                predictor = Predictor(model_path,
                                      batch_size=BATCH_SIZE,
                                      transform=transforms,
                                      device=DEVICE)
                predict_test(test_data, predictor, experiment, fold, batch_num)

    print("Blend folds predictions")
    blend_test_predictions()
