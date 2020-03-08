import torch
import numpy as np
import pandas as pd
from datetime import datetime

from src.stacking.predictor import StackPredictor

from src.utils import get_best_model_path, blend_predictions
from src.datasets import get_test_data_generator
from src.predictor import Predictor
from src.transforms import get_transforms
from src import config


EXPERIMENTS = [
    'effb3ns_002',
]

STACK_FEATURES_EXPERIMENTS = []

STACK_EXPERIMENTS = []

BLEND_EXPERIMENTS = [
    'effb3ns_002'
]

BLEND_SOFTMAX = True
FOLDS_BLEND_TYPE = 'mean'
EXPERIMENTS_BLEND_TYPE = 'mean'
DEVICE = 'cuda'
BATCH_SIZE = 32
DATA_BATCH = 2
IMAGE_SIZE = None


def predict_test(test_data, predictor, experiment, fold, batch_num):
    fold_prediction_dir = config.tmp_predictions_dir / experiment / f'fold_{fold}'
    fold_prediction_dir.mkdir(parents=True, exist_ok=True)

    image_ids = [s['image_id'] for s in test_data]
    preds = predictor.predict(test_data)

    grapheme_pred, vowel_pred, consonant_pred = preds

    np.savez(
        fold_prediction_dir / f'preds_{batch_num}.npz',
        grapheme_pred=grapheme_pred,
        vowel_pred=vowel_pred,
        consonant_pred=consonant_pred,
        image_ids=image_ids,
    )


def concat_fold_experiment_pred(experiment_preds_dir, fold):
    grapheme_pred_lst = []
    vowel_pred_lst = []
    consonant_pred_lst = []
    image_ids_lst = []

    for batch in range(DATA_BATCH):
        preds_path = experiment_preds_dir / f'fold_{fold}' / f'preds_{batch}.npz'
        if not preds_path.exists():
            raise FileNotFoundError

        preds = np.load(preds_path)
        grapheme_pred_lst.append(preds['grapheme_pred'])
        vowel_pred_lst.append(preds['vowel_pred'])
        consonant_pred_lst.append(preds['consonant_pred'])
        image_ids_lst.append(preds['image_ids'])

    grapheme_pred = np.concatenate(grapheme_pred_lst)
    vowel_pred = np.concatenate(vowel_pred_lst)
    consonant_pred = np.concatenate(consonant_pred_lst)
    image_ids = np.concatenate(image_ids_lst)

    np.savez(
        experiment_preds_dir / f'fold_{fold}' / f'preds.npz',
        grapheme_pred=grapheme_pred,
        vowel_pred=vowel_pred,
        consonant_pred=consonant_pred,
        image_ids=image_ids,
    )


def load_fold_experiment_pred(experiment_preds_dir, fold):
    preds_path = experiment_preds_dir / f'fold_{fold}' / f'preds.npz'
    if not preds_path.exists():
        raise FileNotFoundError
    npz_preds = np.load()

    grapheme_pred = npz_preds['grapheme_pred']
    vowel_pred = npz_preds['vowel_pred']
    consonant_pred = npz_preds['consonant_pred']
    image_ids = npz_preds['image_ids']

    preds = grapheme_pred, vowel_pred, consonant_pred
    return preds, image_ids


def load_experiment_predictions(experiment):
    experiment_preds_dir = config.tmp_predictions_dir / experiment

    grapheme_pred_lst = []
    vowel_pred_lst = []
    consonant_pred_lst = []

    prev_image_ids = None

    for fold in config.folds:
        try:
            preds, image_ids = load_fold_experiment_pred(experiment_preds_dir, fold)
        except FileNotFoundError as e:
            print(f"Skip fold {fold} {experiment}")
            continue

        grapheme_pred, vowel_pred, consonant_pred = preds
        grapheme_pred_lst.append(grapheme_pred)
        vowel_pred_lst.append(vowel_pred)
        consonant_pred_lst.append(consonant_pred)

        if prev_image_ids is not None:
            assert np.all(prev_image_ids == image_ids)
        prev_image_ids = image_ids

    grapheme_pred = np.mean(grapheme_pred_lst, axis=0)
    vowel_pred = np.mean(vowel_pred_lst, axis=0)
    consonant_pred = np.mean(consonant_pred_lst, axis=0)

    preds = grapheme_pred, vowel_pred, consonant_pred
    preds = np.concatenate(preds, axis=1)

    return preds, prev_image_ids


def load_experiments_predictions(experiments):
    preds_lst = []
    prev_image_ids = None
    for experiment in experiments:
        preds, image_ids = load_experiment_predictions(experiment)

        if prev_image_ids is not None:
            assert np.all(prev_image_ids == image_ids)
        prev_image_ids = image_ids

        preds_lst.append(preds)

    image_ids = [str(img_id) for img_id in prev_image_ids]

    preds = np.concatenate(preds_lst, axis=1)
    return preds, image_ids


def predict_stacking_test(probs, predictor, experiment, fold):
    fold_prediction_dir = config.tmp_predictions_dir / experiment / f'fold_{fold}'
    fold_prediction_dir.mkdir(parents=True, exist_ok=True)

    preds = predictor.predict(probs)

    grapheme_pred, vowel_pred, consonant_pred = preds

    np.savez(
        fold_prediction_dir / f'preds.npz',
        grapheme_pred=grapheme_pred,
        vowel_pred=vowel_pred,
        consonant_pred=consonant_pred,
        image_ids=image_ids,
    )


def get_class_prediction_df(class_name):
    probs_df_lst = []
    for experiment in BLEND_EXPERIMENTS:
        experiment_probs_df_lst = []
        for fold in config.folds:
            fold_prediction_path = config.tmp_predictions_dir / experiment \
                                   / f'fold_{fold}' / 'preds.npz'
            if not fold_prediction_path.exists():
                print(f"Skip {class_name} {fold_prediction_path}")
                continue
            
            preds = np.load(fold_prediction_path)
            class_preds = preds[class_name.split('_')[0] + '_pred']
            image_ids = preds['image_ids']

            if BLEND_SOFTMAX:
                class_preds = torch.tensor(class_preds)
                class_preds = torch.softmax(class_preds, dim=1)
                class_preds = class_preds.numpy()

            probs_df = pd.DataFrame(data=class_preds,
                                    index=image_ids)
            probs_df.index.name = 'image_id'

            probs_df.sort_values("image_id", inplace=True)
            experiment_probs_df_lst.append(probs_df)

        if len(experiment_probs_df_lst) > 1:
            probs_df = blend_predictions(experiment_probs_df_lst,
                                         blend_type=FOLDS_BLEND_TYPE)
        else:
            probs_df = experiment_probs_df_lst[0]

        probs_df_lst.append(probs_df)

    if len(probs_df_lst) > 1:
        pred_df = blend_predictions(probs_df_lst,
                                    blend_type=EXPERIMENTS_BLEND_TYPE)
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
    print("Experiments:", EXPERIMENTS)
    print("Stack features experiments:", STACK_FEATURES_EXPERIMENTS)
    print("Stack experiments:", STACK_EXPERIMENTS)
    print("Blend experiments:", BLEND_EXPERIMENTS)
    print("Blend softmax:", BLEND_SOFTMAX)
    print("Folds blend type:", FOLDS_BLEND_TYPE)
    print("Experiments blend type:", EXPERIMENTS_BLEND_TYPE)
    print("Batch size:", BATCH_SIZE)
    print("Data batch size:", DATA_BATCH)
    print("Image size:", IMAGE_SIZE)
    print("Time", datetime.now())

    transforms = get_transforms(train=False, size=IMAGE_SIZE)
    test_data_generator = get_test_data_generator(batch=DATA_BATCH, engine='fastparquet')

    for batch_num, test_data in enumerate(test_data_generator):
        print(datetime.now(), "Predict batch", batch_num)

        for experiment in EXPERIMENTS:
            print(datetime.now(), "Predict experiment", experiment)
            for fold in config.folds:
                fold_dir = config.experiments_dir / experiment / f'fold_{fold}'
                model_path = get_best_model_path(fold_dir)

                if model_path is None:
                    print(datetime.now(), "Skip fold", fold)
                    continue

                print(datetime.now(), "Predict fold", fold)
                print("Model path", model_path)
                predictor = Predictor(model_path,
                                      batch_size=BATCH_SIZE,
                                      transform=transforms,
                                      device=DEVICE)
                predict_test(test_data, predictor, experiment, fold, batch_num)

    print(datetime.now(), "Concat data batch predictions")
    for experiment in EXPERIMENTS:
        for fold in config.folds:
            experiment_preds_dir = config.tmp_predictions_dir / experiment
            try:
                concat_fold_experiment_pred(experiment_preds_dir, fold)
            except FileNotFoundError as e:
                print(datetime.now(), f"Skip fold {fold} {experiment}")
                continue

    if STACK_FEATURES_EXPERIMENTS and STACK_EXPERIMENTS:
        preds, image_ids = load_experiments_predictions(STACK_FEATURES_EXPERIMENTS)
        for experiment in STACK_EXPERIMENTS:
            print(datetime.now(), "Predict stacking experiment", experiment)
            for fold in config.folds:
                fold_dir = config.experiments_dir / experiment / f'fold_{fold}'

                model_path = get_best_model_path(fold_dir)

                print(datetime.now(), "Predict fold", fold)
                print("Model path", model_path)
                predictor = StackPredictor(model_path,
                                           batch_size=BATCH_SIZE,
                                           device=DEVICE)

                predict_stacking_test(preds, predictor, experiment, fold)

    print(datetime.now(), "Blend folds predictions")
    blend_test_predictions()
