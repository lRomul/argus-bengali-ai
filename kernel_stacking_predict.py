import numpy as np
import pandas as pd

from src.stacking.predictor import StackPredictor

from src.utils import get_best_model_path, blend_predictions
from src.datasets import get_test_data_generator
from src.predictor import Predictor
from src.transforms import get_transforms
from src import config


EXPERIMENTS = [
    'cooldown_004_nf',
    'cooldown_005',
]

STACK_EXPERIMENTS = [
    'stacking_001'
]

ENSEMBLE_EXPERIMENTS = [
    'cooldown_004_nf',
    'cooldown_005',
    'stacking_001'
]

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


def load_fold_experiment_pred(experiment_preds_dir, fold):
    grapheme_pred_lst = []
    vowel_pred_lst = []
    consonant_pred_lst = []
    image_ids_lst = []

    for batch in range(DATA_BATCH):
        preds_path = experiment_preds_dir / f'fold_{fold}' / f'preds_{batch}.npz'
        preds = np.load(preds_path)
        grapheme_pred_lst.append(preds['grapheme_pred'])
        vowel_pred_lst.append(preds['vowel_pred'])
        consonant_pred_lst.append(preds['consonant_pred'])
        image_ids_lst.append(preds['image_ids'])

    grapheme_pred = np.concatenate(grapheme_pred_lst)
    vowel_pred = np.concatenate(vowel_pred_lst)
    consonant_pred = np.concatenate(consonant_pred_lst)
    image_ids = np.concatenate(image_ids_lst)

    preds = grapheme_pred, vowel_pred, consonant_pred

    np.savez(
        experiment_preds_dir / f'fold_{fold}' / f'preds.npz',
        grapheme_pred=grapheme_pred,
        vowel_pred=vowel_pred,
        consonant_pred=consonant_pred,
        image_ids=image_ids,
    )

    return preds, image_ids


def load_experiment_predictions(experiment):
    experiment_preds_dir = config.tmp_predictions_dir / experiment

    grapheme_pred_lst = []
    vowel_pred_lst = []
    consonant_pred_lst = []

    prev_image_ids = None

    for fold in config.folds:
        preds, image_ids = load_fold_experiment_pred(experiment_preds_dir, fold)
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
    for experiment in ENSEMBLE_EXPERIMENTS:
        experiment_probs_df_lst = []
        for fold in config.folds:
            fold_prediction_dir = config.tmp_predictions_dir / experiment / f'fold_{fold}'
            preds = np.load(fold_prediction_dir / 'preds.npz')
            class_preds = preds[class_name.split('_')[0] + '_pred']

            probs_df = pd.DataFrame(data=class_preds,
                                    index=image_ids)
            probs_df.index.name = 'image_id'

            probs_df.sort_values("image_id", inplace=True)
            experiment_probs_df_lst.append(probs_df)

        if len(experiment_probs_df_lst) > 1:
            probs_df = blend_predictions(experiment_probs_df_lst, use_gmean=False)
        else:
            probs_df = experiment_probs_df_lst[0]

        probs_df_lst.append(probs_df)

    if len(probs_df_lst) > 1:
        pred_df = blend_predictions(probs_df_lst, use_gmean=False)
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
    test_data_generator = get_test_data_generator(batch=DATA_BATCH)

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

    preds, image_ids = load_experiments_predictions(EXPERIMENTS)

    for experiment in STACK_EXPERIMENTS:
        print("Predict stacking experiment", experiment)
        for fold in config.folds:
            fold_dir = config.experiments_dir / experiment / f'fold_{fold}'

            model_path = get_best_model_path(fold_dir)

            print("Predict fold", fold)
            print("Model path", model_path)
            predictor = StackPredictor(model_path,
                                       batch_size=BATCH_SIZE,
                                       device=DEVICE)

            predict_stacking_test(preds, predictor, experiment, fold)

    print("Blend folds predictions")
    blend_test_predictions()
