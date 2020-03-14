import os
import pandas as pd
from pathlib import Path


def _get_class_map(path):
    class_map_df = pd.read_csv(path)
    class_map = {comp: dict() for comp in class_map_df.component_type.unique()}
    for i, row in class_map_df.iterrows():
        class_map[row.component_type][row.label] = row.component
    return class_map


kernel_mode = False
if 'KERNEL_MODE' in os.environ and os.environ['KERNEL_MODE'] == 'predict':
    kernel_mode = True

if kernel_mode:
    input_data_dir = Path('/kaggle/input/bengaliai-cv19/')
    output_data_dir = Path('/kaggle/working/')
else:
    input_data_dir = Path('/workdir/data/')
    output_data_dir = Path('/workdir/data/')

train_csv_path = input_data_dir / 'train.csv'
test_csv_path = input_data_dir / 'test.csv'
sample_submission = input_data_dir / 'sample_submission.csv'
raw_image_shape = 137, 236

class_map_csv_path = input_data_dir / 'class_map.csv'
test_image_data_paths = sorted(input_data_dir.glob('test_image_data_*.parquet'))
train_image_data_paths = sorted(input_data_dir.glob('train_image_data_*.parquet'))

train_folds_path = output_data_dir / 'train_folds_unseen_v5.csv'
if kernel_mode:
    experiments_dir = Path('/kaggle/input/bengali-ai-dataset/')
else:
    experiments_dir = output_data_dir / 'experiments'
predictions_dir = output_data_dir / 'predictions'
tmp_predictions_dir = output_data_dir / 'tmp_predictions'

n_folds = 5
folds = list(range(n_folds))
class_map = _get_class_map(class_map_csv_path)

n_grapheme_roots = len(class_map['grapheme_root'])
n_vowel_diacritics = len(class_map['vowel_diacritic'])
n_consonant_diacritics = len(class_map['consonant_diacritic'])
