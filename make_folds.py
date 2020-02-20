import random
import numpy as np
import pandas as pd

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from src import config


if __name__ == '__main__':
    random_state = 42

    random.seed(random_state)
    np.random.seed(random_state)

    train_df = pd.read_csv(config.train_csv_path)
    train_df['id'] = train_df['image_id'].apply(lambda x: int(x.split('_')[1]))

    columns = ['id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic']
    X, y = train_df[columns].values[:, 0], train_df.values[:, 1:]

    train_df['fold'] = np.nan

    mskf = MultilabelStratifiedKFold(n_splits=config.n_folds, random_state=random_state, shuffle=True)
    for i, (_, test_index) in enumerate(mskf.split(X, y)):
        train_df.iloc[test_index, -1] = i

    train_df['fold'] = train_df['fold'].astype('int')

    train_df.to_csv(config.train_folds_path, index=False)
    print(f"Train folds saved to '{config.train_folds_path}'")
