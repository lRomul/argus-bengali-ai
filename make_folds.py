import random
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from src import config


if __name__ == '__main__':
    random_state = 42

    random.seed(random_state)
    np.random.seed(random_state)

    train_df = pd.read_csv(config.train_csv_path)
    train_df['fold'] = -1

    kf = KFold(n_splits=config.n_folds, random_state=random_state, shuffle=True)

    for fold, (_, val_index) in enumerate(kf.split(train_df)):
        train_df.iloc[val_index, -1] = fold

    train_df.to_csv(config.train_folds_path, index=False)
    print(f"Train folds saved to '{config.train_folds_path}'")
