import random
import numpy as np
import pandas as pd

from src import config


if __name__ == '__main__':
    random_state = 42

    random.seed(random_state)
    np.random.seed(random_state)

    train_df = pd.read_csv(config.train_csv_path)

    grapheme2idx = {grapheme: idx for idx, grapheme in enumerate(train_df.grapheme.unique())}
    train_df['grapheme_id'] = train_df['grapheme'].map(grapheme2idx)

    train_df['fold'] = 1
    train_df.loc[train_df.grapheme_id >= 1245, 'fold'] = 0

    train_df.to_csv(config.train_folds_path, index=False)
    print(f"Train folds saved to '{config.train_folds_path}'")
