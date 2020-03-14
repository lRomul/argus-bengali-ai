import random
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from src import config


if __name__ == '__main__':
    random_state = 47
    num_unseen_graphemes = 100

    random.seed(random_state)
    np.random.seed(random_state)

    train_df = pd.read_csv(config.train_csv_path)
    train_df['fold'] = -1

    kf = KFold(n_splits=config.n_folds, random_state=random_state, shuffle=True)

    for fold, (_, val_index) in enumerate(kf.split(train_df)):
        train_df.iloc[val_index, -1] = fold

    grapheme_root2grapheme_count = dict()
    for grapheme_root in train_df.grapheme_root.unique():
        grapheme_root2grapheme_count[grapheme_root] = len(
            train_df[train_df.grapheme_root == grapheme_root].grapheme.unique())

    grapheme_root_for_unseen = list()
    for grapheme_root, count in grapheme_root2grapheme_count.items():
        if count >= config.n_folds:
            grapheme_root_for_unseen.append(grapheme_root)

    graphemes_for_unseen = train_df[
        train_df.grapheme_root.isin(set(grapheme_root_for_unseen))].grapheme.unique()
    unseen_graphemes = np.random.choice(graphemes_for_unseen, num_unseen_graphemes, replace=False)

    for grapheme in unseen_graphemes:
        fold = np.random.randint(0, config.n_folds)
        train_df.loc[train_df.grapheme == grapheme, 'fold'] = fold

    for fold in range(config.n_folds):
        print("Fold", fold)

        fold_index = train_df.fold == fold
        val_fold_data = train_df[fold_index]
        train_fold_data = train_df[~fold_index]

        assert len(set(train_fold_data.grapheme_root)) == config.n_grapheme_roots
        assert len(set(train_fold_data.vowel_diacritic)) == config.n_vowel_diacritics
        assert len(set(train_fold_data.consonant_diacritic)) == config.n_consonant_diacritics

        train_graphemes = set(train_fold_data.grapheme.unique())
        val_graphemes = set(val_fold_data.grapheme.unique())

        unseen_index = val_fold_data.grapheme.isin(val_graphemes - train_graphemes)
        print("Unseen to all samples ratio", val_fold_data[unseen_index].shape[0] / val_fold_data.shape[0])
        print("Unseen to all graphemes ratio", len(val_graphemes - train_graphemes) / len(val_graphemes))

    train_df.to_csv(config.train_folds_path, index=False)
    print(f"Train folds saved to '{config.train_folds_path}'")
