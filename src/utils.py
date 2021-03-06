import re
import numpy as np
from pathlib import Path
from scipy.stats.mstats import gmean

from src.ema import ModelEma


def initialize_amp(model,
                   opt_level='O1',
                   keep_batchnorm_fp32=None,
                   loss_scale='dynamic'):
    from apex import amp
    model.nn_module, model.optimizer = amp.initialize(
        model.nn_module, model.optimizer,
        opt_level=opt_level,
        keep_batchnorm_fp32=keep_batchnorm_fp32,
        loss_scale=loss_scale
    )
    model.amp = amp


def initialize_ema(model, decay=0.9999, device='', resume=''):
    model.model_ema = ModelEma(model.nn_module,
                               decay=decay,
                               device=device,
                               resume=resume)


def blend_predictions(probs_df_lst, blend_type='mean'):
    blend_df = probs_df_lst[0].copy()
    blend_values = np.stack([df.loc[blend_df.index.values].values
                             for df in probs_df_lst], axis=0)
    if blend_type == 'gmean':
        blend_values = gmean(blend_values, axis=0)
    elif blend_type == 'mean':
        blend_values = np.mean(blend_values, axis=0)
    elif blend_type == 'max':
        blend_values = np.max(blend_values, axis=0)
    else:
        ValueError(f"Unknown blend type: {blend_type}")

    blend_df.values[:] = blend_values
    return blend_df


def get_best_model_path(dir_path: Path, return_score=False):
    model_scores = []
    for model_path in dir_path.glob('*.pth'):
        score = re.search(r'-(\d+(?:\.\d+)?).pth', str(model_path))
        if score is not None:
            score = float(score.group(0)[1:-4])
            model_scores.append((model_path, score))

    if not model_scores:
        return None

    model_score = sorted(model_scores, key=lambda x: x[1])
    best_model_path = model_score[-1][0]
    if return_score:
        best_score = model_score[-1][1]
        return best_model_path, best_score
    else:
        return best_model_path
