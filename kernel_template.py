import gzip
import base64
import os
from pathlib import Path
from typing import Dict

EXPERIMENT_NAME = 'cooldown_005'
KERNEL_MODE = "predict"

# this is base64 encoded source code
file_data: Dict = {file_data}


for path, encoded in file_data.items():
    print(path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.decompress(base64.b64decode(encoded)))


def run(command):
    os.system('export PYTHONPATH=${PYTHONPATH}:/kaggle/working && '
              f'export KERNEL_MODE={KERNEL_MODE} && ' + command)


run('pip install /kaggle/input/bengali-ai-deps/pretrained-models.pytorch')
run('pip install /kaggle/input/bengali-ai-deps/pytorch-cnn-finetune')
run('pip install /kaggle/input/bengali-ai-deps/pytorch-image-models')

run('python make_folds.py')
run(f'python kernel_predict.py --experiment {EXPERIMENT_NAME}')
run('rm -rf argus cnn_finetune pretrainedmodels src predictions')
