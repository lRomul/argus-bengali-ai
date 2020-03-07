import gzip
import base64
import os
from pathlib import Path
from typing import Dict


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


run('pip install /kaggle/input/bengali-ai-deps/pytorch-image-models')

run('python make_folds.py')
run('python kernel_stacking_predict.py')
run('rm -rf argus src tmp_predictions')
