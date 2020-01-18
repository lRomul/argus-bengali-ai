#!/usr/bin/env python3
# Kaggle script build system template: https://github.com/lopuhin/kaggle-script-template
import os
import base64
import gzip
from pathlib import Path


IGNORE_LIST = ["data/", "kernel/"]

PACKAGES = [
    ('https://github.com/lRomul/argus', 'v0.0.9'),
    ('https://github.com/creafz/pytorch-cnn-finetune', 'master'),
    ('https://github.com/Cadene/pretrained-models.pytorch', 'master')
]


def get_package_name(name):
    if name == "pytorch-cnn-finetune":
        return "cnn_finetune"
    elif name == "pretrained-models.pytorch":
        return "pretrainedmodels"
    else:
        return name


def encode_file(path: Path) -> str:
    compressed = gzip.compress(path.read_bytes(), compresslevel=9)
    return base64.b64encode(compressed).decode('utf-8')


def check_ignore(path: Path, ignore_list):
    if not path.is_file():
        return False
    for ignore in ignore_list:
        if str(path).startswith(ignore):
            return False
    return True


def clone_package(git_url, branch="master"):
    name = Path(git_url).name
    os.system('mkdir -p tmp')
    os.system(f'rm -rf tmp/{name}')
    os.system(f'cd tmp && git clone --depth 1 -b {branch} {git_url}')
    os.system(f'cp -R tmp/{name}/{get_package_name(name)} .')
    os.system(f'rm -rf tmp/{name}')


def remove_package(git_url):
    name = Path(git_url).name
    os.system(f'rm -rf {get_package_name(name)}')


def build_script(ignore_list, packages, template_name='kernel_template.py'):
    to_encode = []

    for path in Path('.').glob('**/*.py'):
        if check_ignore(path, ignore_list + packages):
            to_encode.append(path)

    for package, branch in packages:
        clone_package(package, branch)
        package_name = get_package_name(Path(package).name)
        for path in Path(package_name).glob('**/*'):
            if check_ignore(path, ignore_list):
                to_encode.append(path)

    file_data = {str(path): encode_file(path) for path in to_encode}
    print("Encoded python files:")
    for path in file_data:
        print(path)
    template = Path(template_name).read_text('utf8')
    (Path('kernel') / template_name).write_text(
        template.replace('{file_data}', str(file_data)),
        encoding='utf8')

    for package, branch in packages:
        remove_package(package)


if __name__ == '__main__':
    os.system('rm -rf kernel && mkdir kernel')
    build_script(IGNORE_LIST, PACKAGES,
                 template_name='kernel_template.py')
