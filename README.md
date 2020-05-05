# Bengali.AI Handwritten Grapheme Classification

Source code of solution for [Bengali.AI Handwritten Grapheme Classification](https://www.kaggle.com/c/bengaliai-cv19) competition.

## Solution 

Key points: 
* Efficientnets
* CutMix, GridMask
* AdamW with cosine annealing
* EMA

## Quick setup and start 

### Requirements 

*  Nvidia drivers, CUDA >= 10.1, cuDNN >= 7
*  [Docker](https://www.docker.com/), [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) 

The provided dockerfile is supplied to build image with cuda support and cudnn.


### Preparations 

* Clone the repo, build docker image. 
    ```bash
    git clone https://github.com/lRomul/argus-bengali-ai.git
    cd argus-bengali-ai
    make build
    ```

* Download and extract [dataset](https://www.kaggle.com/c/bengaliai-cv19/data) to `data` folder.

### Run

* Run docker container 
```bash
make
```

* Train model
```bash
python train.py --experiment train_001
```

* Predict test and make submission 
```bash
python kernel_predict.py --experiment train_001
```
