FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt-get update &&\
    apt-get -y install build-essential yasm nasm \
    cmake unzip git wget tmux nano \
    sysstat libtcmalloc-minimal4 pkgconf autoconf libtool \
    python3 python3-pip python3-dev python3-setuptools \
    libsm6 libxext6 libxrender-dev &&\
    ln -s /usr/bin/python3 /usr/bin/python &&\
    ln -s /usr/bin/pip3 /usr/bin/pip &&\
    apt-get clean &&\
    apt-get autoremove &&\
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf /var/cache/apt/archives/*

RUN pip3 install --upgrade pip==20.2.3
RUN pip3 install --no-cache-dir numpy==1.18.0

# Install PyTorch
RUN pip3 install --no-cache-dir \
    torch==1.3.1 \
    torchvision==0.4.2

RUN pip3 install -U pip
RUN git clone https://github.com/NVIDIA/apex &&\
    cd apex &&\
    git checkout 606c3dcccd6ca70f4b506714d38a193e0845ee7f &&\
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . &&\
    cd .. && rm -rf apex

# Install python packages
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH $PYTHONPATH:/workdir
ENV TORCH_HOME=/workdir/data/.torch

WORKDIR /workdir
