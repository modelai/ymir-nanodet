FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

# support YMIR=1.0.0, 1.1.0 or 1.2.0
ARG YMIR="1.1.0"
ARG MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh"

ENV PYTHONPATH=.
ENV YMIR_VERSION=$YMIR

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV LANG=C.UTF-8
ENV PATH=/opt/conda/bin:$PATH

# change apt and pypy mirrors
RUN sed -i 's#http://archive.ubuntu.com#https://mirrors.ustc.edu.cn#g' /etc/apt/sources.list \
    && sed -i 's#http://security.ubuntu.com#https://mirrors.ustc.edu.cn#g' /etc/apt/sources.list \
    && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install linux package
RUN	apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC \
    && apt-get update && apt-get install -y gnupg2 git libglib2.0-0 \
    libgl1-mesa-glx libsm6 libxext6 libxrender-dev curl wget zip vim \
    build-essential ninja-build \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install miniconda
RUN wget "${MINICONDA_URL}" -O miniconda.sh -q && \
    mkdir -p /opt && \
    sh miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

# install pytorch
RUN /opt/conda/bin/conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c conda-forge

# Copy file from host to docker and install requirements
COPY . /app
RUN cd /app && mkdir -p /img-man && mv /app/ymir/img-man/*-template.yaml /img-man/ \
    && pip install -r /app/requirements.txt \
    && pip install "git+https://github.com/modelai/ymir-executor-sdk.git@ymir1.0.0" \
    && echo "python3 /app/start.py" > /usr/bin/start.sh

WORKDIR /app

# overwrite entrypoint to avoid ymir1.1.0 import docker image error.
ENTRYPOINT []
CMD bash /usr/bin/start.sh
