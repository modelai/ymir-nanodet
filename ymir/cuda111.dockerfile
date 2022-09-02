FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

ARG YMIR="1.1.0"
ENV PYTHONPATH=.
# change apt and pypy mirrors
RUN sed -i 's#http://archive.ubuntu.com#https://mirrors.ustc.edu.cn#g' /etc/apt/sources.list \
    && sed -i 's#http://security.ubuntu.com#https://mirrors.ustc.edu.cn#g' /etc/apt/sources.list

# Install linux package
# apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN	apt-get update && apt-get install -y gnupg2 git libglib2.0-0 \
    libgl1-mesa-glx libsm6 libxext6 libxrender-dev curl wget zip vim \
    build-essential ninja-build \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy file from host to docker and install requirements
COPY . /app
RUN cd /app && mkdir -p /img-man && mv /app/ymir/img-man/*-template.yaml /img-man/ \
    && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install -r /app/requirements.txt \
    && pip install "git+https://github.com/modelai/ymir-executor-sdk.git@ymir1.0.0" \
    && echo "python3 /app/ymir/start.py" > /usr/bin/start.sh

WORKDIR /app

# overwrite entrypoint to avoid ymir1.1.0 import docker image error.
ENTRYPOINT []
CMD bash /usr/bin/start.sh
