ARG BASE_IMAGE=nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04
FROM ${BASE_IMAGE}

ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Speed up the build, and avoid unnecessary writes to disk
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV PIPENV_VENV_IN_PROJECT=true PIP_NO_CACHE_DIR=false PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    g++ \
    wget \
    unzip \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Install Micromamba
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda

# Install conda dependencies
COPY build-source/requirements.txt /tmp/requirements.txt
COPY build-source/environment.yaml /tmp/environment.yaml

ARG BUILD_FOR_TPU
ARG BUILD_FOR_GPU

RUN if [ ${BUILD_FOR_TPU} = "false" ] ; then echo "Not building for tpu" ; \
    else sed -i 's/jax==/jax[tpu]==/g' /tmp/requirements.txt ; fi

RUN if [ ${BUILD_FOR_GPU} = "false" ] ; then echo "Installing required nvidia pypi registries " ; \
    else echo "nvidia-cudnn-cu12==8.9.7.29" >> /tmp/requirements.txt ; fi

RUN if [ ${BUILD_FOR_GPU} = "false" ] ; then echo "Not building for gpu" ; \
    else sed -i 's/jax==/jax[cuda12_pip]==/g' /tmp/requirements.txt && \
    sed -i 's/libtpu_releases\.html/jax_cuda_releases\.html/g' /tmp/environment.yaml; fi

RUN micromamba create -y --file /tmp/environment.yaml \
    && micromamba clean --all --yes \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete
ENV PATH=/opt/conda/envs/structure-tokenizer/bin/:$PATH
ENV LD_LIBRARY_PATH=/opt/conda/envs/structure-tokenizer/lib/:$LD_LIBRARY_PATH

COPY . /app
# Create main working folder
# RUN mkdir /app
WORKDIR /app
RUN pip install -e .
RUN pip install -U "huggingface_hub[cli]"

# Disable debug, info, and warning tensorflow logs
ENV TF_CPP_MIN_LOG_LEVEL=3
# By default use cpu as the backend for JAX, we will explicitely load data on gpus/tpus as needed.
# ENV JAX_PLATFORM_NAME="cpu"

# Installing the lddt and tm scores
# Install TMalign
RUN mkdir TMalign-build && \
    cd TMalign-build && \
    wget https://zhanglab.dcmb.med.umich.edu/TM-align/TMalign.cpp  --no-check-certificate &&\
    g++ -static -O3 -ffast-math -lm -o TMalign TMalign.cpp && \
    chmod +x ./TMalign && \
    mv TMalign /usr/local/bin/ && \
    cd ../ && \
    rm -rf TMalign-build

# Install TMscore
RUN mkdir TMscore-build && \
    cd TMscore-build && \
    wget https://seq2fun.dcmb.med.umich.edu/TM-score/TMscore.cpp  --no-check-certificate && \
    g++ -static -O3 -ffast-math -lm -o TMscore TMscore.cpp && \
    chmod +x ./TMscore && \
    mv TMscore /usr/local/bin/ && \
    cd ../ && \
    rm -rf TMscore-build

# # Install lddt
# RUN wget https://openstructure.org/static/lddt-linux.zip && \
#     unzip lddt-linux.zip && \
#     mv lddt-linux/lddt /usr/local/bin/ && \
#     mv lddt-linux/stereo_chemical_props.txt /usr/local/bin/ && \
#     rm lddt-linux.zip && \
#     rm -r lddt-linux

# aws
RUN apt update && apt upgrade; apt install curl; apt-get install unzip; curl "https://s3.amazonaws.com/aws-cli/awscli-bundle.zip" -o "awscli-bundle.zip"; unzip awscli-bundle.zip; ./awscli-bundle/install -i /usr/local/aws -b /usr/local/bin/aws

# Add eng user
# The id and group-id of 'eng' can be parametrized to match that of the user that will use this
# docker image so that the eng user can create files in mounted directories seamlessly (without
# permission issues).
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -f --gid ${GROUP_ID} eng
RUN useradd -l --gid ${GROUP_ID} --uid ${USER_ID} --shell /bin/bash --home-dir /app eng
RUN chown -R eng /app

USER eng
