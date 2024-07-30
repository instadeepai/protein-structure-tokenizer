#################
# General setup #

SHELL := /bin/bash
ACCELERATOR = CPU
PORT = 8889

# variables
WORK_DIR = $(PWD)
USER_ID = $$(id -u)
GROUP_ID = $$(id -g)

DOCKER_BUILD_FLAGS = --build-arg USER_ID=$(USER_ID) \
	--build-arg GROUP_ID=$(GROUP_ID) \

DOCKER_RUN_FLAGS_CPU = --rm \
	--shm-size=1024m \
	-v $(WORK_DIR):/app

DOCKER_RUN_FLAGS_GPU = ${DOCKER_RUN_FLAGS_CPU} --gpus all --env-file=.env

DOCKER_RUN_FLAGS_TPU = --rm --user root --privileged -p ${PORT}:${PORT} --network host

# images name
DOCKER_IMAGE_NAME_CPU = structure_tokenizer_cpu
DOCKER_IMAGE_NAME_GPU = structure_tokenizer_gpu
DOCKER_IMAGE_NAME_TPU = structure_tokenizer_tpu

DOCKER_CONTAINER_NAME = structure_tokenizer_container

.PHONY: clean
clean:
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rfv
	find . | grep -E ".pytest_cache" | xargs rm -rfv
	find . | grep -E "nul" | xargs rm -rfv

ifeq ($(ACCELERATOR),CPU)
# cpu/gpu build not tested
.PHONY: build
build:
	sudo docker build -t $(DOCKER_IMAGE_NAME_CPU) -f build-source/dev.Dockerfile . \
		$(DOCKER_BUILD_FLAGS) --build-arg BUILD_FOR_TPU="false" --build-arg BUILD_FOR_GPU="false"

.PHONY: dev_container
dev_container: build
	sudo docker run -it $(DOCKER_RUN_FLAGS_CPU) $(DOCKER_IMAGE_NAME_CPU) /bin/bash

.PHONY: notebook
notebook: build
	echo "Make sure you have properly exposed your VM before, with the gcloud ssh command followed by -- -N -f -L $(PORT):localhost:$(PORT)"
	$(SUDO_FLAG) docker run -p $(PORT):$(PORT) -it $(DOCKER_RUN_FLAGS_CPU) \
		$(DOCKER_VARS_TO_PASS) $(DOCKER_IMAGE_NAME_CPU) \
		jupyter lab --port=$(PORT) --no-browser --ip=0.0.0.0 --allow-root


else ifeq ($(ACCELERATOR),GPU)
build:
	sudo docker build -t $(DOCKER_IMAGE_NAME_GPU) -f build-source/dev.Dockerfile . \
		$(DOCKER_BUILD_FLAGS) --build-arg BUILD_FOR_TPU="false" --build-arg BUILD_FOR_GPU="true" \
        --build-arg BASE_IMAGE="nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04"


.PHONY: dev_container
dev_container: build
	sudo docker run -it $(DOCKER_RUN_FLAGS_GPU) $(DOCKER_IMAGE_NAME_GPU) /bin/bash


.PHONY: notebook
notebook: build
	echo "Make sure you have properly exposed your VM before, with the gcloud ssh command followed by -- -N -f -L $(PORT):localhost:$(PORT)"
	$(SUDO_FLAG) docker run -p $(PORT):$(PORT) -it $(DOCKER_RUN_FLAGS_GPU) \
		$(DOCKER_VARS_TO_PASS) $(DOCKER_IMAGE_NAME_GPU) \
		jupyter lab --port=$(PORT) --no-browser --ip=0.0.0.0 --allow-root



else ifeq ($(ACCELERATOR),TPU)
.PHONY: build
build:
	sudo docker build -t $(DOCKER_IMAGE_NAME_TPU) -f build-source/dev.Dockerfile . \
		$(DOCKER_BUILD_FLAGS) --build-arg BASE_IMAGE="ubuntu:20.04" --build-arg BUILD_FOR_TPU="true" --build-arg BUILD_FOR_GPU="false"

.PHONY: dev_container
dev_container: build
		sudo docker run -it $(DOCKER_RUN_FLAGS_TPU) -v $(WORK_DIR):/app $(DOCKER_IMAGE_NAME_TPU) /bin/bash

.PHONY: notebook
notebook: build
		echo "Make sure you have properly exposed your VM before, with the gcloud ssh command followed by -- -N -f -L $(PORT):localhost:$(PORT)"
		sudo docker run -p $(PORT):$(PORT) -it $(DOCKER_RUN_FLAGS_TPU) -v $(DISK_DIR):/app/data\
		$(DOCKER_VARS_TO_PASS) -v $(WORK_DIR):/app $(DOCKER_IMAGE_NAME) \
		jupyter notebook --port=$(PORT) --no-browser --ip=0.0.0.0 --allow-root
endif