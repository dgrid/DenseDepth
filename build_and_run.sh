#!/bin/bash
set -eu

DOCKER_FILE="$1"
IMAGE_NAME="$2"

docker build --tag ${IMAGE_NAME} --file ${DOCKER_FILE} .

docker run --rm -it --gpus all --user $(id -u):$(id -g) --mount type=bind,source="$(pwd)",target=/workspace ${IMAGE_NAME} bash
