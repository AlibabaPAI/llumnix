#!/bin/bash

set -e

REPO="registry.cn-beijing.aliyuncs.com/llumnix/llumnix-dev"
DATE=$(date +%Y%m%d)
BRANCH=$(git rev-parse --abbrev-ref HEAD)
COMMIT_ID=$(git rev-parse --short=7 HEAD)
TAG="${DATE}_${BRANCH}_${COMMIT_ID}"

echo "Building docker image ${REPO}:${TAG}"

export DOCKER_BUILDKIT=1
docker build . \
    -f tools/docker/Dockerfile.vllm \
    --build-arg BRANCH=${BRANCH} \
    --build-arg COMMIT_ID=${COMMIT_ID} \
    -t ${REPO}:${TAG}
