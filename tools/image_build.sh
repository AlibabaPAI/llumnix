#!/bin/bash

set -e

usage() {
    echo "Usage: $0 --branch <llumnix_branch_name> --engine <vllm|bladellm>"
    echo "Options:"
    echo "  --branch <llumnix_branch_name>    Specify the branch to build from."
    echo "  --engine <vllm|bladellm>  Specify the engine type (vllm or bladellm)."
    exit 1
}

parse_args() {
    if [[ "$#" -lt 4 ]]; then
        usage
    fi

    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --branch)
                BRANCH="$2"
                shift 2
                ;;
            --engine)
                ENGINE="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                usage
                ;;
        esac
    done

    if [[ -z "$BRANCH" || -z "$ENGINE" ]]; then
        echo "Error: Both --branch and --engine must be specified."
        usage
    fi

    if [[ "$ENGINE" != "vllm" && "$ENGINE" != "bladellm" ]]; then
        echo "Error: Invalid engine type. Must be 'vllm' or 'bladellm'."
        usage
    fi
}

parse_args "$@"


REPO="registry.cn-beijing.aliyuncs.com/llumnix/llumnix-${ENGINE}-dev"
DATE=$(date +%Y%m%d%H%M)

COMMIT_ID=$(git ls-remote origin "refs/heads/$BRANCH" | awk '{print $1}')
if [[ -z "$COMMIT_ID" ]]; then
    echo "Error: Unable to find commit ID for branch '$BRANCH'."
    exit 1
fi

export DOCKER_BUILDKIT=1
DOCKERFILE="tools/docker/Dockerfile.${ENGINE}"
if [[ ! -f "$DOCKERFILE" ]]; then
    echo "Error: Dockerfile not found at path '$DOCKERFILE'."
    exit 1
fi

TAG="${DATE}_${BRANCH}_${COMMIT_ID}"
echo "Begin to build docker image ${REPO}:${TAG} ..."

docker build . \
    -f "$DOCKERFILE" \
    --build-arg BRANCH="$BRANCH" \
    --build-arg COMMIT_ID="$COMMIT_ID" \
    -t "${REPO}:${TAG}"

echo "Docker image built successfully: ${REPO}:${TAG}"
