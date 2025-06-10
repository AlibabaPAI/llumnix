#!/bin/bash

# Enable strict mode
set -euo pipefail

# Constants
SCRIPT_NAME=$(basename "$0")
WORK_DIR="/workspace"
CURRENT_DIR=$(pwd)
DOCKERFILE="${DOCKERFILE:-tools/docker/Dockerfile}"

function usage() {
    echo "Usage: $SCRIPT_NAME <engine-release-image> <engine_type>"
    echo "Example: $SCRIPT_NAME vllm:0.0.1 vllm"
    exit 1
}

function validate_engine() {
    local engine="$1"
    case "$engine" in
        bladellm|vllm)
            ;;
        *)
            echo "Error: Invalid engine '$engine'"
            usage
            ;;
    esac
}

function build_wheel() {
    local release_image="$1"

    echo "[+] Building wheel package using image: $release_image..."

    docker run --rm \
        -v "${CURRENT_DIR}:${WORK_DIR}" \
        -w "${WORK_DIR}" \
        "$release_image" \
        sh -c "python3 setup.py bdist_wheel"

    # Extract wheel name and version
    WHEEL_NAME=$(ls dist/*.whl | xargs basename)
    VERSION=$(echo "$WHEEL_NAME" | grep -oP 'llumnix-\K([0-9]+\.[0-9]+\.[0-9]+)')

    export WHEEL_NAME VERSION
}

function get_commit_id_and_time() {
    COMMIT_ID=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    BUILD_TIME=$(date "+%Y%m%d%H%M")
    export COMMIT_ID
    export BUILD_TIME
}

function build_final_image() {
    local new_image="$1" base_image="$2" engine="$3"

    echo "[+] Building Docker image: $new_image"

    DOCKER_BUILDKIT=1 docker build \
        -f "$DOCKERFILE" \
        -t "$new_image" \
        --build-arg BASE_IMAGE="$base_image" \
        --build-arg ENGINE="$engine" \
        .
}

install_erdma() {
    echo "[+] Installing erdma packages..."
    wget -qO - http://mirrors.cloud.aliyuncs.com/erdma/GPGKEY | gpg --dearmour -o /etc/apt/trusted.gpg.d/erdma.gpg && \
    echo "deb [ ] http://mirrors.cloud.aliyuncs.com/erdma/apt/ubuntu jammy/erdma main" | tee /etc/apt/sources.list.d/erdma.list && \
    apt update && \
    apt install -y libibverbs1 ibverbs-providers ibverbs-utils librdmacm1 && \
    apt clean && \
    echo "[+] Erdma installation completed successfully."
}

function run_bladellm_tests() {
    local image="$1"
    set +e

    echo "[+] Running end-to-end tests in container..."

    (
        set +e
        nvidia-docker run --rm \
            -v "${CURRENT_DIR}:${WORK_DIR}" \
            -v /mnt:/mnt \
            -w "${WORK_DIR}" \
            --device=/dev/infiniband/uverbs0 \
            --device=/dev/infiniband/uverbs1 \
            --device=/dev/infiniband/rdma_cm \
            --ulimit memlock=-1 \
            "$image" \
            sh -c "
                set -e;
                /bin/bash -c '$(declare -f install_erdma); install_erdma'; \
                mkdir -p /tmp/llumnix/naming/ && \
                make bladellm_register_service_test && \
                make bladellm_server_test
            " > /dev/null 2>&1
    )

    local exit_code=$?

    if [[ $exit_code -ne 0 ]]; then
        echo "[-] Tests failed with exit code $exit_code. Removing image: $image"
        docker rmi "$image" || { echo "Failed to remove image: $image"; exit 1; }
        exit $exit_code
    else
        echo "[+] Tests passed successfully."
    fi

}

# ============ Main ============

if [ $# -ne 2 ]; then
    usage
fi

RELEASE_IMAGE="$1"
ENGINE="$2"

validate_engine "$ENGINE"

# Step 1: Build wheel package
build_wheel "$RELEASE_IMAGE"

# Step 2: Get Git commit ID
get_commit_id_and_time

# Step 3: Build final image
RELEASE_IMAGE_BASE_NAME=$(basename "${RELEASE_IMAGE//:/_}")
NEW_IMAGE="registry.cn-beijing.aliyuncs.com/llumnix/llumnix-release:${VERSION}_${COMMIT_ID}_${BUILD_TIME}_${RELEASE_IMAGE_BASE_NAME}"
build_final_image "$NEW_IMAGE" "$RELEASE_IMAGE" "$ENGINE"

# # Step 4: Run end-to-end tests
# if [[ "${ENGINE}" == "bladellm" ]]; then
#     run_bladellm_tests "$NEW_IMAGE"
# else
#     echo "[*] Skipping end-to-end tests."
# fi

echo "[+] Build and test release image completed successfully."
