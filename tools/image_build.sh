#!/bin/bash

# 1. install LLM engine env
# 2. install llumnix env
# 3. install bazel
# 4. install pygloo
# 5. install pylint pytest
# 6. apt-get update
# 7. apt-get install bc
# 8. build docker llumnix-dev or llumnix

if [ -z "$1" ]; then
    echo "Usage: $0 <NAMESPACE>"
    exit 1
fi

NAMESPACE="$1"
REPO="registry.cn-beijing.aliyuncs.com/llumnix/${NAMESPACE}"

DATE=$(date +%Y%m%d)
BRANCH=$(git rev-parse --abbrev-ref HEAD)
COMMIT_ID=$(git rev-parse --short=7 HEAD)
TAG="${DATE}_${BRANCH}_${COMMIT_ID}"

# Get the Git user email
USER_EMAIL=$(git config user.email)

if [ -z "$USER_EMAIL" ] || [ "${#USER_EMAIL}" -le 0 ]; then
    echo "Error: Git user email is not set or empty. Please set it using 'git config user.email'"
    exit 1
fi

# Ask for the container ID
echo "Please enter the container ID or name you want to commit:"
read CONTAINER_ID

if ! docker inspect -f '{{.ID}}' "$CONTAINER_ID" &> /dev/null; then
    echo "Error: The container '$CONTAINER_ID' does not exist."
    exit 1
fi

# Display the details about the commit
echo "Preparing to commit the following container:"
echo "Container ID: $CONTAINER_ID"
echo "Image TAG: ${REPO}:${TAG}"
echo "Using Git user email: $USER_EMAIL"

# Confirm the commit action
read -p "Do you want to proceed with the commit? (y/n): " -n 1 -r
echo    # move to a new line
if [[ $REPLY != "y" ]]; then
    echo "Commit aborted."
    exit 1
fi

# Commit the container with an optional message and author
docker commit -a "${USER_EMAIL}" "${CONTAINER_ID}" "${REPO}:${TAG}"

if [ $? -eq 0 ]; then
    echo "Image committed successfully: ${REPO}:${TAG}"
else
    echo "Image commit failed!"
    exit 1
fi
