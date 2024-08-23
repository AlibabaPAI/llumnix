# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

set -e

BAZEL_CMD="bazel"
PYTHON="python"
PYGLOO_DIR="third_party/pygloo"

check_bazel() {
    if ! command -v $BAZEL_CMD &> /dev/null; then
        echo "Error: Bazel is not installed. Please install Bazel >= 5.1.0."
        exit 1
    fi
}

pygloo_install() {
    cd $PYGLOO_DIR && PYTHONWARNINGS="ignore" $PYTHON setup.py install
}

echo "Starting pygloo installation..."
check_bazel
pygloo_install
echo "pygloo installation completed successfully."
