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

if ! command -v nvcc &> /dev/null
then
    echo "CUDA is not installed."
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep -oP 'V\d+\.\d+' | head -1 | tr -d 'V')

case $CUDA_VERSION in
    10.0)
        pip install cupy-cuda100
        ;;
    10.1)
        pip install cupy-cuda101
        ;;
    10.2)
        pip install cupy-cuda102
        ;;
    11.0)
        pip install cupy-cuda110
        ;;
    11.1)
        pip install cupy-cuda111
        ;;
    11.2 | 11.3 | 11.4)
        pip install cupy-cuda11x
        ;;
    12.*)
        pip install cupy-cuda12x
        ;;
    *)
        echo "Detected CUDA version: $CUDA_VERSION, please refer to https://pypi.org/search/?o=&q=cupy-cuda&page=1 and the CuPy documentation for supported versions."
        exit 1
        ;;
esac

echo "CuPy for CUDA $CUDA_VERSION installed successfully."
