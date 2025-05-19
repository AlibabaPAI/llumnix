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

import os

from llumnix import envs as llumnix_envs


def try_convert_to_local_path(data_path: str) -> str:
    if os.path.isabs(data_path):
        return data_path

    assert "/" in data_path
    base_data_name = os.path.basename(data_path)

    base_model_path: str = llumnix_envs.MODEL_PATH
    local_model_path: str = os.path.join(base_model_path, base_data_name)
    if os.path.exists(local_model_path):
        return local_model_path

    base_dataset_path: str = llumnix_envs.DATASET_PATH
    local_dataset_path: str = os.path.join(base_dataset_path, base_data_name)
    if os.path.exists(local_dataset_path):
        return local_dataset_path

    return data_path
