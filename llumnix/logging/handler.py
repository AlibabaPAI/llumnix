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

import logging
import os
import ray


class NodeFileHandler(logging.Handler):
    def __init__(self, base_path):
        super().__init__()
        self.base_path = base_path
        self.ensure_base_path_exists()

    def ensure_base_path_exists(self):
        if not os.path.exists(self.base_path):
            try:
                os.makedirs(self.base_path)
                print(f"Created log node path: {self.base_path}")
            except OSError as e:
                print(f"Error creating log node path {self.base_path}: {e}")

    def emit(self, record):
        node_id = ray.get_runtime_context().get_node_id()
        filename = os.path.join(self.base_path, f"{node_id}.log")
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(self.format(record) + '\n')
