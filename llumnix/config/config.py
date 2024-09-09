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
from typing import List
import yaml
from yacs.config import CfgNode

BASE_KEY = "_BASE_"

class LlumnixConfig(CfgNode):
    @staticmethod
    def load_yaml_with_base(filename: str):
        with open(filename, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        def merge_a_into_b(a, b):
            for k, v in a.items():
                if isinstance(v, dict) and k in b:
                    assert isinstance(b[k], dict), "Cannot inherit key '{}' from base!".format(k)
                    merge_a_into_b(v, b[k])
                else:
                    b[k] = v

        if BASE_KEY in cfg:
            base_cfg_file = cfg[BASE_KEY]

            if base_cfg_file.startswith("~"):
                base_cfg_file = os.path.expanduser(base_cfg_file)
            if not any(map(base_cfg_file.startswith, ["/", "https://", "http://"])):
                base_cfg_file = os.path.join(os.path.dirname(filename), base_cfg_file)
            base_cfg = CfgNode.load_yaml_with_base(base_cfg_file)

            del cfg[BASE_KEY]
            merge_a_into_b(cfg, base_cfg)
            return base_cfg

        return cfg

    def merge_from_file(self, cfg_filename: str):
        loaded_cfg = LlumnixConfig.load_yaml_with_base(cfg_filename)
        loaded_cfg = type(self)(loaded_cfg)
        self.merge_from_other_cfg(loaded_cfg)

    def merge_from_list(self, cfg_list: List):
        keys = set(cfg_list[0::2])
        assert (BASE_KEY not in keys), "The reserved key '{}' can only be used in files!".format(BASE_KEY)
        return super().merge_from_list(cfg_list)
