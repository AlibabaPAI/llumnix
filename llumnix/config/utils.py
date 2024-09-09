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

from typing import Dict, Union
import argparse

from .config import LlumnixConfig
from .default import _C

def get_llumnix_config(cfg_filename: str = "", others: Union[Dict, argparse.Namespace] = None) -> LlumnixConfig:
    if isinstance(others, argparse.Namespace):
        others = vars(others)

    cfg: LlumnixConfig = _C.clone()

    if len(cfg_filename) > 0:
        cfg.merge_from_file(cfg_filename)

    def update_config(config, args):
        for key, value in config.items():
            if isinstance(value, dict):
                update_config(value, args)
            elif key.lower() in args and args[key.lower()] is not None:
                config[key] = args[key.lower()]

    if others is not None:
        update_config(cfg, others)

    cfg.freeze()
    return cfg
