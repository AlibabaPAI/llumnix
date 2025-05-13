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

from typing import Dict, Union, List
import argparse

from llumnix.config import LlumnixConfig
from llumnix.config.default import _C

# The order of args setting: default config (default.py) -> args config (--config-file) -> command line args
# Therefore, command line args have the highest priority, config file is the second, default config args is the third.
def get_llumnix_config(cfg_filename: str = "", args: Union[Dict, argparse.Namespace] = None, opts: List = None) -> LlumnixConfig:
    if isinstance(args, argparse.Namespace):
        args = vars(args)

    cfg: LlumnixConfig = _C.clone()

    if cfg_filename is not None and len(cfg_filename) > 0:
        cfg.merge_from_file(cfg_filename)

    def update_config(config, args):
        for key, value in config.items():
            if isinstance(value, dict):
                update_config(value, args)
            elif key.lower() in args and args[key.lower()] is not None:
                config[key] = args[key.lower()]

    if args is not None:
        update_config(cfg, args)

    if opts is not None:
        cfg.merge_from_list(opts)

    cfg.freeze()
    return cfg
