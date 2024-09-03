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

from typing import List

from .llumnix_config import LlumnixConfig
from .default import _C

def get_llumnix_config(cfg_filename: str, cfg_list: List) -> LlumnixConfig:
    cfg: LlumnixConfig = _C.clone()
    if len(cfg_filename) > 0:
        cfg.merge_from_file(cfg_filename)
    if len(cfg_list) > 0:
        cfg.merge_from_list(cfg_list)
    cfg.freeze()
    return cfg
