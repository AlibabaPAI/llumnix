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
import pytest

from vllm import EngineArgs

from llumnix.arg_utils import _get_engine_args_filename, load_engine_args, save_engine_args
from llumnix.arg_utils import (_get_engine_args_filepath)

# pylint: disable=unused-import
from tests.conftest import ray_env


@pytest.mark.parametrize("save_key", [None, "test"])
def test_save_engine_args_and_load_engine_args(ray_env, save_key):
    engine_args = EngineArgs()
    engine_type = "no_constraints"
    save_path = "."
    save_engine_args(engine_type, save_path, engine_args, save_key)
    final_save_path = _get_engine_args_filepath(save_path, save_key)
    engine_args_load = load_engine_args(engine_type, final_save_path)
    assert engine_args_load == engine_args
    engine_args_filename = _get_engine_args_filename(engine_type)
    save_filename = os.path.join(final_save_path, engine_args_filename)
    assert os.path.exists(save_filename)
    os.remove(save_filename)
