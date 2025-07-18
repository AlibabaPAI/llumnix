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

from dataclasses import asdict
from unittest.mock import patch
from vllm import EngineArgs

from llumnix.arg_utils import LlumnixEngineArgs, InstanceArgs
from llumnix.entrypoints.vllm.arg_utils import VLLMEngineArgs, VLLMEngineArgsFactory
from llumnix.instance_info import InstanceType
from llumnix.internal_config import PDDConfig
from llumnix.utils import BackendType


# pylint: disable=unused-argument
def mocked_load_engine_args(engine_type: str, load_path: str) -> LlumnixEngineArgs:
    return VLLMEngineArgs(
        engine_args=EngineArgs(
            model=engine_type,
            download_dir="/mnt/model",
            worker_use_ray=True,
            tensor_parallel_size=1,
            enforce_eager=True,
            disable_async_output_proc=True,
        )
    )


def test_gen_next_engine_args_vllm():
    llumnix_engine_args_factory = VLLMEngineArgsFactory(
        enable_port_increment=False,
        load_registered_service=False,
        load_registered_service_path="",
        pdd_config=None,
    )
    engine_args = VLLMEngineArgs(
        engine_args=EngineArgs(
            model="facebook/opt-125m",
            download_dir="/mnt/model",
            worker_use_ray=True,
            tensor_parallel_size=1,
            enforce_eager=True,
            disable_async_output_proc=True,
        )
    )
    next_engine_args = llumnix_engine_args_factory.gen_next_engine_args(
        engine_args, InstanceArgs(instance_type=InstanceType.NEUTRAL), 0
    )
    assert next_engine_args is not engine_args
    assert asdict(next_engine_args.engine_args) == asdict(engine_args.engine_args)


def test_gen_next_engine_args_vllm_from_registered_service():
    with patch("llumnix.arg_utils.load_engine_args", new=mocked_load_engine_args):
        llumnix_engine_args_factory = VLLMEngineArgsFactory(
            enable_port_increment=True,
            load_registered_service=True,
            load_registered_service_path="",
            pdd_config=PDDConfig(True, False, False, [1, 2], False),
        )

    engine_args = None
    next_engine_args = llumnix_engine_args_factory.gen_next_engine_args(
        engine_args, InstanceArgs(instance_type=InstanceType.PREFILL), 0
    )
    assert next_engine_args is not engine_args
    assert asdict(next_engine_args.engine_args) == asdict(
        mocked_load_engine_args(InstanceType.PREFILL.value, "").engine_args
    )

    next_engine_args = llumnix_engine_args_factory.gen_next_engine_args(
        engine_args, InstanceArgs(instance_type=InstanceType.DECODE), 0
    )
    assert next_engine_args is not engine_args
    assert asdict(next_engine_args.engine_args) == asdict(
        mocked_load_engine_args(InstanceType.DECODE.value, "").engine_args
    )
