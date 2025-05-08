from dataclasses import asdict
from unittest.mock import patch
from vllm import EngineArgs

from llumnix.arg_utils import LlumnixEngineArgsFactory, LlumnixEngineArgs
from llumnix.entrypoints.vllm.arg_utils import VllmEngineArgs
from llumnix.instance_info import InstanceType
from llumnix.internal_config import PDDConfig


# pylint: disable=unused-argument
def mocked_load_engine_args(engine_type: str, load_path: str) -> LlumnixEngineArgs:
    return VllmEngineArgs(
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
    llumnix_engine_args_factory = LlumnixEngineArgsFactory(
        enable_port_increment=False,
        load_registered_service=False,
        load_registered_service_path="",
        pdd_config=None,
    )
    engine_args = VllmEngineArgs(
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
        engine_args, InstanceType.NO_CONSTRAINTS.value
    )
    assert next_engine_args is not engine_args
    assert asdict(next_engine_args.engine_args) == asdict(engine_args.engine_args)


def test_gen_next_engine_args_vllm_from_registered_service():
    with patch("llumnix.arg_utils.load_engine_args", new=mocked_load_engine_args):
        llumnix_engine_args_factory = LlumnixEngineArgsFactory(
            enable_port_increment=True,
            load_registered_service=True,
            load_registered_service_path="",
            pdd_config=PDDConfig(True, False, [1, 2], False),
        )

    engine_args = None
    next_engine_args = llumnix_engine_args_factory.gen_next_engine_args(
        engine_args, InstanceType.PREFILL.value
    )
    assert next_engine_args is not engine_args
    assert asdict(next_engine_args.engine_args) == asdict(
        mocked_load_engine_args(InstanceType.PREFILL.value, "").engine_args
    )

    next_engine_args = llumnix_engine_args_factory.gen_next_engine_args(
        engine_args, InstanceType.DECODE.value
    )
    assert next_engine_args is not engine_args
    assert asdict(next_engine_args.engine_args) == asdict(
        mocked_load_engine_args(InstanceType.DECODE.value, "").engine_args
    )
