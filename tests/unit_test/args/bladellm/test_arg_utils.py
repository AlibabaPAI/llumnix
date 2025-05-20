from dataclasses import asdict
from unittest.mock import patch
from blade_llm.service.args import ServingArgs, DisaggOptions
from blade_llm.utils.load_model_options import LoadModelOptions

from llumnix.arg_utils import LlumnixEngineArgsFactory, LlumnixEngineArgs
from llumnix.entrypoints.bladellm.arg_utils import BladellmEngineArgs
from llumnix.instance_info import InstanceType
from llumnix.internal_config import PDDConfig


# pylint: disable=unused-argument
def mocked_load_engine_args(engine_type: str, load_path: str) -> LlumnixEngineArgs:
    return BladellmEngineArgs(
        engine_args=ServingArgs(
            load_model_options=LoadModelOptions(model="./"),
            disagg_options=DisaggOptions(),
        )
    )


def test_gen_next_engine_args_baldellm():
    llumnix_engine_args_factory = LlumnixEngineArgsFactory(
        enable_port_increment=False,
        load_registered_service=False,
        load_registered_service_path="",
        pdd_config=PDDConfig(False, False, [1, 2], False),
    )
    serving_args = ServingArgs(
        load_model_options=LoadModelOptions(model="./"), disagg_options=DisaggOptions()
    )
    engine_args = BladellmEngineArgs(serving_args)
    next_engine_args = llumnix_engine_args_factory.gen_next_engine_args(
        engine_args, InstanceType.NO_CONSTRAINTS.value
    )
    assert next_engine_args is not engine_args
    assert asdict(next_engine_args.load_engine_args_if_needed()) == asdict(
        engine_args.load_engine_args_if_needed()
    )


def test_gen_next_engine_args_baldellm_from_registered_service():
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
    assert asdict(next_engine_args.load_engine_args_if_needed()) == asdict(
        mocked_load_engine_args(
            InstanceType.PREFILL.value, ""
        ).load_engine_args_if_needed()
    )

    next_engine_args = llumnix_engine_args_factory.gen_next_engine_args(
        engine_args, InstanceType.DECODE.value
    )
    assert next_engine_args is not engine_args
    assert asdict(next_engine_args.load_engine_args_if_needed()) == asdict(
        mocked_load_engine_args(
            InstanceType.DECODE.value, ""
        ).load_engine_args_if_needed()
    )


def test_gen_next_engine_args_baldellm_enable_port_increment():
    with patch("llumnix.arg_utils.load_engine_args", new=mocked_load_engine_args):
        llumnix_engine_args_factory = LlumnixEngineArgsFactory(
            enable_port_increment=True,
            load_registered_service=False,
            load_registered_service_path="",
            pdd_config=PDDConfig(False, False, [1, 2], False),
        )

    serving_args = ServingArgs(
        load_model_options=LoadModelOptions(model="./"), disagg_options=DisaggOptions()
    )
    engine_args = BladellmEngineArgs(serving_args)
    next_engine_args = llumnix_engine_args_factory.gen_next_engine_args(
        engine_args, InstanceType.PREFILL.value
    )
    next_engine_args2 = llumnix_engine_args_factory.gen_next_engine_args(
        engine_args, InstanceType.PREFILL.value
    )
    assert next_engine_args is not engine_args
    assert (
        next_engine_args.load_engine_args_if_needed().disagg_options.inst_role
        == "prefill"
    )
    assert (
        engine_args.load_engine_args_if_needed().disagg_options.token_port
        == next_engine_args.load_engine_args_if_needed().disagg_options.token_port
    )
    assert (
        engine_args.load_engine_args_if_needed().disagg_options.token_port + 10
        == next_engine_args2.load_engine_args_if_needed().disagg_options.token_port
    )


def test_gen_next_engine_args_baldellm_enable_pdd():
    with patch("llumnix.arg_utils.load_engine_args", new=mocked_load_engine_args):
        llumnix_engine_args_factory = LlumnixEngineArgsFactory(
            enable_port_increment=True,
            load_registered_service=False,
            load_registered_service_path="",
            pdd_config=PDDConfig(False, True, [1, 2], False),
        )

    serving_args = ServingArgs(
        load_model_options=LoadModelOptions(model="./"), disagg_options=DisaggOptions()
    )
    engine_args = BladellmEngineArgs(serving_args)
    next_engine_args = llumnix_engine_args_factory.gen_next_engine_args(
        engine_args, InstanceType.PREFILL.value
    )
    assert next_engine_args is not engine_args
    assert (
        next_engine_args.load_engine_args_if_needed().disagg_options.inst_role
        == "prefill"
    )

    next_engine_args = llumnix_engine_args_factory.gen_next_engine_args(
        engine_args, InstanceType.DECODE.value
    )
    assert next_engine_args is not engine_args
    assert (
        next_engine_args.load_engine_args_if_needed().disagg_options.inst_role
        == "decode"
    )
