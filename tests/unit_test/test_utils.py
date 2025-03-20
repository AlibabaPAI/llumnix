from vllm import EngineArgs

from llumnix.utils import put_data_to_ray_internal_kv, get_engine_args_data_name, get_data_from_ray_internal_kv

# pylint: disable=unused-import
from tests.conftest import ray_env


def test_put_data_to_ray_internal_kv_and_get_data_from_ray_internal_kv(ray_env):
    engine_args = EngineArgs()
    instance_type = "no_constraints"
    put_data_to_ray_internal_kv(get_engine_args_data_name(instance_type), engine_args)
    engine_args_get = get_data_from_ray_internal_kv(get_engine_args_data_name(instance_type))
    assert engine_args_get == engine_args
