import os
import pytest

from vllm import EngineArgs

from llumnix.utils import (save_engine_args, load_engine_args, _get_engine_args_filename,
                           _get_engine_args_filepath)

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
