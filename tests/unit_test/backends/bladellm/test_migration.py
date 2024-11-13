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
import asyncio
import math
import pytest
import ray

from vllm import EngineArgs, SamplingParams
from vllm.utils import random_uuid

from llumnix.backends.vllm.llm_engine import BackendVLLM
from llumnix.llumlet.llumlet import Llumlet
from llumnix.backends.utils import BackendType
from llumnix.internal_config import MigrationConfig
from llumnix.llumlet.request import LlumnixRequest, RequestInferenceType
from llumnix.queue.queue_type import QueueType

from tests.unit_test.queue.utils import request_output_queue_server
# pylint: disable=unused-import
from tests.conftest import setup_ray_env

from .test_llm_engine import MockEngine
from .utils import create_dummy_prompt

TEST_PROMPTS = [
    "hello world, ",
    "Briefly describe the major milestones in the development of artificial intelligence from 1950 to 2020.\n",
    "Write a short story about a robot that dreams for the first time.\n",
    "Explain the cultural significance of the Mona Lisa painting, and how its perception might vary in Western versus Eastern societies.\n",
    "Swahili: 'The early bird catches the worm.'\n"
]
