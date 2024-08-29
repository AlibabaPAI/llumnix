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

import ray
import pytest


@pytest.fixture
def setup_ray_env():
    ray.init(namespace="llumnix", ignore_reinit_error=True)
    yield
    named_actors = ray.util.list_named_actors(True)
    for actor in named_actors:
        actor_handle = ray.get_actor(actor['name'], namespace='llumnix')
        try:
            ray.kill(actor_handle)
        except:
            pass
    ray.shutdown()
