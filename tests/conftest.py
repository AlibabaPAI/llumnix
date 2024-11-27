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

import subprocess
import ray
import pytest

def pytest_sessionstart(session):
    subprocess.run(["ray", "stop"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["ray", "start", "--head", "--disable-usage-stats", "--port=6379"], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def pytest_sessionfinish(session, exitstatus):
    subprocess.run(["ray", "stop"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

@pytest.fixture
def setup_ray_env():
    ray.init(namespace="llumnix", ignore_reinit_error=True)
    yield
    try:
        named_actors = ray.util.list_named_actors(True)
        for actor in named_actors:
            try:
                actor_handle = ray.get_actor(actor['name'], namespace=actor['namespace'])
            # pylint: disable=bare-except
            except:
                continue

            try:
                ray.kill(actor_handle)
            # pylint: disable=bare-except
            except:
                continue
        # Should to be placed after killing actors, otherwise it may occur some unexpected errors when re-init ray.
        ray.shutdown()
    # pylint: disable=bare-except
    except:
        pass
