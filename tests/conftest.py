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

from datetime import datetime
import time
import shutil
import os
import subprocess

import ray
from ray.util.placement_group import PlacementGroup
from ray.util import list_named_actors, placement_group_table, remove_placement_group
from ray._raylet import PlacementGroupID
from ray._private.utils import hex_to_binary
import pytest

from llumnix.utils import random_uuid
from llumnix.utils import get_ip_address

from tests.e2e_test.utils import shutdown_llumnix_service_func


def ray_start():
    for _ in range(5):
        ray_stop()
        subprocess.run(["ray", "start", "--head", "--port=6379"], check=False, stdout=subprocess.DEVNULL)
        time.sleep(5.0)
        result = subprocess.run(["ray", "status"], check=False, capture_output=True, text=True)
        if result.returncode == 0:
            return
        print("Ray start failed, exception: {}.".format(result.stderr.strip()))
        time.sleep(3.0)
    raise Exception("Ray start failed after 5 attempts.")

def ray_stop(max_retries=5, delay=5):
    def is_ray_running():
        result = subprocess.run(["ps", "-ef"], stdout=subprocess.PIPE, text=True, check=False)
        lines = [line for line in result.stdout.splitlines() if
                 'ray' in line and 'rayqueue' not in line and 'rayrpc' not in line
                 and 'use_ray_spmd_worker' not in line]
        if len(lines) > 0:
            print("Ray processes are still running: {}".format('\n'.join(lines)))
        return len(lines) > 0

    attempt = 0
    while attempt <= max_retries:
        subprocess.run(["ray", "stop", "--force"], check=False, stdout=subprocess.DEVNULL)
        if not is_ray_running():
            print("Ray has been successfully stopped.")
            return

        attempt += 1
        print("Ray processes still running. Retry after {} second(s)...".format(delay))
        time.sleep(delay)

    print("Failed to stop Ray processes after maximum retries {}.".format(max_retries))

def cleanup_ray_env_func():
    actor_infos = list_named_actors(True)
    for actor_info in actor_infos:
        try:
            actor_handle = ray.get_actor(actor_info['name'], namespace=actor_info['namespace'])
            ray.kill(actor_handle)
        # pylint: disable=bare-except
        except:
            pass

    time.sleep(1.0)

    pg_table = placement_group_table()
    for placement_group_id in pg_table:
        try:
            pg = PlacementGroup(PlacementGroupID(hex_to_binary(placement_group_id)))
            remove_placement_group(pg)
        # pylint: disable=bare-except
        except:
            pass

    time.sleep(1.0)

    try:
        ray.shutdown()
    # pylint: disable=broad-except
    except Exception as e:
        print("Ray shutdown error: {}".format(e))

    time.sleep(3.0)

def pytest_sessionstart(session):
    ray_start()

def pytest_sessionfinish(session):
    cleanup_ray_env_func()
    ray_stop()
    shutdown_llumnix_service_func()


SKIP_REASON: str = None

@pytest.fixture
def ray_env():
    global SKIP_REASON
    try:
        ray.init(ignore_reinit_error=True, namespace="llumnix")
        SKIP_REASON = None
        yield
    finally:
        if SKIP_REASON is None or len(SKIP_REASON) == 0:
            cleanup_ray_env_func()

def backup_error_log(func_name):
    curr_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    dst_dir = os.path.expanduser(f'/mnt/error_log/{curr_time}_{random_uuid()}')
    os.makedirs(dst_dir, exist_ok=True)

    src_dir = os.getcwd()

    for filename in os.listdir(src_dir):
        if filename.startswith("instance_"):
            src_file = os.path.join(src_dir, filename)
            shutil.copy(src_file, dst_dir)

        elif filename.startswith("bench_"):
            src_file = os.path.join(src_dir, filename)
            shutil.copy(src_file, dst_dir)

        elif filename.startswith("nohup"):
            src_file = os.path.join(src_dir, filename)
            shutil.copy(src_file, dst_dir)

    file_path = os.path.join(dst_dir, 'test.info')
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(f'{func_name}')

    print(f"Backup error instance log to directory {dst_dir}, host: {get_ip_address()}")

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    if report.when == "call" and report.failed:
        func_name = item.name
        backup_error_log(func_name)
