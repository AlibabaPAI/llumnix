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


def ray_start():
    for _ in range(5):
        subprocess.run(["ray", "stop"], check=False, stdout=subprocess.DEVNULL)
        subprocess.run(["ray", "start", "--head", "--port=6379"], check=False, stdout=subprocess.DEVNULL)
        time.sleep(5.0)
        result = subprocess.run(["ray", "status"], check=False, capture_output=True, text=True)
        if result.returncode == 0:
            return
        print("Ray start failed, exception: {}".format(result.stderr.strip()))
        time.sleep(3.0)
    raise Exception("Ray start failed after 5 attempts.")

def ray_stop():
    subprocess.run(["ray", "stop"], check=False, stdout=subprocess.DEVNULL)

def cleanup_ray_env_func():
    actor_infos = list_named_actors(True)
    for actor_info in actor_infos:
        try:
            actor_handle = ray.get_actor(actor_info['name'], namespace=actor_info['namespace'])
            ray.kill(actor_handle)
        # pylint: disable=broad-except
        except Exception as e:
            print("clear ray actor error: ", e)

    time.sleep(1.0)

    pg_table = placement_group_table()
    for placement_group_id in pg_table:
        try:
            pg = PlacementGroup(PlacementGroupID(hex_to_binary(placement_group_id)) )
            remove_placement_group(pg)
        # pylint: disable=broad-except
        except Exception as e:
            print("clear placement group error: ", e)

    time.sleep(1.0)

    try:
        ray.shutdown()
    # pylint: disable=broad-except
    except Exception as e:
        print("ray shutdown error: ", e)

    time.sleep(3.0)

def pytest_sessionstart(session):
    ray_start()

def pytest_sessionfinish(session):
    ray_stop()

@pytest.fixture
def ray_env():
    ray.init(ignore_reinit_error=True, namespace="llumnix")
    yield
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

    file_path = os.path.join(dst_dir, 'test.info')
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(f'{func_name}')

    print(f"Backup error instance log to directory {dst_dir}")

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    if report.when == "call" and report.failed:
        func_name = item.name
        backup_error_log(func_name)
