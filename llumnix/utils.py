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

import uuid
import ray

PLACEMENT_GROUP_NAME_PREFIX = "pg_"
SERVER_NAME_PREFIX = "server_"
INSTANCE_NAME_PREFIX = "instance_"


def random_uuid() -> str:
    return str(uuid.uuid4().hex)

def convert_bytes(bytes_size):
    """Convert bytes to KB, MB, GB, etc."""
    if bytes_size < 0:
        raise ValueError("Size must be a non-negative integer.")

    size_suffixes = ['B', 'KB', 'MB', 'GB', 'TB']
    index = 0

    while bytes_size >= 1024 and index < len(size_suffixes) - 1:
        bytes_size /= 1024.0
        index += 1

    return f"{bytes_size:.2f} {size_suffixes[index]}"

def clear_gloo_backend_state():
    try:
        # clear gloo migrate backend intermediate state
        ray.kill(ray.get_actor("gloo_queue", "llumnix"))
    except ValueError:
        # gloo_queue may not have been created yet; just ignore this error.
        pass

def get_placement_group_name(instance_id: str) -> str:
    return f"{PLACEMENT_GROUP_NAME_PREFIX}{instance_id}"

def get_instance_name(instance_id: str) -> str:
    return f"{INSTANCE_NAME_PREFIX}{instance_id}"

def remove_placement_group(instance_id: str) -> bool:
    try:
        placement_group = ray.util.get_placement_group(get_placement_group_name(instance_id))
        if not placement_group:
            return False
        # asynchronous api
        ray.util.remove_placement_group(placement_group)
    # pylint: disable=broad-except
    except Exception:
        return False
    return True
