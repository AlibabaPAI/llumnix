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

from abc import ABC, abstractmethod
from typing import List
import queue

import torch

class MigrationBackendBase(ABC):
    @abstractmethod
    def init_backend(self, group_name, world_size, rank) -> bool:
        raise NotImplementedError

    @abstractmethod
    def destory_backend(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def warmup(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def migrate_cache(self, src_handle, src_blocks: List[int], dst_blocks: List[int]) -> None:
        raise NotImplementedError

    @abstractmethod
    def do_send(self, dst_handle, blocks: List[int]):
        raise NotImplementedError

    @abstractmethod
    def do_recv(self, src_handle, blocks: List[int]):
        raise NotImplementedError

class BufferMigrationBackend(MigrationBackendBase):
    def __init__(self, num_buffer, buffer_shape, buffer_dtype, buffer_device, pin_memory, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_buffer = num_buffer

        self.dummy_buffer = [
            torch.empty(size=buffer_shape, dtype=buffer_dtype, device=buffer_device, pin_memory=pin_memory)
            for _ in range(self.num_buffer)
        ]

        self.avaiable_buffer_queue = queue.Queue()
        for i in range(self.num_buffer):
            self.avaiable_buffer_queue.put_nowait(i)

    def get_available_cache(self):
        return self.avaiable_buffer_queue.get()

    def put_back_cache(self, buffer_id):
        self.avaiable_buffer_queue.put_nowait(buffer_id)
