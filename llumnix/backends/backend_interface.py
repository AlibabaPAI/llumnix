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
from enum import Enum
from typing import Any, Iterable, List, Optional, Union

from llumnix.llumlet.migrating_request import MigratingRequest
from llumnix.server_info import ServerInfo


class BackendType(str, Enum):
    VLLM = "VLLM"
    SIM_VLLM = "SIM_VLLM"

    @staticmethod
    def is_sim_backend(status: "BackendType") -> bool:
        return status in [
            BackendType.SIM_VLLM,
        ]

class BackendInferenceType(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"

class BackendInterface(ABC):
    # Methods for inference
    @abstractmethod
    def add_request(self, request_id: str, server_info: ServerInfo,
                    *args, **kwargs) -> None:
        """Adds a new inference request to the backend's processing queue.

        This method should capture all necessary metadata of an inference request
        such as request ID, arrival time, and any other pertinent information.

        Args:
            request_id: Request ID.
            server_info: The information of the api server where the request come.
            *args: Positional arguments that represent request-specific data.
            **kwargs: Keyword arguments that contain metadata of the backend request
                      (request_id, arrival_time, etc.).
        """
        raise NotImplementedError

    @abstractmethod
    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        """Aborts one or more requests currently being processed.

        Args:
            request_id: A single request ID or an iterable of request IDs to abort.
        """
        raise NotImplementedError

    @abstractmethod
    def _start_engine_loop(self) -> None:
        """Start step loop of backend engine.
        """
        raise NotImplementedError

    # Methods for migration
    @abstractmethod
    def get_request_incremental_blocks(self, backend_request: Any, pre_stage_num_blocks: int) -> List[int]:
        """Retrieves the incremental block table for a given request.

        This method is used to fetch a list of block numbers that represent the incremental
        data associated with a particular backend request. It is typically called during a
        migration process where only transfer incremental blocks since last migration stage.

        Args:
            backend_request: An object representing the backend request. The type of this
                             object is dependent on the backend implementation and the details
                             of the request.
            pre_stage_num_blocks: An integer representing the number of blocks that were
                                   transferred in the previous migration stages.
                                   This is used to determine the incremental blocks that
                                   need to be fetched in the current stage.

        Returns:
            A list of integers, where each integer represents a block number that indicates
            physical index of kv cache block tensor. These block numbers can then be used
            to transfer to dstination instance.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_running_request(self, request_id: str) -> None:
        """
        Removes a request from the backend's running queue.

        This method is responsible for safely halting and removing an active request from the running
        queue of the backend engine. This action is performed in last stage migration when
        a request should be suspend.

        Args:
            request_id: A string identifier for the request that is to be removed from the running
                        queue. This ID uniquely identifies the request within the backend system.
        """
        raise NotImplementedError

    @abstractmethod
    def add_migrating_out_request_last_stage(self, backend_request: Any) -> None:
        """
        Adds a backend request to the list of migrating out request in last stage.

        This method adds a backend request to the list of migrating out request in last stage. This action is performed after
        the migrating out request has been removed from the running queue.

        Args:
            backend_request: An object representing the backend request. The type of this
                             object is dependent on the backend implementation and the details
                             of the request.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_migrating_out_request_last_stage(self, backend_request: Any) -> None:
        """
        Removes a backend request from the list of migrating out request in last stage.

        This method adds a backend request to the list of migrating out request in last stage. This action is performed after
        the migration is finished successfully.

        Args:
            backend_request: An object representing the backend request. The type of this
                             object is dependent on the backend implementation and the details
                             of the request.
        """
        raise NotImplementedError

    @abstractmethod
    def pop_migrating_out_requests_last_stage(self) -> List[Any]:
        """
        Pops the list of migrating out request in last stage.

        This method pops the list of migrating out request in last stage. This action is performed to free migrating out requests
        in last stage when the migration encounters exception.

        Returns:
            The list of migrating out request in last stage.
        """
        raise NotImplementedError

    @abstractmethod
    def pre_alloc(self, request_id: str, block_num: int) -> List[int]:
        """Pre-allocates cache blocks for a migrating request.

        This method selects a specified number of free cache blocks to be reserved for an incoming
        migration request identified by the given request ID. It updates the pre-allocation cache
        dictionary with the allocated blocks, which ensures that these blocks are not used by
        another process until the migration is finished.

        Args:
            request_id: The unique identifier of the migration request for which cache blocks
                        are to be pre-allocated.
            block_num: The number of cache blocks that need to be pre-allocated for the request.

        Returns:
            A list of integers where each integer represents the block table reserved for the migration request.
        """
        raise NotImplementedError

    @abstractmethod
    def should_abort_migration(self, backend_request: Any, last_stage_time: int) -> bool:
        """
        Determines whether the migration for a specific backend request should be aborted.

        This method evaluates the conditions under which a migration process, associated with a
        backend request, should be terminated before completion. If the backend request is no longer valid or a preemption
        event has occurred since the last migration stage, the migration need to be aborted.

        Args:
            backend_request: An object representing the backend request. The type of this
                             object is dependent on the backend implementation and the details
                             of the request.
            last_stage_time: An integer timestamp representing the last successful stage of the
                             migration process. This is used to determine if any significant event
                             has occurred after this point that would warrant the abortion of the
                             migration.

        Returns:
            True if the migration should be aborted based on the evaluation of the backend_request
            and last_stage_time; False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def add_running_request(self, backend_request: Any) -> None:
        """
        Adds a backend request to the running queue for processing.

        This method enqueues a backend request into engine running queue, marking it for
        active processing. It is used when a suspend migrating request should be added back
        to running queue.

        Args:
            backend_request: An object representing the backend request. The type of this
                             object is dependent on the backend implementation and the details
                             of the request.
        """
        raise NotImplementedError

    @abstractmethod
    def is_request_running(self, backend_request: Any) -> bool:
        """Checks if a given backend request is currently in the running queue.

        This method determines whether a backend request is present and actively being processed
        in the running queue.

        Args:
            backend_request: An object representing the backend request. The type of this
                             object is dependent on the backend implementation and the details
                             of the request.

        Returns:
            True if the backend request is currently in the running queue; False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def free_dst_pre_alloc_cache(self, request_id: str = None) -> None:
        """Frees pre-allocated blocks for a migrating request on the destination instance.

        This method is responsible for releasing any cache blocks or other resources that were
        pre-allocated on the destination instance for a migrating request. This is typically called
        after the migration aborted to ensure that resources are not left reserved unnecessarily.

        Args:
            request_id: A string representing the unique identifier of the request for which resources
                        are to be freed on the destination instance.
        """
        raise NotImplementedError

    @abstractmethod
    def free_src_request(self, backend_request: Any) -> None:
        """Frees blocks associated with a migrating request on the source instance.

        Upon completion or cancellation of a migration process, this method is invoked to clean up and
        release any reserved resources such as block tables and other metadata on the source instance.

        Args:
            backend_request: An object representing the backend request. The type of this
                             object is dependent on the backend implementation and the details
                             of the request.
        """
        raise NotImplementedError

    @abstractmethod
    def send_blocks(self, dst_ray_actor: "ray.actor.ActorHandle", src_blocks: List[int], dst_blocks: List[int]):
        """
        Sends cache blocks from the source instance to the destination instance.

        This method orchestrates the physical transfer of cache blocks between instances by Ray. It is responsible
        for ensuring that the specified blocks from the source instance's cache are sent to and properly received by
        the destination instance, where they are mapped according to the destination block table.

        Args:
            dst_ray_actor: A handle to the Ray actor representing the destination instance where the cache
                           blocks are to be sent. This handle is used to reference the destination's
                           execution context and manage the block transfer.
            src_blocks: A list of integers representing the block indexs in the source instance's
                             cache that need to be sent to the destination.
            dst_blocks: A list of integers representing the block indexs in the destination instance's
                             cache where the incoming blocks should be stored.
        """
        raise NotImplementedError

    @abstractmethod
    def commit_dst_request(self, backend_request: Any, server_info: ServerInfo) -> None:
        """Commits the migrating request to the destination instance.

        This method finalizes the migration process by transferring all necessary metadata and resource
        information (such as the block table) to the destination instance's engine. Upon completion,
        the destination instance should have all the required data to resume and handle the request as if it originated there natively.

        Args:
            backend_request: An object representing the backend request. The type of this
                             object is dependent on the backend implementation and the details
                             of the request.
            server_info: The information of the api server where the request come.
        """
        raise NotImplementedError

    @abstractmethod
    def get_last_running_request(self) -> Optional[MigratingRequest]:
        """Retrieves the last non-prefilling request from the running queue.

        This method iterates over the running queue in reverse order and returns the last request
        that has moved past the prefilling stage. Prefilling requests are not considered for
        migration, as they have not yet begun processing and therefore do not have any state that
        needs to be preserved.

        Returns:
            An instance of MigratingRequest representing the last request in the running queue that
            is not prefilling, or None if there are no such requests in the queue.
        """
        raise NotImplementedError

    @abstractmethod
    def get_longest_running_request(self) -> Optional[MigratingRequest]:
        """Retrieves the request with the longest sequence length from the running queue.

        This method should sorts the running queue based on length of the requests and
        returns the non-prefilling longest one.

        Returns:
            An instance of MigratingRequest representing the longest-running request in the queue that
            has generated output, or None if no such request exists.
        """
        raise NotImplementedError

    @abstractmethod
    def get_shortest_running_request(self) -> Optional[MigratingRequest]:
        """Retrieves the request with the shortest sequence length from the running queue.

        This method should sorts the running queue based on length of the requests and
        returns the non-prefilling shortest one.

        Returns:
            An instance of MigratingRequest representing the shortest-running request in the queue that
            has generated output, or None if no such request exists.
        """
        raise NotImplementedError

    @abstractmethod
    def get_request_server_info(self, request_id: str) -> ServerInfo:
        """Retrieves the information of the api server where the request come.

        This method is used by the migration coordinator to get the information of server where the migrating request come.

        Args:
            request_id: Request ID.
        Returns:
            The request output queue of the api server where the request come.
        """
        raise NotImplementedError

    @abstractmethod
    def get_all_request_ids(self) -> List[str]:
        """Retrieves all requests in instance.

        This method is used by the manager to get all requests in instance when it restarts.

        Returns:
            The list of request ID.
        """
        raise NotImplementedError

    @abstractmethod
    def free_request_states(self, request_id: Union[str, Iterable[str]]) -> None:
        """Free request states recorded in backend engine.

        This method is used by the llumlet or backend engine to free the request states when the request is finished/migrated/aborted.

        Args:
            request_id: Single/List of request ID.
        """
        raise NotImplementedError
