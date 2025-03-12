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
from typing import Iterable, List, Union, Deque, Tuple

from llumnix.llumlet.request import LlumnixRequest, RequestStatus
from llumnix.server_info import ServerInfo


class EngineState(str, Enum):
    INIT = "INIT"
    CRASHED = "CRASHED"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"


class BackendType(str, Enum):
    VLLM = "VLLM"
    BLADELLM = "BLADELLM"
    SIM_VLLM = "SIM_VLLM"

    @staticmethod
    def is_sim_backend(status: "BackendType") -> bool:
        return status in [BackendType.SIM_VLLM]


class BackendInterface(ABC):
    @abstractmethod
    def add_request(self, request_id: str, server_info: ServerInfo, expected_steps: int,
                    *args, **kwargs) -> None:
        """Add a new inference request to the backend.

        This method should capture all necessary metadata of an inference request
        such as request ID, arrival time, and any other related information.

        Args:
            request_id: Request ID.
            server_info: The information of the api server where the request comes.
            expected_steps: The expected number of steps for the request to run. The number of steps
                            represents the times 'engine.step()' has been called by the backend
                            instance for the request. Currently, `expected_steps` is used to
                            implement prefill-decoding disaggregation. For requests dispatched to
                            prefill instances, `expected_steps` is set to 1.
            *args: Positional arguments that represent request-specific data.
            **kwargs: Keyword arguments that contain metadata of the backend request
                      (request_id, arrival_time, etc.).
        """
        raise NotImplementedError

    @abstractmethod
    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        """Abort one or multiple requests of the backend.

        Args:
            request_id: A request ID or an iterable of request IDs to abort.
        """
        raise NotImplementedError

    # Methods for migration
    @abstractmethod
    def get_request_incremental_blocks(self, backend_request: LlumnixRequest, pre_stage_num_blocks: int) \
        -> Tuple[List[int], List[int]]:
        """Get the incremental blocks and token ids for a given request.

        This method is used to fetch a list of block numbers and a list of token ids that represent the
        incremental data associated with a particular backend request. It is typically called during a
        migration process where only transfer incremental blocks since previous migration stage.

        Args:
            backend_request: An object representing the backend request.
            pre_stage_num_blocks: An integer representing the number of blocks that were transferred in
                                  the previous migration stage. This is used to determine the incremental
                                  blocks that need to be fetched in the current stage.

        Returns:
            A list of block numbers and a list of token ids. A block number is the physical index of kv
            cache block tensor. A token id is the ID of a token in prompt or response. These block numbers
            are then to be used to transfer blocks to the destination instance.
        """
        raise NotImplementedError

    @abstractmethod
    def get_running_queue(self) -> Deque[LlumnixRequest]:
        """
        Return the running queue of backend.
        """
        raise NotImplementedError

    @abstractmethod
    def get_waiting_queue(self) -> Deque[LlumnixRequest]:
        """
        Return the waiting queue of backend.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_running_request(self, request_id: str) -> bool:
        """
        Remove a request from the running queue of backend.

        This method is responsible for safely stopping and removing an active request from the running
        queue of the backend engine. This action is performed in last stage migration when a request
        should be suspend.

        Args:
            request_id: The ID of the request that is to be removed from the running queue.

        Returns:
            True if the request was successfully removed from the running queue, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_waiting_request(self, request_id: str) -> bool:
        """
        Remove a request from the waiting queue of backend.

        This method is responsible for safely halting and removing an active request from the waiting
        queue of the backend engine. This action is performed in waiting request migration.

        Args:
            request_id: The ID of the request that is to be removed from the waiting queue.

        Returns:
            True if the request was successfully removed from the waiting queue, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def add_migrating_out_request_last_stage(self, dst_instance_id: str, backend_request: LlumnixRequest) -> None:
        """
        Add a backend request to the list of migrating out requests in last stage.

        This method adds a backend request to the list of migrating out requests in last stage.
        This action is performed after the migrating out request has been removed from the
        running queue.

        Args:
            dst_instance_id: The ID of destination instance.
            backend_request: An object representing the backend request.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_migrating_out_request_last_stage(self, dst_instance_id: str, backend_request: LlumnixRequest) -> None:
        """
        Remove a backend request from the list of migrating out requests in last stage.

        This method adds a backend request to the list of migrating out requests in last stage.
        This action is performed after the migration is finished successfully.

        Args:
            dst_instance_id: The ID of destination instance.
            backend_request: An object representing the backend request.
        """
        raise NotImplementedError

    @abstractmethod
    def pop_migrating_out_requests_last_stage(self, dst_instance_id: str) -> List[LlumnixRequest]:
        """
        Pop the list of migrating out requests in last stage.

        This method pops the list of migrating out requests in last stage. This action is performed
        to free migrating out requests in last stage when the migration encounters exception.

        Returns:
            dst_instance_id: The ID of destination instance.
            The list of migrating out requests in last stage.
        """
        raise NotImplementedError

    @abstractmethod
    def pre_alloc(self,
                  instance_id: str,
                  request_id: str,
                  request_status: RequestStatus,
                  request_arrival_time: float,
                  block_num: int,
                  token_ids: List[int]) -> List[int]:
        """Pre-allocate cache blocks for a migrating request.

        This method selects a specified number of free cache blocks to be reserved for an incoming
        migration request identified by the given request ID. It updates the pre-allocation cache
        dictionary with the allocated blocks, which ensures that these blocks are not used by
        another process until the migration is finished. For the waiting request, it only reserves
        free cache blocks when the request is the earliest arrival one among the requests of dst
        instance's waiting queue.

        Args:
            instance_id: The ID of the source instance.
            request_id: The ID of the migration request for which cache blocks are to be pre-allocated.
            request_status: The status (waiting/running) of the request.
            request_arrival_time: The arrival time of the request.
            block_num: The number of cache blocks that need to be pre-allocated for the request.
            token_ids: The token IDs of the request.
        Returns:
            A list of integers where each integer represents the block table reserved for the migration request.
        """
        raise NotImplementedError

    @abstractmethod
    def add_running_request(self, backend_request: LlumnixRequest) -> None:
        """
        Add a backend request to the running queue of backend.

        This method enqueues a backend request to the running queue of backend.
        It is used when a suspend migrating request should be added back to running queue.

        Args:
            backend_request: An object representing the backend request.
        """
        raise NotImplementedError

    @abstractmethod
    def add_waiting_request(self, backend_request: LlumnixRequest) -> None:
        """
        Add a backend request to the waiting queue of backend.

        This method enqueues a backend request into the waiting queue of backend.
        It is used when a suspend migrating request should be added back to waiting queue.

        Args:
            backend_request: An object representing the backend request.
        """
        raise NotImplementedError

    @abstractmethod
    def free_dst_pre_alloc_cache(self, instance_id: str, request_id: str = None) -> None:
        """Free pre-allocated blocks for a migrating request on the destination instance.

        This method is responsible for releasing any cache blocks and other resources that were
        pre-allocated on the destination instance for a migrating request. This is typically called
        after the migration aborted to ensure that resources are not left reserved unnecessarily.

        Args:
            instance_id: The ID of the source instance.
            request_id: The ID of the request for which resources are to be freed on the destination instance.
        """
        raise NotImplementedError

    @abstractmethod
    def free_src_request(self, backend_request: LlumnixRequest) -> None:
        """Free blocks associated with a migrating request on the source instance.

        Upon completion or cancellation of a migration process, this method is invoked to clean up and
        release any reserved resources such as block tables and other metadata on the source instance.

        Args:
            backend_request: An object representing the backend request.
        """
        raise NotImplementedError

    @abstractmethod
    async def send_blocks(self, dst_ray_actor: "ray.actor.ActorHandle", src_blocks: List[int], dst_blocks: List[int]) -> None:
        """
        Send cache blocks from the source instance to the destination instance.

        This method orchestrates the physical transfer of cache blocks between instances by Ray. It is responsible
        for ensuring that the specified blocks from the source instance's cache are sent to and properly received by
        the destination instance, where they are mapped according to the destination block table.

        Args:
            dst_ray_actor: A handle to the Ray actor representing the destination instance where the cache blocks are
                           to be sent. This handle is used to reference the destination's execution context and manage
                           the block transfer.
            src_blocks: A list of integers representing the block indexs in the source instance's cache that need to be
                        sent to the destination.
            dst_blocks: A list of integers representing the block indexs in the destination instance's cache where the
                        incoming blocks should be stored.
        """
        raise NotImplementedError

    @abstractmethod
    def commit_dst_request(self, instance_id: str, backend_request: LlumnixRequest) -> None:
        """Commit the migrating request to the destination instance.

        This method finalizes the migration process by transferring all necessary metadata and resource information
        (such as the block table) to the destination instance. Upon completion, the destination instance should have
        all the required data to resume and handle the request as if it originated there natively.

        Args:
            instance_id: The ID of the source instance.
            backend_request: An object representing the backend request.
        """
        raise NotImplementedError

    @abstractmethod
    def get_all_request_ids(self) -> List[str]:
        """Get all requests in instance.

        This method is used by the manager to get all requests in instance when it restarts.

        Returns:
            The list of request ID.
        """
        raise NotImplementedError
