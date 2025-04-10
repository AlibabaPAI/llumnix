from abc import ABC, abstractmethod
import threading

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from llumnix.arg_utils import EntrypointsArgs
from llumnix.utils import get_ip_address
from llumnix.entrypoints.utils import EntrypointsContext
from llumnix.llumlet.llumlet import Llumlet
from llumnix.queue.utils import init_request_output_queue_server, QueueType
from llumnix.logging.logger import init_logger
from llumnix.ray_utils import log_actor_ray_info

logger = init_logger(__name__)


class APIServerActor(ABC):
    def __init__(self, server_name: str, entrypoints_args: EntrypointsArgs, engine_args):
        log_actor_ray_info(actor_class_name=self.__class__.__name__)
        self.instance_id = server_name.split("_")[-1]
        logger.info("APIServerActor(instance_id={})".format(self.instance_id))
        self.entrypoints_args = entrypoints_args
        self.engine_args = engine_args
        if entrypoints_args.host in ("127.0.0.1", "0.0.0.0"):
            self.host = entrypoints_args.host
        else:
            self.host = get_ip_address()
        self.request_output_queue_port = self.entrypoints_args.request_output_queue_port
        self.request_output_queue_type = QueueType(self.entrypoints_args.request_output_queue_type)
        self.request_output_queue = init_request_output_queue_server(
                                        self.host, self.request_output_queue_port, self.request_output_queue_type)

    def __repr__(self):
        return f"{self.__class__.__name__}(iid={self.instance_id[:5]})"

    def _setup_entrypoints_context(self,
                                   manager: "ray.actor.ActorHandle",
                                   instance_id: str,
                                   instance: Llumlet):
        # avoid circular import
        # pylint: disable=import-outside-toplevel
        from llumnix.entrypoints.setup import setup_entrypoints_context
        self.entrypoints_context = setup_entrypoints_context(
                                        self.entrypoints_args, manager, [instance_id], [instance], self.request_output_queue)

    @abstractmethod
    def _run_server(self,
                    entrypoints_args: EntrypointsArgs,
                    engine_args,
                    entrypoints_context: EntrypointsContext):
        raise NotImplementedError

    def run(self,
            manager: "ray.actor.ActorHandle",
            instance_id: str,
            instance: Llumlet):
        self._setup_entrypoints_context(manager, instance_id, instance)
        self.run_server_thread = threading.Thread(
            target=self._run_server, args=(self.entrypoints_args, self.engine_args, self.entrypoints_context),
            daemon=True, name="run_server"
        )
        self.run_server_thread.start()

    @classmethod
    def from_args(cls,
                  num_gpus: int,
                  server_name: str,
                  placement_group: PlacementGroup,
                  entrypoints_args: EntrypointsArgs,
                  engine_args):
        try:
            api_server_class = ray.remote(num_cpus=1,
                                          num_gpus=num_gpus,
                                          name=server_name,
                                          namespace="llumnix",
                                          lifetime="detached")(cls).options(
                                                scheduling_strategy=PlacementGroupSchedulingStrategy(
                                                    placement_group=placement_group,
                                                    placement_group_bundle_index=0,
                                                    placement_group_capture_child_tasks=True
                                                )
                                             )
            api_server = api_server_class.remote(server_name, entrypoints_args, engine_args)
        # pylint: disable=broad-except
        except Exception as e:
            logger.exception("Failed to initialize APIServer: {}".format(e))
            raise

        return api_server

    def is_ready(self) -> bool:
        return True
