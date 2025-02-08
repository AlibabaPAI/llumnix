import threading
import traceback
import uvicorn

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from llumnix.arg_utils import EntrypointsArgs
from llumnix.entrypoints.utils import EntrypointsContext, get_ip_address
from llumnix.llumlet.llumlet import Llumlet
from llumnix.queue.utils import init_request_output_queue_server, QueueType
from llumnix.logging.logger import init_logger

logger = init_logger(__name__)


class APIServerActor:
    def __init__(self, server_name: str, entrypoints_args: EntrypointsArgs):
        self.job_id = ray.get_runtime_context().get_job_id()
        self.worker_id = ray.get_runtime_context().get_worker_id()
        self.actor_id = ray.get_runtime_context().get_actor_id()
        self.node_id = ray.get_runtime_context().get_node_id()
        self.instance_id = server_name.split("_")[-1]
        logger.info("APIServerActor(job_id={}, worker_id={}, actor_id={}, node_id={}, instance_id={})".format(
                        self.job_id, self.worker_id, self.actor_id, self.node_id, self.instance_id))
        self.entrypoints_args = entrypoints_args
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
                                        self.entrypoints_args,manager, [instance_id], [instance], self.request_output_queue)

    def _run_uvicorn_server(self,
                            entrypoints_args: EntrypointsArgs,
                            entrypoints_context: EntrypointsContext):
        # pylint: disable=import-outside-toplevel
        import llumnix.entrypoints.vllm.api_server
        from llumnix.entrypoints.vllm.client import LlumnixClientVLLM
        llumnix.entrypoints.vllm.api_server.llumnix_client = LlumnixClientVLLM(entrypoints_context)
        app = llumnix.entrypoints.vllm.api_server.app

        logger.info("Start api server on '{}:{}'.".format(self.host, entrypoints_args.port))
        uvicorn.run(app,
                    host=self.host,
                    port=entrypoints_args.port,
                    log_level=entrypoints_args.log_level,
                    timeout_keep_alive=llumnix.entrypoints.vllm.api_server.SERVER_TIMEOUT_KEEP_ALIVE,
                    ssl_keyfile=entrypoints_args.ssl_keyfile,
                    ssl_certfile=entrypoints_args.ssl_certfile)

    def run(self,
            manager: "ray.actor.ActorHandle",
            instance_id: str,
            instance: Llumlet):
        self._setup_entrypoints_context(manager, instance_id, instance)
        self.run_uvicorn_server_thread = threading.Thread(
            target=self._run_uvicorn_server, args=(self.entrypoints_args, self.entrypoints_context),
            daemon=True, name="run_uvicorn_server"
        )
        self.run_uvicorn_server_thread.start()

    @classmethod
    def from_args(cls,
                  server_name: str,
                  placement_group: PlacementGroup,
                  entrypoints_args: EntrypointsArgs):
        try:
            api_server_class = ray.remote(num_cpus=1,
                                          name=server_name,
                                          namespace="llumnix",
                                          lifetime="detached")(cls).options(
                                                scheduling_strategy=PlacementGroupSchedulingStrategy(
                                                    placement_group=placement_group,
                                                    placement_group_bundle_index=0,
                                                    placement_group_capture_child_tasks=True
                                                )
                                             )
            api_server = api_server_class.remote(server_name, entrypoints_args)
        # pylint: disable=broad-except
        except Exception as e:
            logger.error("Failed to initialize APIServer: {}".format(e))
            logger.error("Exception traceback: {}".format(traceback.format_exc()))
            raise

        return api_server

    def is_ready(self) -> bool:
        return True
