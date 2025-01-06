import threading

from ray.util.placement_group import PlacementGroup

from llumnix.arg_utils import EntrypointsArgs
from llumnix.entrypoints.utils import EntrypointsContext, get_ip_address
from llumnix.llumlet.llumlet import Llumlet
from llumnix.utils import get_server_name
from llumnix.queue.utils import init_request_output_queue_server, QueueType


class FastAPIServer:
    def __init__(self, entrypoints_args: EntrypointsArgs):
        self.entrypoints_args = entrypoints_args
        self.request_output_queue_port = self.entrypoints_args.request_output_queue_port
        self.request_output_queue_type = QueueType(self.entrypoints_args.request_output_queue_type)
        ip = get_ip_address()
        self.request_output_queue = init_request_output_queue_server(
                                        ip, self.request_output_queue_port, self.request_output_queue_type)

    def setup_entrypoints_context(self,
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

        logger.info("Start api server on '{}:{}'.".format(entrypoints_args.host, entrypoints_args.port))
        uvicorn.run(app,
                    host=entrypoints_args.host,
                    port=entrypoints_args.port,
                    log_level=entrypoints_args.log_level,
                    timeout_keep_alive=llumnix.entrypoints.vllm.api_server.TIMEOUT_KEEP_ALIVE,
                    ssl_keyfile=entrypoints_args.ssl_keyfile,
                    ssl_certfile=entrypoints_args.ssl_certfile)

    def run(self):
        self.run_uvicorn_server_thread = threading.Thread(
            target=self._run_uvicorn_server, args=(self.entrypoints_args, self.entrypoints_context),
            daemon=True, name="run_uvicorn_server"
        )
        self.run_uvicorn_server_thread.start()

    @classmethod
    def from_args(cls,
                  instance_id: str,
                  placement_group: PlacementGroup,
                  entrypoints_args: EntrypointsArgs):
        try:
            fastapi_server_class = ray.remote(num_cpus=1,
                                              name=get_server_name(instance_id),
                                              namespace="llumnix",
                                              lifetime="detached")(cls).options(
                                                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                                                        placement_group=placement_group,
                                                        placement_group_bundle_index=0,
                                                        placement_group_capture_child_tasks=True
                                                    )
                                             )
            fastapi_server = fastapi_server_class.remote(entrypoints_args)
        # pylint: disable=broad-except
        except Exception as e:
            logger.error("failed to initialize FastAPIServer: {}".format(e))
            logger.error("exception traceback: {}".format(traceback.format_exc()))

        return fastapi_server

    def is_ready(self) -> bool:
        return True
