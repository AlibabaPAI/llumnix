import copy
import time
import asyncio
import ray

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncStream
from vllm import SamplingParams

from llumnix.backends.vllm.utils import check_engine_args
from llumnix.arg_utils import LlumnixEntrypointsArgs, EngineManagerArgs
from llumnix.logger import init_logger
from llumnix.entrypoints.utils import LlumnixEntrypointsContext
from llumnix.server_info import RequestTimestamps

logger = init_logger(__name__)

WAIT_MANAGER_INTERVAL = 5


def add_cli_args(parser):
    parser.set_namespace("llumnix")
    parser = LlumnixEntrypointsArgs.add_cli_args(parser)
    parser = EngineManagerArgs.add_cli_args(parser)
    parser.set_namespace("vllm")
    parser = AsyncEngineArgs.add_cli_args(parser)
    cli_args = parser.parse_args()
    return cli_args

def get_args(cfg, parser, cli_args):
    llumnix_entrypoints_args = LlumnixEntrypointsArgs.from_llumnix_config(cfg)
    LlumnixEntrypointsArgs.check_args(llumnix_entrypoints_args, parser)
    engine_manager_args = EngineManagerArgs.from_llumnix_config(cfg)
    EngineManagerArgs.check_args(engine_manager_args, parser)
    engine_args = AsyncEngineArgs.from_cli_args(cli_args)
    check_engine_args(engine_args, engine_manager_args)

    logger.info("llumnix_entrypoints_args: {}".format(llumnix_entrypoints_args))
    logger.info("engine_manager_args: {}".format(engine_manager_args))
    logger.info("engine_args: {}".format(engine_args))

    return llumnix_entrypoints_args, engine_manager_args, engine_args

async def manager_generate(prompt: str,
                           sampling_params: SamplingParams,
                           request_id: str,
                           llumnix_context: LlumnixEntrypointsContext) -> AsyncStream:
    results_generator = AsyncStream(request_id)
    llumnix_context.request_streams[request_id] = results_generator

    if sampling_params.n > 1 or sampling_params.use_beam_search:
        raise ValueError("Unsupported feature: multiple sequence decoding")
    # This request's outputs will be put to the request_output_queue of this api server no matter which instance it's running in.
    # If manager is unavailable, request will be directly added to the llumlet held by api server.
    try:
        server_info_copy = copy.deepcopy(llumnix_context.server_info)
        if llumnix_context.log_request_timestamps:
            # Hack request timestamps in server_info for latency breakdown.
            server_info_copy.request_timestamps = RequestTimestamps()
            server_info_copy.request_timestamps.api_server_manager_generate_timestamp = time.time()
        # await to catch exception
        await llumnix_context.engine_manager.generate.remote(request_id, server_info_copy, prompt, sampling_params)
        llumnix_context.manager_available = True
    except ray.exceptions.RayActorError:
        # Do not re-generate the request to avoid duplicate requests.
        if llumnix_context.manager_available:
            llumnix_context.manager_available = False
            return results_generator
        try:
            if llumnix_context.instance_num_requests:
                instance_id = min(llumnix_context.instance_num_requests, key=llumnix_context.instance_num_requests.get)
                llumnix_context.instance_num_requests[instance_id] += 1
                await llumnix_context.instances[instance_id].generate.remote(request_id, server_info_copy, prompt, sampling_params)
                logger.info("Manager is unavailable, directly pass request {} to instance {}".format(request_id, instance_id))
            else:
                logger.info("Manager is unavailable, but there is no instance behind this api server, "
                      "sleep {}s, waiting for manager restarts".format(WAIT_MANAGER_INTERVAL))
                await asyncio.sleep(WAIT_MANAGER_INTERVAL)
                return await asyncio.create_task(manager_generate(prompt, sampling_params, request_id, llumnix_context))
        except (ray.exceptions.RayActorError, KeyError):
            if instance_id in llumnix_context.instances:
                logger.info("[manager_generate] instance {} is dead".format(instance_id))
                del llumnix_context.instances[instance_id]
                del llumnix_context.instance_num_requests[instance_id]
                return await asyncio.create_task(manager_generate(prompt, sampling_params, request_id, llumnix_context))

    return results_generator

async def manager_abort(request_id: str, llumnix_context: LlumnixEntrypointsContext) -> None:
    try:
        logger.info("abort request: {}.".format(request_id))
        await llumnix_context.engine_manager.abort.remote(request_id)
    except ray.exceptions.RayActorError:
        logger.info("manager is unavailable")

async def manager_is_ready(llumnix_context: LlumnixEntrypointsContext):
    ready_status = await llumnix_context.engine_manager.is_ready.remote()
    return ready_status
