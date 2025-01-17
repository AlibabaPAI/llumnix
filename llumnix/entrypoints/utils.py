import socket
from enum import Enum
from typing import Dict
import subprocess
import asyncio
import time
import ray

from llumnix.logging.logger import init_logger
from llumnix import constants

MAX_TASK_RETRIES = constants.MAX_TASK_RETRIES
RETRIES_INTERVAL = constants.RETRIES_INTERVAL

logger = init_logger(__name__)


# Put it in utils.py to avoid circular import.
class LaunchMode(str, Enum):
    LOCAL = "LOCAL"
    GLOBAL = "GLOBAL"


# Use "" type hint to avoid circular import.
class EntrypointsContext:
    def __init__(self,
                 manager: "Manager",
                 instances: Dict[str, "Llumlet"],
                 request_output_queue: "QueueServerBase",
                 server_info: "ServerInfo",
                 log_requests: bool,
                 log_request_timestamps: bool):
        self.manager = manager
        self.instances = instances
        self.request_output_queue = request_output_queue
        self.server_info = server_info
        self.log_requests = log_requests
        self.log_request_timestamps = log_request_timestamps


def get_ip_address():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address

def is_gpu_available() -> bool:
    try:
        subprocess.check_output(['nvidia-smi'])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def retry_manager_method_sync(ray_call, method_name, *args, **kwargs):
    for attempt in range(MAX_TASK_RETRIES):
        try:
            ret = ray.get(ray_call(*args, **kwargs))
            break
        except ray.exceptions.RayActorError:
            if attempt < MAX_TASK_RETRIES - 1:
                logger.warning("Manager is unavailable, sleep {}s, and retry {} again.".format(RETRIES_INTERVAL, method_name))
                time.sleep(RETRIES_INTERVAL)
            else:
                logger.error("Manager is still unavailable after {} times retries.".format(MAX_TASK_RETRIES))
                raise
    return ret

async def retry_manager_method_async(ray_call, method_name, *args, **kwargs):
    for attempt in range(MAX_TASK_RETRIES):
        try:
            ret = await ray_call(*args, **kwargs)
            break
        except ray.exceptions.RayActorError:
            if attempt < MAX_TASK_RETRIES - 1:
                logger.warning("Manager is unavailable, sleep {}s, and retry {} again.".format(RETRIES_INTERVAL, method_name))
                await asyncio.sleep(RETRIES_INTERVAL)
            else:
                logger.error("Manager is still unavailable after {} times retries.".format(MAX_TASK_RETRIES))
                raise
    return ret

def init_per_token_latency_breakdown_dict() -> Dict[str, int]:
    per_token_latency_breakdown_dict = {
        'step_latency_engine': [],
        'process_model_outputs_latency': [],
        'step_postprocess_latency': [],
        'across_async_put_queue_thread_latency': [],
        'across_async_put_queue_actor_latency': [],
        'queue_rpc_latency': [],
        'background_process_get_queue_latency': [],
        'generate_benchmark_return_output_latency': []
    }
    return per_token_latency_breakdown_dict

def record_per_token_latency_breakdown(per_token_latency_breakdown_dict: Dict[str, int], request_timestamps: "RequestTimestamps"):
    for key in per_token_latency_breakdown_dict.keys():
        per_token_latency_breakdown_dict[key].append(getattr(request_timestamps, key))
