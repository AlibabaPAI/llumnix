from enum import Enum
from typing import Dict
import subprocess

from llumnix.logging.logger import init_logger

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
                 server: "APIServerActor",
                 server_info: "ServerInfo",
                 log_requests: bool,
                 log_request_timestamps: bool):
        self.manager = manager
        self.instances = instances
        self.request_output_queue = request_output_queue
        self.server = server
        self.server_info = server_info
        self.log_requests = log_requests
        self.log_request_timestamps = log_request_timestamps

def is_gpu_available() -> bool:
    try:
        subprocess.check_output(['nvidia-smi'])
        return True
    # pylint: disable=bare-except
    except:
        return False
