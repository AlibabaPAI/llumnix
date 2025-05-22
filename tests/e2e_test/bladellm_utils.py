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

from typing import List
import os
from threading import Thread

import psutil

from blade_llm.testing.server import ServerProc
from blade_llm.utils.network import get_free_port

from llumnix.logging.logger import init_logger

logger = init_logger(__name__)


class LlumnixServerProc(ServerProc):
    """
    A helper class wrapping around `llumnix.entrypoints.bladellm.serve` command to start and stop the server.
    """

    def __init__(self, cmd_args: List[str]):
        """
        Start a server process with the given command arguments.
        Arguments:
            cmd_args: command arguments appending to 'python3 -m llumnix.entrypoints.bladellm.serve'.
                      If --port is not given, a random port will be used and can be accessed via .port property.
                      If --host is not given, localhost will be used and can be accessed via .host property.
        """
        self._host = 'localhost'
        self._port = None

        for arg in cmd_args:
            if '--host=' in arg:
                self._host = arg.split('=')[1]
            if '--port=' in arg:
                self._port = int(arg.split('=')[1])

        if self._port is None:
            self._port = get_free_port()
            cmd_args.append(f'--port={self._port}')

        self._running = True

        cmd = (
            f"RAY_DEDUP_LOGS=0 "
            f"nohup python -u -m llumnix.entrypoints.bladellm.serve "
            f"{' '.join(cmd_args)} "
            f"> instance_{self._port}.out 2>&1 &"
        )
        logger.info("Starting server with cmd: {}".format(cmd))
        self._proc = psutil.Popen(
            cmd,
            shell=True,
            preexec_fn=os.setsid,
        )
        self._io_thread = Thread(target=self._streaming_outputs, daemon=True)
        self._io_thread.start()
