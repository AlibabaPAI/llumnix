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

# BLLM_KVTRANS_FSNAMING_KEEPALIVE_S 36000
# BLLM_KVTRANS_FSNAMING_TOLERATE_S 360000

import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

if TYPE_CHECKING:
    BLLM_KVTRANS_FSNAMING_KEEPALIVE_S: Optional[str] = None
    BLLM_KVTRANS_FSNAMING_TOLERATE_S: Optional[str] = None


environment_variables: Dict[str, Callable[[], Any]] = {
    # ================== Llumnix backend environment variables ==================

    # KVT configuration
    "BLLM_KVTRANS_FSNAMING_KEEPALIVE_S": lambda: os.getenv("BLLM_KVTRANS_FSNAMING_KEEPALIVE_S", "36000"),
    "BLLM_KVTRANS_FSNAMING_TOLERATE_S": lambda: os.getenv("BLLM_KVTRANS_FSNAMING_TOLERATE_S", "360000"),
}


# pylint: disable=invalid-name
def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# pylint: disable=invalid-name
def __dir__():
    return list(environment_variables.keys())
