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
from typing import Any, Dict

from loguru import logger

class Dumper(ABC):
    @abstractmethod
    def dump(self, metrics: Dict[str, Any]) -> None:
        ...

class LoggerDumper(Dumper):
    def dump(self, metrics: Dict[str, Any]) -> None:
        logger.info("Metrics: {}", metrics)

class DummyDumper(Dumper):
    def dump(self, metrics: Dict[str, Any]) -> None:
        pass
