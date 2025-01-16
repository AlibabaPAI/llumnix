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
from typing import Any, Dict, Optional


class Registery:
    def __init__(self):
        self._metrics: Dict[str, Variable] = {}

    def get(self, name: str) -> Optional['Variable']:
        return self._metrics.get(name, None)

    def register(self, name: str, metric: 'Variable'):
        if name in self._metrics:
            raise RuntimeError(f"Metric name already registered: {name}")
        self._metrics[name] = metric

    def describe_all(self) -> Dict[str, Any]:
        ret = {}
        for _, metric in self._metrics.items():
            ret.update(metric.describe())
        return ret

    def clear(self):
        self._metrics.clear()

    def remove(self, key) -> None:
        del self._metrics[key]


_REGISTRY = Registery()


class Variable(ABC):
    def __init__(self, name: str):
        self._name: str = name
        _REGISTRY.register(name, self)

    @abstractmethod
    def collect(self) -> Any:
        ...

    @abstractmethod
    def observe(self, value: Any) -> None:
        ...

    def describe(self):
        return {self.name : self.collect()}

    @property
    def name(self) -> str:
        return self._name


class Status(Variable):
    def __init__(self, name: str, initial_value: Any = None):
        super().__init__(name)
        self._value: Any = initial_value

    def collect(self) -> Any:
        return self._value

    def observe(self, value: Any) -> None:
        self._value = value
