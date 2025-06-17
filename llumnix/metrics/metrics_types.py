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
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import time

import numpy as np

class TimeRecorder:
    """
    A context manager class for recording the execution time of a code block.
    
    It starts timing when entering the context and stops when exiting,
    then records the elapsed time using the provided metrics.
    
    Attributes:
        metrics (Summary): The metrics object to record the elapsed time.
        enabled (bool): Whether the timing is enabled.
        labels (Dict[str, str]):  Metric labels.
        _start_time (float): The start time of the context.
        _end_time (float): The end time of the context.
    """
    def __init__(self, metrics: "Summary", enabled: bool = True, labels: Dict[str, str] = None):
        self.metrics: Summary = metrics
        self.enabled = enabled
        self.labels = labels
        self._start_time = 0.0
        self._end_time = 0.0

    @property
    def start_time(self) -> float:
        return self._start_time

    @property
    def end_time(self) -> float:
        return self._end_time

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.done()

    def start(self):
        if self.enabled:
            self._start_time = time.perf_counter()

    def done(self) -> Optional[float]:
        if not self.enabled:
            return
        self._end_time = time.perf_counter()
        cost_time = (self._end_time - self._start_time) * 1000  # record in millisecond
        self.metrics.observe(value=cost_time, labels=self.labels)


@dataclass
class MetricEntry:
    name: str
    value: float
    labels: Optional[Dict[str, str]] = None

    def __str__(self):
        if self.labels:
            return f"{self.name}{{{self.labels}}}: {self.value}."\

        return f"{self.name}{{}}: {self.value}."

    def __repr__(self):
        return self.__str__()


class Registery:
    def __init__(self):
        self._metrics: Dict[str, MetricWrapperBase] = {}

    def get(self, name: str) -> Optional["MetricWrapperBase"]:
        return self._metrics.get(name, None)

    def register(self, name: str, metric: "MetricWrapperBase"):
        if name in self._metrics:
            raise RuntimeError(f"Metric name already registered: {name}")
        self._metrics[name] = metric

    def describe(self) -> List[MetricEntry]:
        ret: List[MetricEntry] = []
        for _, metric in self._metrics.items():
            ret.extend(metric.collect())
        return ret

    def describe_ignore_labels(self) -> List[MetricEntry]:
        ret: List[MetricEntry] = []
        for _, metric in self._metrics.items():
            ret.extend(metric.collect_without_labels())
        return ret

    def clear(self):
        self._metrics.clear()

    # reset summary metrics value
    def reset(self):
        for _, metric in self._metrics.items():
            if isinstance(metric, (Summary, TimeAveragedCounter)):
                metric.reset()

    def remove(self, key) -> None:
        del self._metrics[key]


_REGISTRY = Registery()


class MetricWrapperBase(ABC):
    def __init__(self, name: str, registry: Registery = None, metrics_sampling_interval: int = 0):
        self._name: str = name
        self.metrics_sampling_interval: int = metrics_sampling_interval
        self.curr_metrics_index: int = -1
        if registry:
            registry.register(name, self)
        else:
            _REGISTRY.register(name, self)

    def increase_index_and_check_need_sample(self, increase_count: int =1):
        if self.metrics_sampling_interval <= 0:
            # disable metrics
            return False
        self.curr_metrics_index = (self.curr_metrics_index + increase_count) % self.metrics_sampling_interval
        return self.curr_metrics_index == 0

    @abstractmethod
    def collect(self) -> List[MetricEntry]:
        ...

    @abstractmethod
    def collect_without_labels(self) -> List[MetricEntry]:
        ...

    @abstractmethod
    def observe(self, value: Any, labels: Dict[str, str] = None) -> None:
        ...

    @property
    def name(self) -> str:
        return self._name


class Status(MetricWrapperBase):
    """
    A metric wrapper that represents a status value.
    
    This class holds the latest value of a metric and can be used to track 
    the current state of a system. It supports optional labels to differentiate 
    between groups of metrics.

    Attributes:
        _value (Any): The current value of the metric.
        _label_values (Dict[str, int]): Dictionary mapping label hashes to values.
        _hash_label_map (Dict[str, str]): Dictionary mapping label hashes to label dictionaries.
    """
    def __init__(
        self, name: str, registry: Registery = None, initial_value: Any = None, metrics_sampling_interval: int = 0
    ):
        super().__init__(name, registry, metrics_sampling_interval)
        self._value: Any = initial_value
        self._label_values: Dict[str, int] = {} # count groups by label
        self._hash_label_map: Dict[str, str] = {}

    def collect(self) -> List[MetricEntry]:
        res = []
        for label_hash, value in self._label_values.items():
            res.append(
                MetricEntry(
                    name=self.name,
                    value=value,
                    labels=self._hash_label_map.get(label_hash),
                )
            )
        res.append(MetricEntry(name=self.name, value=self._value))
        return res

    def collect_without_labels(self) -> List[MetricEntry]:
        return [MetricEntry(name=self.name, value=self._value)]

    def observe(self, value: Any, labels: Dict[str, str] = None) -> None:
        if not self.increase_index_and_check_need_sample():
            return
        self._value = value
        if labels is not None:
            label_hash = hash(frozenset(labels.items()))
            self._hash_label_map[label_hash] = labels
            self._label_values[label_hash] = value


class PassiveStatus(MetricWrapperBase):
    """
    A metric wrapper for passive status tracking.
    
    This class does not store values directly but instead relies on an external function 
    to provide the current value when requested. Useful for metrics that are observed externally.

    Attributes:
        get_func (Callable): A function that returns the current metric value.
    """
    def __init__(self, name, registry: Registery = None):
        super().__init__(name, registry)
        self.get_func = None

    def collect(self) -> List[MetricEntry]:
        return (
            [MetricEntry(name=self.name, value=self.get_func())]
            if self.get_func
            else None
        )

    def collect_without_labels(self):
        return (
            [MetricEntry(name=self.name, value=self.get_func())]
            if self.get_func
            else None
        )

    def observe(self, value, labels: Dict[str, str] = None) -> None:
        self.get_func = value


class Counter(MetricWrapperBase):
    """
    A metric wrapper representing a counter.
    
    Counters are cumulative and strictly increasing. They are typically used 
    to count events or occurrences over time.

    Attributes:
        _count (int): The total count.
        _label_counts (Dict[str, int]): Dictionary mapping label hashes to counts.
        _hash_label_map (Dict[str, str]): Dictionary mapping label hashes to label dictionaries.
    """
    def __init__(self, name, registry: Registery = None, metrics_sampling_interval=1):
        super().__init__(name, registry, metrics_sampling_interval)
        self._count: int = 0
        self._label_counts: Dict[str, int] = {} # count groups by label
        self._hash_label_map: Dict[str, str] = {}

    def collect(self) -> List[MetricEntry]:
        res = []
        for label_hash, count in self._label_counts.items():
            res.append(
                MetricEntry(
                    name=self.name,
                    value=count,
                    labels=self._hash_label_map.get(label_hash),
                )
            )
        res.append(MetricEntry(name=self.name, value=self._count))
        return res

    def collect_without_labels(self):
        return [MetricEntry(name=self.name, value=self._count)]

    def observe(self, value: float, labels: Dict[str, str] = None):
        self.increase(increase_count=value,labels=labels)

    def increase(self, increase_count: int = 1, labels: Dict[str, str] = None) -> None:
        if self.metrics_sampling_interval <= 0:
            # Counter type metrics only support enable or disable
            return
        self._count += increase_count
        if labels is not None:
            label_hash = hash(frozenset(labels.items()))
            self._hash_label_map[label_hash] = labels
            self._label_counts[label_hash] = (
                self._label_counts.get(label_hash, 0) + increase_count
            )

class TimeAveragedCounter(MetricWrapperBase):
    """
    A counter which produce sum value devided by time in seconds between each collect invocation.
    Note: It's not simply the sum of each sample.
    Used for qps.
    """

    def __init__(self, name, registry: Registery = None, metrics_sampling_interval=0):
        super().__init__(name, registry, metrics_sampling_interval)
        self.reset()

    def reset(self):
        self._count: int = 0
        self._label_counts: Dict[str, int] = {} # count groups by label
        self._hash_label_map: Dict[str, str] = {}
        self._create_time = time.perf_counter()

    def observe(self, value: float, labels: Dict[str, str] = None):
        self.increase(increase_count=value, labels=labels)

    # pylint: disable=arguments-renamed
    def increase(self, increase_count: int = 1, labels: Dict[str, str] = None) -> None:
        if self.metrics_sampling_interval <= 0:
            # Counter type metrics only support enable or disable
            return
        self._count += increase_count
        if labels is not None:
            label_hash = hash(frozenset(labels.items()))
            self._hash_label_map[label_hash] = labels
            self._label_counts[label_hash] = (
                self._label_counts.get(label_hash, 0) + increase_count
            )

    def collect(self) -> List[MetricEntry]:
        res = []
        time_sec = time.perf_counter() - self._create_time
        for label_hash, count in self._label_counts.items():
            res.append(
                MetricEntry(
                    name=self.name,
                    value=count / time_sec,
                    labels=self._hash_label_map.get(label_hash),
                )
            )
        res.append(MetricEntry(name=self.name, value=self._count / time_sec))
        return res

    def collect_without_labels(self):
        time_sec = time.perf_counter() - self._create_time
        return [MetricEntry(name=self.name, value=self._count / time_sec)]


class Summary(MetricWrapperBase):
    """
    A metric wrapper for summarizing a series of values.
    
    This class records a sequence of values and computes statistical summaries such as mean,
    min, max, and percentile values (e.g., p99).

    Attributes:
        label_samples (Dict[str, list[float]]): Dictionary mapping label hashes to lists of values.
        _hash_label_map (Dict[str, str]): Dictionary mapping label hashes to label dictionaries.
        _samples (List[float]): List of all recorded values.
    """

    def __init__(self, name: str, registry: Registery = None, metrics_sampling_interval: int = 0):
        super().__init__(name, registry, metrics_sampling_interval)
        self.reset()

    def reset(self):
        self.label_samples: Dict[str, list[float]] = {}
        self._hash_label_map: Dict[str, str] = {}  # value groups by label
        self._samples: List[float] = []  # all values

    def observe(self, value: float, labels: Dict[str, str] = None):
        """Record a value"""
        if not self.increase_index_and_check_need_sample():
            return
        if labels:
            label_hash = hash(frozenset(labels.items()))
            if label_hash not in self._hash_label_map:
                self.label_samples[label_hash] = []
                self._hash_label_map[label_hash] = labels
            self.label_samples[label_hash].append(value)
        self._samples.append(value)

    def observe_time(self, labels: Dict[str, str] = None):
        """Return a Timer context object to record excution time of a scope."""

        return TimeRecorder(metrics=self, enabled=self.increase_index_and_check_need_sample(), labels=labels)

    def collect(self) -> List[MetricEntry]:
        res = []
        # calculate label samples
        for label_hash, value_list in self.label_samples.items():
            arr = np.array(value_list)
            mean_val = arr.mean().item()
            min_val = arr.min().item()
            max_val = arr.max().item()
            p99_val = np.percentile(arr, 99).item()
            labels = self._hash_label_map.get(label_hash)
            res.append(MetricEntry(f"{self._name}_mean", mean_val, labels))
            res.append(MetricEntry(f"{self._name}_min", min_val, labels))
            res.append(MetricEntry(f"{self._name}_max", max_val, labels))
            res.append(MetricEntry(f"{self._name}_p99", p99_val, labels))

        # calculate all samples
        res.extend(self.collect_without_labels())

        return res

    def collect_without_labels(self):
        res = []
        # calculate all samples
        if self._samples:
            arr = np.array(self._samples)
            mean_val = arr.mean().item()
            min_val = arr.min().item()
            max_val = arr.max().item()
            p99_val = np.percentile(arr, 99).item()
            res.append(MetricEntry(f"{self._name}_mean", mean_val))
            res.append(MetricEntry(f"{self._name}_min", min_val))
            res.append(MetricEntry(f"{self._name}_max", max_val))
            res.append(MetricEntry(f"{self._name}_p99", p99_val))
        return res
