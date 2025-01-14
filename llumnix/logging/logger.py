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

"""Logging configuration for Llumnix."""

# Adapted from vLLM(v0.6.6.post1):
# https://github.com/vllm-project/vllm/blob/5340a30d0193547a19e236757fec1f3f246642f9/vllm/logger.py

import json
import logging
from functools import lru_cache
from logging import Logger
from logging.config import dictConfig
from os import path
from types import MethodType
from typing import Any, cast

# pylint: disable=consider-using-from-import
import llumnix.envs as envs

try:
    # import vllm logger first avoid other logger being disabled
    # pylint: disable=unused-import
    import vllm.logger
except ImportError:
    pass

LLUMNIX_CONFIGURE_LOGGING = envs.LLUMNIX_CONFIGURE_LOGGING
LLUMNIX_LOGGING_CONFIG_PATH = envs.LLUMNIX_LOGGING_CONFIG_PATH
LLUMNIX_LOGGING_LEVEL = envs.LLUMNIX_LOGGING_LEVEL
LLUMNIX_LOGGING_PREFIX = envs.LLUMNIX_LOGGING_PREFIX
LLUMNIX_LOG_STREAM = envs.LLUMNIX_LOG_STREAM
LLUMNIX_LOG_NODE_PATH = envs.LLUMNIX_LOG_NODE_PATH

_FORMAT = (f"{LLUMNIX_LOGGING_PREFIX}%(levelname)s %(asctime)s "
           "%(filename)s:%(lineno)d] %(message)s")

DEFAULT_LOGGING_CONFIG = {
    "formatters": {
        "llumnix": {
            "class": "llumnix.logging.NewLineFormatter",
            "format": _FORMAT,
        },
    },
    "handlers": {
    },
    "loggers": {
        "llumnix": {
            "handlers": [],
            "level": "DEBUG",
            "propagate": False,
        },
    },
    "version": 1,
    "disable_existing_loggers": False
}

if LLUMNIX_LOG_STREAM:
    DEFAULT_LOGGING_CONFIG["handlers"]["stream"] = {
        "class": "logging.StreamHandler",
        "formatter": "llumnix",
        "level": LLUMNIX_LOGGING_LEVEL,
        "stream": "ext://sys.stdout",
    }
    DEFAULT_LOGGING_CONFIG["loggers"]["llumnix"]["handlers"].append("stream")

if LLUMNIX_LOG_NODE_PATH:
    DEFAULT_LOGGING_CONFIG["handlers"]["file"] = {
        "class": "llumnix.logging.NodeFileHandler",
        "formatter": "llumnix",
        "level": LLUMNIX_LOGGING_LEVEL,
        "base_path": LLUMNIX_LOG_NODE_PATH,
    }
    DEFAULT_LOGGING_CONFIG["loggers"]["llumnix"]["handlers"].append("file")

# pylint: disable=redefined-outer-name
@lru_cache
def _print_info_once(logger: Logger, msg: str) -> None:
    # Set the stacklevel to 2 to print the original caller's line info
    logger.info(msg, stacklevel=2)


# pylint: disable=redefined-outer-name
@lru_cache
def _print_warning_once(logger: Logger, msg: str) -> None:
    # Set the stacklevel to 2 to print the original caller's line info
    logger.warning(msg, stacklevel=2)


class _LlumnixLogger(Logger):
    """
    Note:
        This class is just to provide type information.
        We actually patch the methods directly on the :class:`logging.Logger`
        instance to avoid conflicting with other libraries such as
        `intel_extension_for_pytorch.utils._logger`.
    """

    def info_once(self, msg: str) -> None:
        """
        As :meth:`info`, but subsequent calls with the same message
        are silently dropped.
        """
        _print_info_once(self, msg)

    def warning_once(self, msg: str) -> None:
        """
        As :meth:`warning`, but subsequent calls with the same message
        are silently dropped.
        """
        _print_warning_once(self, msg)


def _configure_llumnix_root_logger() -> None:
    logging_config = dict[str, Any]()

    if not LLUMNIX_CONFIGURE_LOGGING and LLUMNIX_LOGGING_CONFIG_PATH:
        raise RuntimeError(
            "LLUMNIX_CONFIGURE_LOGGING evaluated to false, but "
            "LLUMNIX_LOGGING_CONFIG_PATH was given. LLUMNIX_LOGGING_CONFIG_PATH "
            "implies LLUMNIX_CONFIGURE_LOGGING. Please enable "
            "LLUMNIX_CONFIGURE_LOGGING or unset LLUMNIX_LOGGING_CONFIG_PATH.")

    print(f"LLUMNIX_CONFIGURE_LOGGING: {LLUMNIX_CONFIGURE_LOGGING}")
    print(f"LLUMNIX_LOG_STREAM: {LLUMNIX_LOG_STREAM}")
    print(f"LLUMNIX_LOG_NODE_PATH: {LLUMNIX_LOG_NODE_PATH}")

    if LLUMNIX_CONFIGURE_LOGGING:
        if LLUMNIX_LOG_STREAM:
            print(f"LLUMNIX_LOG_STREAM: {LLUMNIX_LOG_STREAM}")
            DEFAULT_LOGGING_CONFIG["handlers"]["stream"] = {
                "class": "logging.StreamHandler",
                "formatter": "llumnix",
                "level": LLUMNIX_LOGGING_LEVEL,
                "stream": "ext://sys.stdout",
            }
            DEFAULT_LOGGING_CONFIG["loggers"]["llumnix"]["handlers"].append("stream")

        if LLUMNIX_LOG_NODE_PATH:
            print(f"LLUMNIX_LOG_NODE_PATH: {LLUMNIX_LOG_NODE_PATH}")
            DEFAULT_LOGGING_CONFIG["handlers"]["file"] = {
                "class": "llumnix.logging.NodeFileHandler",
                "formatter": "llumnix",
                "level": LLUMNIX_LOGGING_LEVEL,
                "base_path": LLUMNIX_LOG_NODE_PATH,
            }
            DEFAULT_LOGGING_CONFIG["loggers"]["llumnix"]["handlers"].append("file")
        
        print(f"DEFAULT_LOGGING_CONFIG: {DEFAULT_LOGGING_CONFIG}")

        logging_config = DEFAULT_LOGGING_CONFIG

    if LLUMNIX_LOGGING_CONFIG_PATH:
        if not path.exists(LLUMNIX_LOGGING_CONFIG_PATH):
            # pylint: disable=raising-format-tuple
            raise RuntimeError(
                "Could not load logging config. File does not exist: %s",
                LLUMNIX_LOGGING_CONFIG_PATH)
        with open(LLUMNIX_LOGGING_CONFIG_PATH, encoding="utf-8") as file:
            custom_config = json.loads(file.read())

        if not isinstance(custom_config, dict):
            # pylint: disable=raising-format-tuple
            raise ValueError("Invalid logging config. Expected Dict, got %s.",
                             type(custom_config).__name__)
        logging_config = custom_config

    if logging_config:
        dictConfig(logging_config)


def init_logger(name: str) -> _LlumnixLogger:
    """The main purpose of this function is to ensure that loggers are
    retrieved in such a way that we can be sure the root llumnix logger has
    already been configured."""

    # pylint: disable=redefined-outer-name
    logger = logging.getLogger(name)

    methods_to_patch = {
        "info_once": _print_info_once,
        "warning_once": _print_warning_once,
    }

    for method_name, method in methods_to_patch.items():
        setattr(logger, method_name, MethodType(method, logger))

    return cast(_LlumnixLogger, logger)


# The root logger is initialized when the module is imported.
# This is thread-safe as the module is only imported once,
# guaranteed by the Python GIL.
_configure_llumnix_root_logger()

logger = init_logger(__name__)
