# Copyright 2024 InstaDeep Ltd. All rights reserved.#

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from typing import List

root_loggers: List[str] = []


def get_logger(name: str) -> logging.Logger:
    """Configuring logging.

    The level is configured thanks to the environment variable 'LOG_LEVEL'
        (default 'INFO').
    """
    global root_loggers

    name_root_logger = name.split(".")[0]

    if name_root_logger not in root_loggers:
        logger = logging.getLogger(name_root_logger)
        logger.propagate = False
        formatter = logging.Formatter(
            fmt=(
                "%(asctime)s | %(process)d | %(levelname)s | %(module)s:%(funcName)s:"
                "%(lineno)d | %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)
        default_log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
        logger.setLevel(default_log_level if default_log_level else "INFO")

        root_loggers.append(name_root_logger)

    return logging.getLogger(name)
