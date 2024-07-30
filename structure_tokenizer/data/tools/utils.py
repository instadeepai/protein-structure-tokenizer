# Copyright 2021 DeepMind Technologies Limited
#
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
"""Common utilities for data pipeline tools."""
import contextlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, List, Optional, Sequence, Tuple, TypeVar, Union  # noqa: F401

from absl import logging


T = TypeVar("T")


class SubprocessManager:
    def __init__(
        self,
        exit_on_sigint: bool = True,
    ) -> None:
        self.exit_on_sigint = exit_on_sigint
        self._process = None  # type: Any
        signal.signal(signal.SIGTERM, self._kill_subprocess)
        signal.signal(signal.SIGINT, self._kill_subprocess)

    def _kill_subprocess(self, signal_nb, frame):
        if self._process is not None:
            self._process.kill()
            self._process.wait()

        if self.exit_on_sigint and signal_nb == int(signal.SIGINT):
            sys.exit(0)

    def run(
        self,
        command: List[str],
        timeout: float = 60.0,
        decode_stderr_using_ascii: bool = False,
    ) -> Tuple[bool, str, Optional[Union[bytes, str]]]:
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ.copy(),
        )

        try:
            stdout, stderr = self._process.communicate(timeout=timeout)
            retcode = self._process.returncode
            self._process = None
        except subprocess.TimeoutExpired:
            self._kill_subprocess(None, None)
            return (False, f"Timeout of {round(timeout, 1)} expired", None)

        success = retcode == 0
        if not success:
            return (
                False,
                f"Process exited with exit code {retcode}, stderr: "
                f"{stderr.decode('ascii') if decode_stderr_using_ascii else stderr}, "
                "stdout: "
                f"{stdout.decode('ascii') if decode_stderr_using_ascii else stdout}",
                None,
            )
        return (True, "", stdout)
