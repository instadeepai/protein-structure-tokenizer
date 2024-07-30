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

import contextlib
import shutil
import tempfile
from typing import Any, List, Optional, Union, Iterator

import hydra
from ml_collections import ConfigDict, config_dict
from omegaconf import DictConfig, ListConfig


def convert_to_ml_dict(dct: Union[DictConfig, Any]) -> Union[ConfigDict, Any]:
    """
    This function converts the DictConfig returned by Hydra
    into a ConfigDict. The recusion allows to convert
    all the nested DictConfig elements of the config. The recursion stops
    once the reached element is not a DictConfig.
    """
    if not type(dct) is DictConfig:
        if type(dct) is ListConfig:
            return list(dct)
        return dct
    dct_ml = config_dict.ConfigDict()
    for k in list(dct.keys()):
        dct_ml[k] = convert_to_ml_dict(dct[k])
    return dct_ml


def load_config(
    name: str,
    job_name: str,
    overrides: Optional[List[str]] = None,
    config_path: str = "../config",
):
    if overrides is None:
        overrides = []

    with hydra.initialize(config_path=config_path, job_name=job_name):
        config = hydra.compose(config_name=name, overrides=overrides)
        return convert_to_ml_dict(config)


@contextlib.contextmanager
def tmpdir_manager(base_dir: Optional[str] = None) -> Iterator[str]:
    """Context manager that deletes a temporary directory on exit."""
    tmpdir = tempfile.mkdtemp(dir=base_dir)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

