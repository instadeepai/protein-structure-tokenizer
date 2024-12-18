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

from typing import List

from structure_tokenizer.data.tools.utils import SubprocessManager


def call_pymol(
    pymol_script_path: str,
    arguments: List,
) -> None:
    """Plots the predicted/target PDB structures using a PyMol script.

    Args:
      pymol_script_path: path to the pymol script to be run.
      arguments: .
    """
    # Run PyMol script to plot the predicted and target structures.
    subprocess_manager = SubprocessManager()
    (success, error_msg, stdout) = subprocess_manager.run(
        [
            "pymol",
            "-cq",
            pymol_script_path,
            "--",
        ] + arguments
    )
    if not success:
        print(f"Failed to run PyMol script: {error_msg}.")

    result = stdout.decode("ascii")
    print(result)
