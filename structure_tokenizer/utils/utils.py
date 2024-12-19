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

import os
import contextlib
import shutil
import tempfile
import subprocess
import re
from typing import Any, List, Optional, Union, Iterator

import hydra
from ml_collections import ConfigDict, config_dict
from omegaconf import DictConfig, ListConfig

import numpy as np


def convert_to_ml_dict(dct: Union[DictConfig, Any]) -> Union[ConfigDict, Any]:
    """
    This function converts the DictConfig returned by Hydra
    into a ConfigDict. The recursion allows to convert
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


def make_tm_compare(folder1, folder2):
    """
    Compare protein structures using TM-align and extract both RMSD and TM-score.

    Args:
        folder1 (str): Path to the first folder containing PDB files.
        folder2 (str): Path to the second folder containing PDB files.

    Returns:
        dict: RMSD values for matching structures.
        dict: TM-scores for matching structures.
    """
    # Path to the TM-align executable (ensure it's in your PATH or provide full path)
    tm_align_executable = "TMalign"

    # Get list of files from both directories
    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2)

    # Create a dictionary to store the matches from folder1
    pdb_files1 = {}
    for file in files1:
        # Normalize the file name to match the pattern without "structure_"
        if file.endswith(".pdb"):
            normalized_name = file.replace("structure_", "").replace(".pdb", "")
            pdb_files1[normalized_name] = os.path.join(folder1, file)

    # Dictionaries to store RMSD and TM-scores
    rmsd_dict = {}
    tm_dict = {}

    # Iterate over files in folder2 and find matching files in folder1
    for file in files2:
        if file.endswith(".pdb"):
            normalized_name = file.replace(".pdb", "")
            # Check if the normalized file name matches one from folder1
            # if normalized_name in pdb_files1:
            match = [x for x in pdb_files1 if x in normalized_name]
            if len(match) > 0:
                file1_path = pdb_files1[match[0]]
                file2_path = os.path.join(folder2, file)

                # Construct the TM-align command
                command = [tm_align_executable, file1_path, file2_path]

                print(f"Running TM-align for: {file1_path} and {file2_path}")

                # Execute the TM-align command and capture the output
                result = subprocess.run(command, capture_output=True, text=True)

                # Extract the RMSD using a regular expression
                # Example line to match: "Aligned length=.*,\s*RMSD=\s*1.23"
                rmsd_match = re.search(r"Aligned length=.*,\s*RMSD=\s*([\d.]+)",
                                       result.stdout)

                # Extract the TM-score using a regular expression
                # Example line to match: "TM-score= 0.1234"
                tm_match = re.search(r"TM-score=\s*([\d.]+)", result.stdout)

                if rmsd_match:
                    rmsd_value = float(rmsd_match.group(1))
                    rmsd_dict[normalized_name] = rmsd_value
                    print(f"RMSD for {normalized_name}: {rmsd_value}")
                else:
                    print(f"RMSD value not found for {normalized_name}")

                if tm_match:
                    tm_value = float(tm_match.group(1))
                    tm_dict[normalized_name] = tm_value
                    print(f"TM-score for {normalized_name}: {tm_value}")
                else:
                    print(f"TM-score not found for {normalized_name}")

    print("All matching files have been processed.")
    print("RMSD values:", rmsd_dict)
    print("TM-scores:", tm_dict)

    # Print aggregate statistics
    if rmsd_dict:
        print("RMSD Mean:", np.mean(list(rmsd_dict.values())))
        print("RMSD Std Dev:", np.std(list(rmsd_dict.values())))
    if tm_dict:
        print("TM-score Mean:", np.mean(list(tm_dict.values())))
        print("TM-score Std Dev:", np.std(list(tm_dict.values())))

    return rmsd_dict, tm_dict
