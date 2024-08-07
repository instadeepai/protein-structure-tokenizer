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

# mypy: ignore-errors

import os
from typing import Dict, Optional, Tuple

import numpy as np

from structure_tokenizer.data.protein import Protein, to_pdb
from structure_tokenizer.data.tools.utils import SubprocessManager, tmpdir_manager
from structure_tokenizer.utils.log import get_logger

logger = get_logger(__name__)


def tm_score_from_pdbs(
    filepath_pdb: str,
    filepath_reference_pdb: str,
) -> Tuple[bool, str, Optional[Dict[str, float]]]:
    """Computes the TM-score, RMSD and GDT metrics from prediction/target PDB files.

    Args:
      filepath_pdb: path to the predicted PDB file.
      filepath_reference_pdb: path to the target PDB file.

    Returns:
      scores_dict: a dictionary containing the metrics {rmsd, tm, maxsub, gdt_ts,
        gdt_ha}.
    """

    for filepath in [filepath_pdb, filepath_reference_pdb]:
        if not os.path.isfile(filepath):
            return (False, f"File not found {filepath}", None, None)
    subprocess_manager = SubprocessManager()
    (success, error_msg, stdout) = subprocess_manager.run(
        [
            "TMscore",
            filepath_pdb,
            filepath_reference_pdb,
        ],
        decode_stderr_using_ascii=True,
    )
    if not success:
        return (False, f"Failed to run TMscore: {error_msg}.", None)

    result = stdout.decode("ascii")
    scores_dict = {}

    if "RMSD of" not in result:
        logger.info(f"RMSD not found in TMscore stdout: {stdout}.")
        return (False, f"RMSD not found in TMscore stdout: {stdout}.", None)
    scores_dict["rmsd"] = float(result[result.find("RMSD of") :].split()[5])

    if "TM-score    =" not in result:
        logger.info(f"TM-score not found in TMscore stdout: {stdout}.")
        return (False, f"TM-score not found in TMscore stdout: {stdout}.", None)
    scores_dict["tm"] = float(result[result.find("TM-score    =") :].split()[2])

    if "MaxSub-score=" not in result:
        logger.info(f"MaxSub-score not found in TMscore stdout: {stdout}.")
        return (False, f"MaxSub-score not found in TMscore stdout: {stdout}.", None)
    scores_dict["maxsub"] = float(result[result.find("MaxSub-score=") :].split()[1])

    if "GDT-TS-score=" not in result:
        logger.info(f"GDT-TS-score not found in TMscore stdout: {stdout}.")
        return (False, f"GDT-TS-score not found in TMscore stdout: {stdout}.", None)
    scores_dict["gdt_ts"] = float(result[result.find("GDT-TS-score=") :].split()[1])

    if "GDT-HA-score=" not in result:
        logger.info(f"GDT-HA-score not found in TMscore stdout: {stdout}.")
        return (False, f"GDT-HA-score not found in TMscore stdout: {stdout}.", None)
    scores_dict["gdt_ha"] = float(result[result.find("GDT-HA-score=") :].split()[1])

    return (True, "", scores_dict)


def tm_score_from_atom37_and_pdb(
    atom37_positions: np.ndarray,
    atom37_gt_exists: np.ndarray,
    atom37_atom_exits: np.ndarray,
    aatype: np.ndarray,
    filepath_ground_truth_pdb: str,
    chain_id: str,
) -> Tuple[bool, str, Optional[Dict[str, float]]]:
    """Creates a Protein object for the predicted structure from its atom37
        representation and saves it to a PDB file, used to compute the TM-score.

    Args:
      atom37_positions: [num_res, 37, 3] atom positions in the prediction.
      atom37_atom_exits: [num_res, 37] indicates if the atoms exist in the prediction.
      aatype: [num_res, 21] amino acid types for the prediction.
      filepath_ground_truth_pdb: path to the target PDB file.

    Returns: Output metrics from ``tm_score_from_pdbs``.
    """

    protein = Protein.from_atom37_rep(
        atom37_positions=atom37_positions,
        atom37_gt_exists=atom37_gt_exists,
        atom37_atom_exits=atom37_atom_exits,
        aatype=aatype,
        chain_id=chain_id,
    )
    with tmpdir_manager(base_dir="/tmp") as tmp_dir:
        prediction_pdb_filepath = os.path.join(tmp_dir, "prediction.pdb")
        with open(prediction_pdb_filepath, "w") as prediction_pdb_file:
            prediction_pdb_file.write(to_pdb(protein))

        return tm_score_from_pdbs(
            prediction_pdb_filepath,
            filepath_ground_truth_pdb,
        )


def tm_score_from_atom37(
    atom37_positions: np.ndarray,
    atom37_gt_exists: np.ndarray,
    atom37_atom_exits: np.ndarray,
    aatype: np.ndarray,
    atom37_positions_ground_truth: np.ndarray,
) -> Tuple[bool, str, Optional[Dict[str, float]]]:
    """Creates a Protein object for the target structure from its atom37
        representation and saves it to a PDB file, used to compute the TM-score.

    Args:
      atom37_positions: [num_res, 37, 3] atom positions in the prediction.
      atom37_gt_exits: [num_res, 37] indicates if the atoms exist in the target.
      atom37_atom_exits: [num_res, 37] indicates if the atoms exist in the prediction.
      aatype: [num_res, 21] amino acid types for the prediction.
      atom37_positions_ground_truth: [num_res, 37, 3] atom positions in the target.

    Returns: Output metrics from ``tm_score_from_atom37_and_pdb``.
    """

    ground_truth_protein = Protein.from_atom37_rep(
        atom37_positions=atom37_positions_ground_truth,
        atom37_gt_exists=atom37_gt_exists,
        atom37_atom_exits=atom37_atom_exits,
        aatype=aatype,
        chain_id="A",
    )
    with tmpdir_manager(base_dir="/tmp") as tmp_dir:
        filepath_ground_truth_pdb = os.path.join(tmp_dir, "ground_truth.pdb")
        with open(filepath_ground_truth_pdb, "w") as file_ground_truth_pdb:
            file_ground_truth_pdb.write(to_pdb(ground_truth_protein))
        return tm_score_from_atom37_and_pdb(
            atom37_positions,
            atom37_gt_exists,
            atom37_atom_exits,
            aatype,
            filepath_ground_truth_pdb,
            chain_id="A",
        )
