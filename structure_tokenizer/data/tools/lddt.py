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
from typing import List, Optional, Tuple

import numpy as np

from structure_tokenizer.data.protein import PDB_CHAIN_IDS, Protein, from_pdb_string, to_pdb
from structure_tokenizer.data.tools.utils import SubprocessManager
from structure_tokenizer.utils.utils import tmpdir_manager


def lddt_score_from_pdbs(
    filepath_pdb: str,
    filepath_reference_pdb: str,
    distance_inclusion_radius: float = 15.0,
    use_alpha_carbons_only: bool = False,
    include_structural_checks: bool = False,
) -> Tuple[bool, str, Optional[float], Optional[List[float]]]:
    for filepath in [filepath_pdb, filepath_reference_pdb]:
        if not os.path.isfile(filepath):
            return (False, f"File not found {filepath}", None, None)

    filepath_to_stereo_chemical_props = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../common/stereo_chemical_props.txt",
    )

    subprocess_manager = SubprocessManager()
    (success, error_msg, stdout) = subprocess_manager.run(
        ["lddt"]
        + (["-c"] if use_alpha_carbons_only else [])
        + (
            ["-f", "-p", filepath_to_stereo_chemical_props]
            if include_structural_checks
            else []
        )
        + [
            "-r",
            str(distance_inclusion_radius),
            filepath_pdb,
            filepath_reference_pdb,
        ],
        decode_stderr_using_ascii=True,
    )
    if not success:
        return (False, f"Failed to run lddt: {error_msg}.", None, None)

    result = stdout.decode("ascii").replace("\t", " ")  # type: ignore

    next_line_is_local_lddt_score = False
    global_lddt_score = None
    all_local_ldd_scores = []
    column_nb_local_ldd_score = 5 if include_structural_checks else 4
    for line in result.split("\n"):
        if next_line_is_local_lddt_score and bool(line):
            splitted_line = line.split(" ")
            if len(splitted_line) <= column_nb_local_ldd_score:
                return (
                    False,
                    f"Failed to parse local lddt scores from lddt stdout: {result}.",
                    None,
                    None,
                )
            all_local_ldd_scores.append(float(splitted_line[column_nb_local_ldd_score]))
        elif line.startswith("Chain"):
            next_line_is_local_lddt_score = True
        elif line.startswith("Global LDDT score:"):
            global_lddt_score = float(line.split(" ")[3])

    if global_lddt_score is None:
        return (
            False,
            f"Global LDDT score not found in lddt stdout: {str(stdout)}.",
            None,
            None,
        )

    if not bool(all_local_ldd_scores):
        return (
            False,
            f"Local LDDT sscore not found in lddt stdout: {str(stdout)}.",
            None,
            None,
        )
    return (True, "", global_lddt_score, all_local_ldd_scores)


def lddt_score_from_atom37_and_pdb(
    atom37_positions: np.ndarray,
    atom37_gt_exists: np.ndarray,
    atom37_atom_exits: np.ndarray,
    aatype: np.ndarray,
    filepath_ground_truth_pdb: str,
    chain_id: str,
    **kargs,
) -> Tuple[bool, str, Optional[float], Optional[List[float]]]:
    assert chain_id in PDB_CHAIN_IDS, f"invalid chain id {chain_id}"
    chain_index = PDB_CHAIN_IDS.find(chain_id)

    aa_gt_exists = np.any(np.logical_and(atom37_atom_exits, atom37_gt_exists), axis=-1)

    # Read residue indices from the ground truth pdb and use the same
    # indices when writing the prediction
    with open(filepath_ground_truth_pdb, "r") as file_ground_truth_pdb:
        residue_index = from_pdb_string(
            file_ground_truth_pdb.read(), chain_id=chain_id
        ).residue_index

    with tmpdir_manager(base_dir="/tmp") as tmp_dir:
        # Only include the amino acids that are in the ground truth pdb since
        # lddt rejects pairs of pdbs that do not match exactly
        protein = Protein(
            atom_positions=atom37_positions[aa_gt_exists, ...],
            atom_mask=np.logical_and(atom37_atom_exits, atom37_gt_exists)[
                aa_gt_exists, ...
            ],
            aatype=np.where(aatype[aa_gt_exists, ...])[1],
            residue_index=residue_index,
            chain_index=np.array([chain_index for _ in range(residue_index.shape[0])]),
            b_factors=np.zeros(atom37_gt_exists[aa_gt_exists, ...].shape),
        )

        prediction_pdb_filepath = os.path.join(tmp_dir, "prediction.pdb")
        with open(prediction_pdb_filepath, "w") as prediction_pdb_file:
            prediction_pdb_file.write(to_pdb(protein))
        (
            success,
            error_msg,
            global_lddt_score,
            all_local_ldd_scores,
        ) = lddt_score_from_pdbs(
            prediction_pdb_filepath, filepath_ground_truth_pdb, **kargs
        )

    if success and len(all_local_ldd_scores) != len(residue_index):
        return (
            False,
            f"Missing ldd scores in the output (got {len(all_local_ldd_scores)} "
            f"lddt local scores but expected {len(residue_index)})",
            None,
            None,
        )
    return (
        success,
        error_msg,
        global_lddt_score,
        all_local_ldd_scores,
    )


def lddt_score_from_atom37(
    atom37_positions: np.ndarray,
    atom37_gt_exists: np.ndarray,
    atom37_atom_exits: np.ndarray,
    aatype: np.ndarray,
    atom37_positions_ground_truth: np.ndarray,
    **kargs,
) -> Tuple[bool, str, Optional[float], Optional[List[float]]]:
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
        return lddt_score_from_atom37_and_pdb(
            atom37_positions,
            atom37_gt_exists,
            atom37_atom_exits,
            aatype,
            filepath_ground_truth_pdb,
            chain_id="A",
            **kargs,
        )
