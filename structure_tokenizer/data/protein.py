# mypy: ignore-errors
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

"""Protein data type."""
import dataclasses
import io
import os
from typing import Any, Dict, Mapping, Optional

import jax
import numpy as np
from Bio.PDB import PDBParser
from six.moves import cPickle as pickle  # type: ignore # noqa: N813
from typing_extensions import TypeAlias

from structure_tokenizer.data import residue_constants

PyTree: TypeAlias = Any
FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.

# Complete sequence of chain IDs supported by the PDB format.
PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.


@dataclasses.dataclass(frozen=True)
class Protein:
    """Protein structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # 0-indexed number corresponding to the chain in the protein that this residue
    # belongs to.
    chain_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    def __post_init__(self):
        if len(np.unique(self.chain_index)) > PDB_MAX_CHAINS:
            raise ValueError(
                f"Cannot build an instance with more than {PDB_MAX_CHAINS} chains "
                "because these cannot be written to PDB format."
            )
        assert self.chain_index.shape[0] == self.aatype.shape[0]

    def get_sequence(self) -> str:
        all_chain_ids = np.unique(self.chain_index.astype(np.int32))
        assert (
            len(all_chain_ids) == 1
        ), "get_sequence is only supported for single-chain proteins"
        assert np.max(self.aatype) < len(residue_constants.restypes_with_x)
        return "".join(
            residue_constants.restypes_with_x[aa_index] for aa_index in self.aatype
        )

    @classmethod
    def from_atom37_rep(
        cls,
        atom37_positions: np.ndarray,
        atom37_gt_exists: np.ndarray,
        atom37_atom_exits: np.ndarray,
        aatype: np.ndarray,
        chain_id: str,
    ):
        assert chain_id in PDB_CHAIN_IDS
        assert len(aatype.shape) == 2
        assert aatype.shape[1] in [
            len(residue_constants.restypes),
            len(residue_constants.restypes_with_x),
        ]

        chain_index = PDB_CHAIN_IDS.find(chain_id)
        nb_residue = aatype.shape[0]

        return cls(
            atom_positions=atom37_positions,
            atom_mask=np.logical_and(atom37_atom_exits, atom37_gt_exists),
            aatype=np.where(aatype)[1],
            residue_index=np.array(range(nb_residue)),
            chain_index=np.array([chain_index for _ in range(nb_residue)]),
            b_factors=np.zeros(atom37_gt_exists.shape),
        )


def from_pdb_string(pdb_str: str, chain_id: Optional[str] = None) -> Protein:
    """Takes a PDB string and constructs a Protein object.

    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.

    Args:
      pdb_str: The contents of the pdb file
      chain_id: If chain_id is specified (e.g. A), then only that chain
        is parsed. Otherwise all chains are parsed.

    Returns:
      A new `Protein` parsed from the pdb contents.
    """
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(
            f"Only single model PDBs are supported. Found {len(models)} models."
        )
    model = models[0]

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []

    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for res in chain:
            if res.id[2] != " ":
                raise ValueError(
                    f"PDB contains an insertion code at chain {chain.id} and residue "
                    f"index {res.id[1]}. These are not supported."
                )
            res_shortname = residue_constants.restype_3to1.get(res.resname, "X")
            restype_idx = residue_constants.restype_order.get(
                res_shortname, residue_constants.restype_num
            )
            pos = np.zeros((residue_constants.atom_type_num, 3))
            mask = np.zeros((residue_constants.atom_type_num,))
            res_b_factors = np.zeros((residue_constants.atom_type_num,))
            for atom in res:
                if atom.name not in residue_constants.atom_types:
                    continue
                pos[residue_constants.atom_order[atom.name]] = atom.coord
                mask[residue_constants.atom_order[atom.name]] = 1.0
                res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue
            aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)

    # Chain IDs are usually characters so map these to ints.
    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=chain_index,
        b_factors=np.array(b_factors),
    )


def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
    chain_end = "TER"
    return (
        f"{chain_end:<6}{atom_index:>5}      {end_resname:>3} "
        f"{chain_name:>1}{residue_index:>4}"
    )


def to_pdb(prot: Protein) -> str:
    """Converts a `Protein` instance to a PDB string.

    Args:
      prot: The protein to convert to PDB.

    Returns:
      PDB string.
    """
    restypes = residue_constants.restypes + ["X"]
    res_1to3 = lambda r: residue_constants.restype_1to3.get(  # noqa: E731
        restypes[r], "UNK"
    )
    atom_types = residue_constants.atom_types
    # atom_idx = [residue_constants.atom_order[a] for a in ("N", "CA", "C", "O")]
    # atom_types = [residue_constants.atom_types[i] for i in atom_idx]

    pdb_lines = []

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    chain_index = prot.chain_index.astype(np.int32)
    b_factors = prot.b_factors

    if np.any(aatype > residue_constants.restype_num):
        raise ValueError("Invalid aatypes.")

    # Construct a mapping from chain integer indices to chain ID strings.
    chain_ids = {}
    for i in np.unique(chain_index):  # np.unique gives sorted output.
        if i >= PDB_MAX_CHAINS:
            raise ValueError(
                f"The PDB format supports at most {PDB_MAX_CHAINS} chains."
            )
        chain_ids[i] = PDB_CHAIN_IDS[i]

    pdb_lines.append("MODEL     1")
    atom_index = 1
    last_chain_index = chain_index[0]
    # Add all atom sites.
    for i in range(aatype.shape[0]):
        # Close the previous chain if in a multichain PDB.
        if last_chain_index != chain_index[i]:
            pdb_lines.append(
                _chain_end(
                    atom_index,
                    res_1to3(aatype[i - 1]),
                    chain_ids[chain_index[i - 1]],
                    residue_index[i - 1],
                )
            )
            last_chain_index = chain_index[i]
            atom_index += 1  # Atom index increases at the TER symbol.

        res_name_3 = res_1to3(aatype[i])
        for atom_name, pos, mask, b_factor in zip(
            atom_types, atom_positions[i], atom_mask[i], b_factors[i]
        ):
            if mask < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            element = atom_name[0]  # Protein supports only C, N, O, S, this works.
            charge = ""
            # PDB is a columnar format, every space matters here!
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_ids[chain_index[i]]:>1}"
                f"{residue_index[i]:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1

    # Close the final chain.
    pdb_lines.append(
        _chain_end(
            atom_index,
            res_1to3(aatype[-1]),
            chain_ids[chain_index[-1]],
            residue_index[-1],
        )
    )
    pdb_lines.append("ENDMDL")
    pdb_lines.append("END")

    # Pad all lines to 80 characters.
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return "\n".join(pdb_lines) + "\n"  # Add terminating newline.


def ideal_atom_mask(prot: Protein) -> np.ndarray:
    """Computes an ideal atom mask.

    `Protein.atom_mask` typically is defined according to the atoms that are
    reported in the PDB. This function computes a mask according to heavy atoms
    that should be present in the given sequence of amino acids.

    Args:
      prot: `Protein` whose fields are `numpy.ndarray` objects.

    Returns:
      An ideal atom mask.
    """
    return residue_constants.STANDARD_ATOM_MASK[prot.aatype]


def from_prediction(
    features: FeatureDict,
    result: ModelOutput,
    b_factors: Optional[np.ndarray] = None,
    remove_leading_feature_dimension: bool = True,
) -> Protein:
    """Assembles a protein from a prediction.

    Args:
      features: Dictionary holding model inputs.
      result: Dictionary holding model outputs.
      b_factors: (Optional) B-factors to use for the protein.
      remove_leading_feature_dimension: Whether to remove the leading dimension
        of the `features` values.

    Returns:
      A protein instance.
    """
    fold_output = result["structure_module"]

    def _maybe_remove_leading_dim(arr: np.ndarray) -> np.ndarray:
        return arr[0] if remove_leading_feature_dimension else arr

    if "asym_id" in features:
        chain_index = _maybe_remove_leading_dim(features["asym_id"])
    else:
        chain_index = np.zeros_like(_maybe_remove_leading_dim(features["aatype"]))

    if b_factors is None:
        b_factors = np.zeros_like(fold_output["final_atom_mask"])

    return Protein(
        aatype=_maybe_remove_leading_dim(features["aatype"]),
        atom_positions=fold_output["final_atom_positions"],
        atom_mask=fold_output["final_atom_mask"],
        residue_index=_maybe_remove_leading_dim(features["residue_index"]) + 1,
        chain_index=chain_index,
        b_factors=b_factors,
    )


def get_atom37_from_prediction(
    predictions: Dict[str, Any],
    features: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Obtains the atom37 representation for the predicted and target structures.

    Args:
      predictions: a dictionary with the predictions for one sample.
      features: a dictionary with the target features for one sample.

    Returns:
      output_dict: a dictionary containing:
        aatype: [num_res, 21] one-hot encoding of amino acids for the prediction.
        atom37_positions: [num_res, 37, 3] atom positions in the prediction.
        atom37_positions_ground_truth: [num_res, 37, 3] atom positions in the target.
        atom37_gt_exists: [num_res, 37] indicates if the atoms exist in the target.
        atom37_atom_exits: [num_res, 37] indicates if the atoms exist in the prediction.
    """
    # aatype_onehot = features["aatype"]
    # aatype = jnp.argmax(features["aatype"], axis=-1)

    # atom37_pred_positions = predictions["final_atom_positions"]

    # # We inverted the order of Cb and O in the atom14 representation of the feat.
    # Okay but needs to be fixed
    # idx_atom37_to_atom14 = get_atom37_to_atom14_map(aatype)
    # atom37_gt_mask = utils.batched_gather(
    #     features["atom14_gt_exists"], idx_atom37_to_atom14, batch_dims=1
    # )
    # atom37_gt_positions = atom14_to_atom37(
    #     features["atom14_gt_positions"], aatype
    # )

    output_dict = {
        "aatype": features["aatype"],
        "atom37_positions": predictions["final_atom_positions"],
        "atom37_positions_ground_truth": features["atom37_gt_positions"],
        "atom37_gt_exists": features["atom37_gt_exists"],
        "atom37_atom_exits": features["atom37_atom_exists"],
    }
    return output_dict


def pdb_from_atom37(
    atom37_positions: np.ndarray,
    atom37_gt_exists: np.ndarray,
    atom37_atom_exits: np.ndarray,
    aatype: np.ndarray,
):
    """Creates a Protein object for the pred structure from its atom37
        representation and return PDB str.

    Args:
      atom37_positions: [num_res, num_atom_type, 3] atom positions in the prediction.
      atom37_gt_exits: [num_res, num_atom_type]
        indicates if the atoms exist in the target.
      atom37_atom_exits: [num_res, num_atom_type]
        indicates if the atoms exist in the prediction.
      aatype: [num_res, 21] amino acid types for the prediction.

    Return: PDB string.
    """

    protein = Protein.from_atom37_rep(
        atom37_positions=atom37_positions,
        atom37_gt_exists=atom37_gt_exists,
        atom37_atom_exits=atom37_atom_exits,
        aatype=aatype,
        chain_id="A",
    )
    return to_pdb(protein)


def save_predicted_target_pdbs(
    predictions: Dict[str, Any],
    features: Dict[str, np.ndarray],
    output_dir: str,
    metrics: Dict[str, Any],
) -> None:
    """Gets the atom37 representation for the predicted/target structures and
        saves the corresponding pdbs. It also save the validation metrics.

    Args:
      predictions: a dictionary with the predictions for one sample.
      features: a dictionary with the target features for one sample.
      output_dir: path to the directory where the output files are stored
      (target.pdb and prediction.pdb).
      metrics: validation metrics to save
    """

    # nres = features.pop("nb_residues")
    nres = metrics["n_res"]

    # Saving metrics
    filepath_metrics = os.path.join(output_dir, "metrics.pkl")
    with open(filepath_metrics, "wb") as f:
        pickle.dump(metrics, f)

    # Unpad
    def unpad(x: PyTree, i: int) -> PyTree:
        return jax.tree_util.tree_map(lambda x: x[: int(i)], x)

    unpad_preds = unpad(
        predictions, nres
    )  # BG: warning, the unpadding does not work on many but unused keys
    unpad_feats = unpad(features, nres)
    # Get atom37 representation for predictions
    atom37_preds = get_atom37_from_prediction(unpad_preds, unpad_feats)
    # Build pred protein from atom37
    pred_protein = Protein.from_atom37_rep(
        atom37_positions=atom37_preds["atom37_positions"],
        atom37_gt_exists=atom37_preds["atom37_gt_exists"],
        atom37_atom_exits=atom37_preds["atom37_atom_exits"],
        aatype=atom37_preds["aatype"],
        chain_id="A",
    )
    # save pdb
    pred_filepath = os.path.join(output_dir, "prediction.pdb")
    with open(pred_filepath, "w") as f:
        f.write(to_pdb(pred_protein))

    # Build target protein from atom37
    target_protein = Protein.from_atom37_rep(
        atom37_positions=atom37_preds["atom37_positions_ground_truth"],
        atom37_gt_exists=atom37_preds["atom37_gt_exists"],
        atom37_atom_exits=atom37_preds["atom37_atom_exits"],
        aatype=atom37_preds["aatype"],
        chain_id="A",
    )
    # save pdb
    target_filepath = os.path.join(output_dir, "target.pdb")
    with open(target_filepath, "w") as f:
        f.write(to_pdb(target_protein))
