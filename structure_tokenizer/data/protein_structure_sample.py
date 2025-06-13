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

import io
import os
from typing import NamedTuple, Tuple, Optional

import jax
import numpy as np
from Bio.PDB import PDBParser

from structure_tokenizer.data import residue_constants
from structure_tokenizer.model import all_atom, quat_affine


class ProteinStructureSample(NamedTuple):
    chain_id: str
    nb_residues: int
    aatype: np.ndarray  # of type bool with shape (nb_residues, 21),
    # One-hot representation of the input amino acid sequence (20 amino acids + unknown)
    # with the residues indexed according to residue_contants.RESTYPES
    atom37_positions: np.ndarray  # of type float32 with shape (nb_residues, 37, 3),
    # atom37 representations of the 3D structure of the protein
    atom37_gt_exists: np.ndarray  # of type bool with shape (nb_residues, 37), Mask
    # denoting whether the corresponding atom's position was specified in the
    # pdb databank entry.
    atom37_atom_exists: np.ndarray  # of type bool with shape (nb_residues, 37), Mask
    # denoting whether the corresponding atom exists for each residue in the atom37
    # representation.
    resolution: float  # experimental resolution of the 3D structure as specified in the
    # pdb databank entry if available, otherwise 0.
    pdb_cluster_size: int  # size of the cluster in PDB this sample belongs to, 1 if not
    # available

    @classmethod
    def from_file(cls, filepath: str) -> "ProteinStructureSample":
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"{filepath} does not exist")

        with open(filepath, "rb") as file:
            dict_representation = np.load(file, allow_pickle=True)[()]

        return cls(**dict_representation)

    def to_file(self, filepath: str) -> None:
        assert os.path.exists(os.path.dirname(filepath))
        dict_representation = self._asdict()
        np.save(
            filepath,
            dict_representation,
        )

    def get_missing_backbone_coords_mask(self) -> np.ndarray:
        return ~(
            self.atom37_gt_exists[:, residue_constants.CA_INDEX]
            & self.atom37_gt_exists[:, residue_constants.N_INDEX]
            & self.atom37_gt_exists[:, residue_constants.C_INDEX]
            & self.atom37_gt_exists[:, residue_constants.O_INDEX]
        )

    def get_local_reference_frames(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ca_coords = self.atom37_positions[:, residue_constants.CA_INDEX]
        n_coords = self.atom37_positions[:, residue_constants.N_INDEX]
        c_coords = self.atom37_positions[:, residue_constants.C_INDEX]

        any_missing_coords = self.get_missing_backbone_coords_mask()

        x_axis = n_coords - ca_coords
        x_axis_norm = np.where(any_missing_coords, 1.0, np.linalg.norm(x_axis, axis=-1))
        assert np.all(x_axis_norm > 1e-3)
        x_axis /= np.expand_dims(x_axis_norm, axis=-1)

        z_axis = np.cross(x_axis, c_coords - ca_coords)
        z_axis_norm = np.where(any_missing_coords, 1.0, np.linalg.norm(z_axis, axis=-1))
        assert np.all(z_axis_norm > 1e-3)
        z_axis /= np.expand_dims(z_axis_norm, axis=-1)

        y_axis = np.cross(z_axis, x_axis)
        assert x_axis.shape == y_axis.shape == z_axis.shape
        return (x_axis, y_axis, z_axis)

    def make_protein_features(self):
        features = {
            "aatype": self.aatype,
            "atom37_gt_positions": self.atom37_positions,
        }
        # backbone atoms idx
        atom37_atom_to_idx = {
            k: v
            for (k, v) in residue_constants.atom_order.items()
            if k in ("N", "CA", "C", "O")
        }
        # O and CB position are swapped in atom14 representation
        atom14_atom_to_idx = {
            "N": residue_constants.atom_order["N"],
            "CA": residue_constants.atom_order["CA"],
            "O": residue_constants.atom_order["C"],
            "C": residue_constants.atom_order["O"],
        }
        # create atom37 masks
        atom37_gt_exists = np.zeros((self.nb_residues, 37), dtype=np.int32)
        atom37_atom_exists = np.zeros((self.nb_residues, 37), dtype=np.int32)
        for idx in atom14_atom_to_idx.values():  # BG: can we do more efficient here?
            atom37_gt_exists[:, idx] = self.atom37_gt_exists[:, idx]
            atom37_atom_exists[:, idx] = self.atom37_atom_exists[:, idx]
        features["atom37_gt_exists"] = atom37_gt_exists
        features["atom37_atom_exists"] = atom37_atom_exists
        # we need atom14 representation
        atom14_gt_positions = np.zeros((self.nb_residues, 14, 3), dtype=np.float32)
        atom14_gt_exists = np.zeros((self.nb_residues, 14), dtype=np.int32)
        for atom in ("N", "CA", "C", "O"):  # BG: can we do more efficient here?
            atom14_gt_positions[:, atom14_atom_to_idx[atom]] = self.atom37_positions[
                :, atom37_atom_to_idx[atom]
            ]
            atom14_gt_exists[:, atom14_atom_to_idx[atom]] = self.atom37_gt_exists[
                :, atom37_atom_to_idx[atom]
            ]
        features["atom14_gt_positions"] = atom14_gt_positions
        features["atom14_gt_exists"] = atom14_gt_exists
        # compute backbone affine
        rot, trans = quat_affine.make_transform_from_reference(
            n_xyz=self.atom37_positions[:, atom37_atom_to_idx["N"], :],
            ca_xyz=self.atom37_positions[:, atom37_atom_to_idx["CA"], :],
            c_xyz=self.atom37_positions[:, atom37_atom_to_idx["C"], :],
        )

        def rot_to_quat(x):
            return quat_affine.rot_to_quat(x[0], True)

        features["backbone_affine_tensor"] = np.concatenate(
            [
                np.array(list(map(rot_to_quat, np.split(rot, rot.shape[0])))),
                trans,
            ],
            axis=-1,
        )

        features["backbone_affine_mask"] = (
            self.atom37_gt_exists[..., atom37_atom_to_idx["N"]].astype(np.float32)
            * self.atom37_gt_exists[..., atom37_atom_to_idx["CA"]].astype(np.float32)
            * self.atom37_gt_exists[..., atom37_atom_to_idx["C"]].astype(np.float32)
        )
        # rigid grouped features
        rigidgroups_features = all_atom.atom37_to_frames(
            np.argmax(self.aatype, axis=-1),
            self.atom37_positions,
            self.atom37_gt_exists & self.atom37_atom_exists,
        )

        features.update(rigidgroups_features)

        return features


def protein_structure_from_pdb_string(
    pdb_str: str, chain_id: Optional[str] = None
) -> ProteinStructureSample:
    """Takes a PDB string and constructs a ProteinStructureSample object.

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
    all_atoms_exist = []

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
            res_name = residue_constants.restype_1to3.get(res_shortname, "UNK")
            restype_idx = residue_constants.restype_order.get(
                res_shortname, residue_constants.restype_num
            )
            pos = np.zeros((residue_constants.atom_type_num, 3))
            mask = np.zeros((residue_constants.atom_type_num,))
            res_b_factors = np.zeros((residue_constants.atom_type_num,))

            atoms_exist = residue_constants.res_atom37_exist.get(res_name)

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
            all_atoms_exist.append(np.asarray(atoms_exist))

    return ProteinStructureSample(
        chain_id=chain_id,
        nb_residues=len(atom_positions),
        aatype=jax.nn.one_hot(x=np.asarray(aatype), num_classes=21),
        atom37_positions=np.asarray(atom_positions),  # n, 37, 3
        atom37_gt_exists=np.asarray(atom_mask).astype(bool),
        atom37_atom_exists=np.asarray(all_atoms_exist).astype(bool),
        resolution=0.0,
        pdb_cluster_size=1,
    )


def onehot_to_sequence(one_hot_encoding: np.ndarray) -> str:
    """
    Maps a one-hot encoding to a sequence of amino acids

    Args:
        one_hot_encoding: np.array of type np.bool with shape (*, 21)

    Returns:
      The amino acid sequence
    """
    assert len(one_hot_encoding.shape) == 2
    assert one_hot_encoding.shape[1] == residue_constants.restype_num_with_x, str(
        one_hot_encoding.shape
    )
    assert np.all(np.sum(one_hot_encoding.astype(np.uint16), axis=1) == 1)
    residue_id_encoding = np.where(one_hot_encoding)[1]
    return "".join(
        [
            residue_constants.restypes_with_x[residue_id]
            for residue_id in residue_id_encoding
        ]
    )
