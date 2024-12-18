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


from typing import List, Tuple

import jax
import numpy as np
import scipy.spatial as spa
from biopandas.pdb import PandasPdb

from structure_tokenizer.types import (
    Coordinates,
    EdgeFeatures,
    Residue,
    RotationMatrix,
    TranslationVector,
)


def get_CA_coords(pdb_file: str) -> Coordinates:  # noqa: N802
    """
    Gets coordinates of CA atom for each residue of the sequence

    Args:
        pdb_file (str): path of pdb file

    Returns:
        (Coordinates): 3D coordinates of CA atoms
    """
    ppdb_model = PandasPdb().read_pdb(pdb_file)
    df = ppdb_model.df["ATOM"]
    df = df[df["atom_name"] == "CA"]
    return np.array(
        df[["x_coord", "y_coord", "z_coord"]].to_numpy().squeeze().astype(np.float32)
    )


def rigid_transform_kabsch_3d(
    a: Coordinates, b: Coordinates
) -> Tuple[RotationMatrix, TranslationVector]:
    """Applies Kabsch algorithm: it find the right rotation/translation to move
        a point cloud a_1...N to another point cloud b_1...N`

    Args:
        a (Coordinates): 3D point cloud
        b (Coordinates): 3D point cloud

    Raises:
        Exception: if data point cloud a has wrong size
        Exception: if data point cloud b has wrong size

    Returns:
        r: rotation matrix
        t: translation vector
    """
    # find mean column wise: 3 x 1

    centroid_a = jax.device_put(
        np.mean(a, axis=1, keepdims=True), device=jax.devices("cpu")[0]
    )
    centroid_b = jax.device_put(
        np.mean(b, axis=1, keepdims=True), device=jax.devices("cpu")[0]
    )

    # subtract mean
    am = a - centroid_a
    bm = b - centroid_b

    h = am @ bm.T

    # find rotation
    u, s, vt = np.linalg.svd(h)

    r = vt.T @ u.T

    # special reflection case
    if np.linalg.det(r) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        ss = np.diag(np.array([1.0, 1.0, -1.0]))
        r = (vt.T @ ss) @ u.T

    t = -r @ centroid_a + centroid_b
    return r, t


def residue_embedding(residue: Residue) -> int:
    """
    Returns residue index (between 0 and 20)

    Args:
        residue (Residue): residue sequence

    Returns:
        index (int)
    """

    dit = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
        "HIP": "H",
        "HIE": "H",
        "TPO": "T",
        "HID": "H",
        "LEV": "L",
        "MEU": "M",
        "PTR": "Y",
        "GLV": "E",
        "CYT": "C",
        "SEP": "S",
        "HIZ": "H",
        "CYM": "C",
        "GLM": "E",
        "ASQ": "D",
        "TYS": "Y",
        "CYX": "C",
        "GLZ": "G",
    }

    rare_residues = {
        "HIP": "H",
        "HIE": "H",
        "TPO": "T",
        "HID": "H",
        "LEV": "L",
        "MEU": "M",
        "PTR": "Y",
        "GLV": "E",
        "CYT": "C",
        "SEP": "S",
        "HIZ": "H",
        "CYM": "C",
        "GLM": "E",
        "ASQ": "D",
        "TYS": "Y",
        "CYX": "C",
        "GLZ": "G",
    }

    if residue in rare_residues.keys():
        print("Some rare residue: ", residue)

    indicator = {
        "Y": 0,
        "R": 1,
        "F": 2,
        "G": 3,
        "I": 4,
        "V": 5,
        "A": 6,
        "W": 7,
        "E": 8,
        "H": 9,
        "C": 10,
        "N": 11,
        "M": 12,
        "D": 13,
        "T": 14,
        "S": 15,
        "K": 16,
        "L": 17,
        "Q": 18,
        "P": 19,
    }
    res_name = residue
    if res_name not in dit.keys():
        return 20
    else:
        res_name = dit[res_name]
        return indicator[res_name]


def name_residue(index: int) -> Residue:
    """Return index of residue for Equidock data

    Args:
        index: index of residue

    Returns:
        residue (str): residue name
    """
    list_residues = [
        "Y",
        "R",
        "F",
        "G",
        "I",
        "V",
        "A",
        "W",
        "E",
        "H",
        "C",
        "N",
        "M",
        "D",
        "T",
        "S",
        "K",
        "L",
        "Q",
        "P",
    ]
    if index in range(20):
        return list_residues[index]
    else:
        return ""  # TODO What is the last amino acid ?


def residue_list_featurizer(
    residue_list: List[str],
) -> np.ndarray:
    """
    Creates residual features given PDB data as input.
    A residue is simply embedded by an index.

    Args:
        residue_list (List[str]): list of 3-letter amino-acid codes

    Returns:
        Residual features
    """
    feature_list = [[residue_embedding(residue)] for residue in residue_list]
    feature_list = np.array(feature_list).astype(np.int32)
    return feature_list


def distance_list_featurizer(dist_list: np.ndarray) -> np.ndarray:
    """Computes graph features based on the distance between residues

    Args:
        dist_list (List[float]): list of distances between residues

    Returns:
        np.ndarray: distance features
    """
    length_scale_list = [1.5**x for x in range(15)]
    center_list = [0.0 for _ in range(15)]

    num_edge = len(dist_list)
    dist_list = np.array(dist_list)

    transformed_dist = [
        np.exp(-((dist_list - center) ** 2) / float(length_scale))
        for length_scale, center in zip(length_scale_list, center_list)
    ]

    transformed_dist = (
        (np.array(transformed_dist).T).reshape((num_edge, -1)).astype(np.float32)
    )

    return transformed_dist


def protein_align_unbound_and_bound(
    stacked_residue_representatives_coordinates: Coordinates,
    protein_n_i_feat: np.ndarray,
    protein_u_i_feat: np.ndarray,
    protein_v_i_feat: np.ndarray,
    alphac_atom_coordinates: Coordinates,
) -> Tuple[Coordinates, np.ndarray, np.ndarray, np.ndarray]:
    """
    Aligns the bound and unbound structures.
    In the bound structure, the residue coordinate
    is defined by the centroid coordinates of the atoms of. In the unbound structure,
    the residue coordinates are defined by the alpha-c coordinates.

    Args:
        stacked_residue_representatives_coordinates (Coordinates): coordinates of
            atoms centroids of residues
        protein_n_i_feat (np.ndarray): protein features
        protein_u_i_feat (np.ndarray): protein features
        protein_v_i_feat (np.ndarray): protein features
        alphac_atom_coordinates (Coordinates): coordinates of alpha-c atoms
    """

    ret_r_protein, ret_t_protein = rigid_transform_kabsch_3d(
        stacked_residue_representatives_coordinates.T,
        alphac_atom_coordinates.T,
    )
    list_residue_representatives_coordinates = (
        (ret_r_protein @ (stacked_residue_representatives_coordinates).T)
        + ret_t_protein
    ).T
    protein_n_i_feat = (ret_r_protein @ (protein_n_i_feat).T).T
    protein_u_i_feat = (ret_r_protein @ (protein_u_i_feat).T).T
    protein_v_i_feat = (ret_r_protein @ (protein_v_i_feat).T).T
    return (
        list_residue_representatives_coordinates,
        protein_n_i_feat,
        protein_u_i_feat,
        protein_v_i_feat,
    )


def compute_nearest_neighbors_graph(
    protein_num_residues: int,
    list_atom_coordinates: List[Coordinates],
    # residue_list: List[str],
    stacked_residue_coordinates: Coordinates,
    protein_n_i_feat: np.ndarray,
    protein_u_i_feat: np.ndarray,
    protein_v_i_feat: np.ndarray,
    num_neighbor: int,
    noise_level: float = 0.0,
) -> Tuple[int, int, Coordinates, EdgeFeatures, np.ndarray, np.ndarray]:
    """Computes kNN graph based on residues coordinates

    Args:
        protein_num_residues (int): number of residues
        list_atom_coordinates (List[Coordinates]):
            atom coordinates in residues
        residue_list (List[str]): list of 3-letter amino-acid codes
        stacked_residue_coordinates (Coordinates):
            residue coordinates
        protein_n_i_feat (np.ndarray): residues features
        protein_u_i_feat (np.ndarray): residues features
        protein_v_i_feat (np.ndarray): residues features
        num_neighbor (int): maximum number of nearest neighbors
        noise_level: float

    Returns:
        n_node (int): number of nodes
        n_edge (int): number of edges
        nodes_x (Coordinates): nodes coordinates
        nodes_res_feat (NodeFeatures): residual features
        edges_features (EdgeFeatures): edge features
        senders (np.ndarray): indexes of edge senders
        receivers (np.ndarray): indexes of edges receivers
    """

    assert protein_num_residues == stacked_residue_coordinates.shape[0]
    assert stacked_residue_coordinates.shape[1] == 3

    n_node = protein_num_residues
    if protein_num_residues <= num_neighbor:
        # Seq can now be sharter than NN
        num_neighbor = protein_num_residues

    n_edge = num_neighbor * protein_num_residues

    # First  compute edges features
    # Residues coordinates set to atoms coordinates mean
    means_of_atom_coordinates = np.stack(
        [
            np.mean(per_res_atom_coordinates, axis=0)
            for per_res_atom_coordinates in list_atom_coordinates
        ]
    )
    # Compute pairwise distances
    protein_distance = spa.distance.cdist(
        means_of_atom_coordinates + np.random.normal(0, noise_level),
        means_of_atom_coordinates + np.random.normal(0, noise_level),
    )
    # Get node source and destination ordered by distance (destination centered)
    if num_neighbor == n_node:
        # Fully connected graph with self connections
        valid_src = np.argsort(protein_distance, axis=-1)
    else:
        valid_src = np.argsort(protein_distance, axis=-1)[:, 1 : (num_neighbor + 1)]
    valid_dst = np.repeat(
        np.arange(protein_num_residues)[..., np.newaxis], num_neighbor, axis=-1
    )
    # Get distances for kNN
    valid_dist = np.stack(
        [protein_distance[i, valid_src[i]] for i in range(protein_num_residues)]
    )
    # Get senders, receivers idx and corresponding distances
    senders = valid_src.flatten()
    receivers = valid_dst.flatten()
    protein_dist_list = list(valid_dist.flatten())

    # Get distance edges features
    edges_features = distance_list_featurizer(protein_dist_list)

    # Build the various p_ij, q_ij, k_ij, t_ij pairs
    basis_matrices = np.stack(
        [protein_n_i_feat, protein_u_i_feat, protein_v_i_feat], axis=1
    )
    stacked_res = stacked_residue_coordinates
    diff_stacked_res = stacked_res[:, np.newaxis, :] - stacked_res[np.newaxis, :, :]
    p_ij = np.einsum(
        "ijk,nik->inj", basis_matrices, diff_stacked_res
    )  # position edges features
    q_ij = np.einsum(
        "ijk,nk->inj", basis_matrices, protein_n_i_feat
    )  # orientation edges features 1
    k_ij = np.einsum(
        "ijk,nk->inj", basis_matrices, protein_u_i_feat
    )  # orientation edges features 1
    t_ij = np.einsum(
        "ijk,nk->inj", basis_matrices, protein_v_i_feat
    )  # orientation edges features 1
    s_ij = np.concatenate([p_ij, q_ij, k_ij, t_ij], axis=-1)
    # Get edges features for valid (send, reveiver) pairs
    protein_edge_feat_ori_list = [
        s_ij[receivers[i], senders[i]] for i in range(len(protein_dist_list))
    ]
    protein_edge_feat_ori_feat = np.stack(
        protein_edge_feat_ori_list, axis=0
    )  # shape (num_edges, 4, 3)

    edges_features = np.concatenate(
        [edges_features, protein_edge_feat_ori_feat], axis=1
    )

    nodes_x = stacked_residue_coordinates

    return (n_node, n_edge, nodes_x, edges_features, senders, receivers)
