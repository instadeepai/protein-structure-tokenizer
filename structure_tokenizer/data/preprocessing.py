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


import jax
import numpy as np

from structure_tokenizer.data import residue_constants
from structure_tokenizer.data.protein_structure_sample import ProteinStructureSample
from structure_tokenizer.model import quat_affine
from structure_tokenizer.types import BatchDataVQ3D, ProteinGraph
from structure_tokenizer.utils.protein_utils import (
    compute_nearest_neighbors_graph,
    protein_align_unbound_and_bound,
)


def filter_out_sample(
    sample: ProteinStructureSample,
    min_number_valid_residues: int,
    max_number_residues: int,
) -> bool:
    missing_coords_residue_mask = sample.get_missing_backbone_coords_mask()
    num_residue_coords_known = np.sum(~missing_coords_residue_mask)
    return bool(
        num_residue_coords_known < min_number_valid_residues
        or sample.nb_residues > max_number_residues
    )


def preprocess_sample(
    sample: ProteinStructureSample,
    num_neighbor: int,
    downsampling_ratio: int,
    residue_loc_is_alphac: bool,
    padding_num_residue: int,
    crop_index: int,
    noise_level: float,
) -> BatchDataVQ3D:
    """
        Util fn to preprocess a ProteinStructure readily for batching

    :param sample: instance to process, extract data from and pad
    :param num_neighbor: num of neighbor to consider to construct the graph
    :param residue_loc_is_alphac: whether we consider
    :param padding_num_residue: maximal number of residue to consider in the
        protein backbone

    :return:
        BatchDataVQ3D
            .graph
            .rigid_transformation
            .rigid_transformation
            .backbone_affine_mask
    """

    # Building protein graph
    atom37_coords = sample.atom37_positions

    # carefully, this can be tricky !
    atom37_mask = sample.atom37_gt_exists & sample.atom37_atom_exists
    missing_coords_residue_mask = sample.get_missing_backbone_coords_mask()

    num_residues_with_coords = np.sum(~missing_coords_residue_mask)

    # Local frames
    # BB atoms index
    (
        n_index,
        ca_index,
        c_index,
    ) = [residue_constants.atom_order[a] for a in ("N", "CA", "C")]
    # Frame rot
    (
        rot,
        _,
    ) = quat_affine.make_transform_from_reference(
        n_xyz=sample.atom37_positions[:, n_index, :],
        ca_xyz=sample.atom37_positions[:, ca_index, :],
        c_xyz=sample.atom37_positions[:, c_index, :],
    )
    # Get frame axis
    [u_i_feat, v_i_feat, n_i_feat] = np.split(rot, 3, axis=-1)
    u_i_feat = u_i_feat[..., 0]
    v_i_feat = v_i_feat[..., 0]
    n_i_feat = n_i_feat[..., 0]

    # Remove unobserved coordinatesatom37_positions
    (
        n_i_feat,
        u_i_feat,
        v_i_feat,
        atom37_coords,
        atom37_mask,
        aatype,
    ) = jax.tree_util.tree_map(
        lambda x: x[~missing_coords_residue_mask],
        (
            n_i_feat,
            u_i_feat,
            v_i_feat,
            atom37_coords,
            atom37_mask,
            sample.aatype,
        ),
    )

    # crop
    crop_start_idx = (
        0
        if num_residues_with_coords <= crop_index
        else np.random.randint(0, num_residues_with_coords - crop_index)
    )

    (
        n_i_feat,
        u_i_feat,
        v_i_feat,
        atom37_coords,
        atom37_mask,
        aatype,
    ) = jax.tree_util.tree_map(
        lambda x: x[crop_start_idx : crop_start_idx + crop_index],
        (
            n_i_feat,
            u_i_feat,
            v_i_feat,
            atom37_coords,
            atom37_mask,
            aatype,
        ),
    )

    res_representatives_loc_feat = (
        atom37_coords[:, ca_index]
        if residue_loc_is_alphac
        else np.mean(atom37_coords, axis=1, where=atom37_mask)
    )

    # Align unbound and bound structures, if needed
    if not residue_loc_is_alphac:
        (
            res_representatives_loc_feat,
            n_i_feat,
            u_i_feat,
            v_i_feat,
        ) = protein_align_unbound_and_bound(
            stacked_residue_representatives_coordinates=res_representatives_loc_feat,
            protein_n_i_feat=n_i_feat,
            protein_u_i_feat=u_i_feat,
            protein_v_i_feat=v_i_feat,
            alphac_atom_coordinates=atom37_coords[:, ca_index],
        )

    # Build the k-NN graph
    num_residues_with_coords = np.minimum(num_residues_with_coords, crop_index)
    n_neighbor = num_residues_with_coords if num_neighbor == -1 else num_neighbor
    list_atom_coordinates = [
        atom37_coords[i, atom37_mask[i]] for i in range(num_residues_with_coords)
    ]

    (
        n_node,
        n_edge,
        nodes_x,
        edges_features,
        senders,
        receivers,
    ) = compute_nearest_neighbors_graph(
        protein_num_residues=num_residues_with_coords,
        list_atom_coordinates=list_atom_coordinates,
        stacked_residue_coordinates=res_representatives_loc_feat,
        protein_n_i_feat=n_i_feat,
        protein_u_i_feat=u_i_feat,
        protein_v_i_feat=v_i_feat,
        num_neighbor=n_neighbor,
        noise_level=noise_level,
    )

    # Padding for batching
    nodes_mask = np.ones((n_node,), dtype=bool)
    n_pad_after = max(padding_num_residue - n_node, 0)
    # 0-pad below
    (
        nodes_x,
        nodes_mask,
        aatype,
    ) = jax.tree_map(
        lambda x: np.pad(
            x[:padding_num_residue],
            ((0, n_pad_after), *((0, 0) for _ in x.shape[1:])),
        ),
        (
            nodes_x,
            nodes_mask,
            aatype,
        ),
    )

    # token padding when downsampling structure to sequence with fixed ratio
    max_token_num = int(padding_num_residue / downsampling_ratio)
    token_num = int(n_node / downsampling_ratio)
    tokens_mask = np.ones((token_num,), dtype=bool)
    tokens_n_pad = max(max_token_num - token_num, 0)
    tokens_mask = np.pad(tokens_mask[:max_token_num], (0, tokens_n_pad))

    # pad edges features
    if num_neighbor < 0:
        padding_num_edges = padding_num_residue * padding_num_residue
    else:
        padding_num_edges = n_neighbor * padding_num_residue
    n_pad_after = max(padding_num_edges - n_edge, 0)
    edges_features = np.pad(
        edges_features[:padding_num_edges], ((0, n_pad_after), (0, 0))
    )

    # define PADDED graphs senders and receivers
    if num_residues_with_coords < num_neighbor or num_neighbor < 0:
        # BG: helper to pad senders and receivers when fully connected graph
        n_pad_after = max(n_neighbor - n_node, 0)

        def pad_directed_edges(x):
            edge_m = np.reshape(x, (n_node, -1))
            edge_padded_after = np.pad(
                edge_m[:, :n_neighbor],
                ((0, 0), (0, n_pad_after)),
                mode="constant",
                constant_values=n_node,
            )
            edge_padded_below = np.concatenate(
                [  # shape (padding_num_residue, n_neighbor)
                    edge_padded_after,
                    np.repeat(
                        np.arange(edge_padded_after.shape[0], padding_num_residue)[
                            :, None
                        ],
                        n_neighbor,
                        axis=-1,
                    ),
                ],
                axis=0,
            )[:padding_num_residue]

            return edge_padded_below

        (senders, receivers) = jax.tree_map(
            lambda x: pad_directed_edges(x).flatten(),
            (senders, receivers),
        )
    else:
        (senders, receivers) = jax.tree_map(
            lambda x: np.concatenate(  # BG: shape (padding_num_residue, num_neighbor)
                [
                    np.array(x),
                    np.repeat(np.arange(n_node, padding_num_residue), n_neighbor),
                ],
                axis=0,
            )[:padding_num_edges],
            (senders, receivers),
        )

    protein_graph = ProteinGraph(  # type: ignore
        n_node=np.expand_dims(n_node, axis=-1),
        n_edge=np.expand_dims(n_edge, axis=-1),
        nodes_mask=np.expand_dims(nodes_mask, axis=-1),
        nodes_original_coordinates=nodes_x,
        node_features=nodes_x,
        edge_features=edges_features,
        tokens_mask=np.expand_dims(tokens_mask, axis=-1),
        senders=senders,
        receivers=receivers,
    )

    # Building protein features for SM loss

    features = sample.make_protein_features()
    # Remove unobserved coordinates
    features = jax.tree_util.tree_map(
        lambda x: x[~sample.get_missing_backbone_coords_mask()], features
    )
    # crop
    features = jax.tree_util.tree_map(
        lambda x: x[crop_start_idx : crop_start_idx + crop_index], features
    )
    # Padding for batching
    n_pad_after = max(padding_num_residue - n_node, 0)
    # 0-pad below
    features = jax.tree_map(
        lambda x: np.pad(
            x[:padding_num_residue],
            ((0, n_pad_after), *((0, 0) for _ in x.shape[1:])),
        ),
        features,
    )
    features["nb_residues"] = n_node

    return BatchDataVQ3D(  # type: ignore
        graph=protein_graph,
        features=features,
    )
