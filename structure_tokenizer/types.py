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

from typing import Any, Dict, Tuple, TypedDict

import jax.numpy as jnp
import jax_dataclasses as jdc
import pandas
from typing_extensions import NotRequired, TypeAlias

Mask: TypeAlias = jnp.ndarray
Coordinates: TypeAlias = jnp.ndarray
Loss: TypeAlias = jnp.float32
RNGKey: TypeAlias = jnp.ndarray
Residue: TypeAlias = str
Sequence: TypeAlias = str
NodeFeatures: TypeAlias = jnp.ndarray
EdgeFeatures: TypeAlias = jnp.ndarray
PdbRawData: TypeAlias = pandas.DataFrame
SortedPdbData: TypeAlias = Tuple[Tuple[Residue, int, Residue], PdbRawData]
TranslationVector: TypeAlias = jnp.ndarray
RotationMatrix: TypeAlias = jnp.ndarray
Metrics: TypeAlias = Tuple[Any, ...]
SingleResidueRepresentation: TypeAlias = jnp.ndarray
PairResidueRepresentation: TypeAlias = jnp.ndarray

# Codebook Variables
Structure: TypeAlias = Dict[str, jnp.ndarray]
Codebook: TypeAlias = jnp.ndarray
Quantized: TypeAlias = jnp.ndarray
StraightThroughQuantized: TypeAlias = jnp.ndarray
CodebookMetric: TypeAlias = jnp.ndarray
Latents: TypeAlias = jnp.ndarray


# @dataclass
@jdc.pytree_dataclass
class ProteinGraph:
    """
    n_node (jnp.ndarray): number of nodes
    n_edge (jnp.ndarray): number of edges
    nodes_original_coordinates (Coordinates): original coordinates (i.e. coordinates
        at the beginning of the message passing process. This corresponds to
        rotated coordinates of the protein)
    nodes_ground_truth_coordinates (Coordinates): ground-truth coordinates
    node_embeddings (jnp.ndarray): nodes embeddings
    nodes_mask (Mask): mask indicating which residues are padded
    tokens_mask (Mask): mask indicating which tokens in the emb. seq. are padded
    node_embeddings (jnp.ndarray): edges embeddings
    original_edge_features (EdgeFeatures): original edge features (
        cf Appendix of Equidock paper)
    senders (jnp.ndarray): origin of graph vertices
    receivers (jnp.ndarray): destination of graph vertices
    """

    n_node: jnp.ndarray
    n_edge: jnp.ndarray
    nodes_mask: Mask
    nodes_original_coordinates: Coordinates  # what are these? are they needed?
    node_features: NodeFeatures
    edge_features: EdgeFeatures
    tokens_mask: Mask
    senders: jnp.ndarray
    receivers: jnp.ndarray


# @dataclass
@jdc.pytree_dataclass
class BatchDataVQ3D:
    """Batch data of vq3d
    Args:
        graph (ProteinGraph): graph of the protein
    """

    graph: ProteinGraph
    features: dict


@jdc.pytree_dataclass
# @dataclass
class ProteinInteractionBatch:
    """Batch data of Protein Interaction task
    Args:
        batch_data_vq3d (BatchDataVQ3D): batch data for protein 1
        batch_data_vq3d_2 (BatchDataVQ3D): batch data for protein 2
        target (jnp.ndarray): target value indicating whtether the two
            proteins interact
    """

    batch_data_vq3d: BatchDataVQ3D
    batch_data_vq3d_2: BatchDataVQ3D
    target: jnp.ndarray


@jdc.pytree_dataclass
# @dataclass
class FunctionPredictionBatch:
    """Batch data of Function Prediction task
    Args:
        batch_data_vq3d (BatchDataVQ3D): batch data for protein
        target (jnp.ndarray): target array indicating the labels of protein
    """

    batch_data_vq3d: BatchDataVQ3D
    target: jnp.ndarray


@jdc.pytree_dataclass
# @dataclass
class ProteinFeatures:
    """Set of features that characterize a protein graph
    Args:
        predicted_coordinates (Coordinates): 3D coordinates of protein
        original_coordinates (Coordinates): original 3D coordinates of
            protein (does not change)
        predicted_embeddings (NodeFeatures): node embeddings that are
            updated at every layer iteration
        original_node_features (NodeFeatures): node features
        original_edge_features (EdgeFeatures): edge features
    """

    predicted_coordinates: Coordinates
    original_coordinates: Coordinates
    predicted_embeddings: NodeFeatures
    original_node_features: NodeFeatures
    original_edge_features: EdgeFeatures


class QuantizerOutput(TypedDict):
    usage_counter: NotRequired[CodebookMetric]
    key_x: NotRequired[RNGKey]  # was used for mmd
    key_y: NotRequired[RNGKey]
    distances: NotRequired[CodebookMetric]
    mixing_coeff: NotRequired[CodebookMetric]
    continuous_embedding: Latents
    continuous_embedding_pre_proj: NotRequired[Latents]
    quantize_post_proj: NotRequired[Latents]
    quantize: Quantized
    straight_through_quantized: StraightThroughQuantized
    soft_proba: CodebookMetric
    avg_proba: NotRequired[CodebookMetric]
    codebook: NotRequired[Codebook]
    perplexity: CodebookMetric
    tokens: Latents
