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

import functools
import math
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from structure_tokenizer.types import EdgeFeatures, NodeFeatures


class PositionalEncodingLayer(hk.Module):
    """Independent-Equivariant Graph Matching Network"""

    def __init__(self, positional_encoding_dimension: int):
        """Initializes a Positional Encoding Layer

        Args:
            positional_encoding_dimension (int): dimension of positional embedding
        """

        super(PositionalEncodingLayer, self).__init__()
        self.positional_encoding_dimension = positional_encoding_dimension

        # Create random orthogonal matrix

        matrix1 = np.random.randn(
            positional_encoding_dimension, positional_encoding_dimension
        )
        u, s, vh = np.linalg.svd(matrix1, full_matrices=False)
        matrix2 = u @ vh
        self.orthogonal_matrix = matrix2 @ matrix2.T

    def sinusoidal_positional_encoding(
        self, x: int, n: int, d: int, k: int
    ) -> jnp.float32:
        """Computes positional encoding for two indices x and k

        Args:
            x (int): position index
            n (int): number of indices
            d (int): number of feature dimensions
            k (int): position index

        Returns:
            sinusoidal positional encoding (jnp.float32)
        """

        return jnp.mod(k, 2) * jnp.cos((x * math.pi) / n ** (2 * (k - 1) / d)) - (
            jnp.mod(k, 2) - 1
        ) * jnp.sin((x * math.pi) / n ** (2 * k / d))

    def sinusoidal_positional_encoding_features(
        self, indice_i: int, indice_j: int, number_residues: int
    ) -> jnp.array:
        """Computes positional encoding vector for two indices i and j

        Args:
            indice_i (int): position index
            indice_j (int): position index
            number_residues (int): number of residue indices

        Returns:
            sinusoidal positional encoding vector (jnp.array)
        """

        difference_indices = indice_i - indice_j
        list_indices = jnp.arange(1, self.positional_encoding_dimension + 1)

        sinusoidal_fn = functools.partial(
            self.sinusoidal_positional_encoding,
            difference_indices,
            number_residues,
            self.positional_encoding_dimension,
        )
        return jax.vmap(sinusoidal_fn)(list_indices)

    def __call__(
        self,
        n_node: int,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        diffusion: bool,
        diffusion_time_step: int,
    ) -> Tuple[NodeFeatures, EdgeFeatures]:
        """Computes positional embeddings for nodes and edges of a give protein

        Args:
            n_node (int):  number of nodes
            senders (jnp.ndarray): origins of graph vertices
            receivers (jnp.ndarray): destinations of graph vertices
            diffusion (bool): whether we perform diffusion
            diffusion_time_step (int): diffusion time step

        Returns:
            node_positional_encoding (NodeFeatures): node positional embeddings
            edge_positional_encoding (EdgeFeatures): edge positional embeddings
        """

        # Build nodes positional embeddings

        node_sinusoidal_fn = functools.partial(
            self.sinusoidal_positional_encoding_features,
            indice_j=0,
            number_residues=n_node,
        )

        nodes_positional_embeddings = jax.vmap(node_sinusoidal_fn)(jnp.arange(n_node))

        if diffusion:
            diffusion_embedding = jnp.matmul(
                jax.vmap(node_sinusoidal_fn)(diffusion_time_step * jnp.ones(n_node)),
                self.orthogonal_matrix,
            )
            nodes_positional_embeddings = (
                nodes_positional_embeddings + diffusion_embedding
            )

        # Build edges positional embeddings

        edge_sinusoidal_fn = functools.partial(
            self.sinusoidal_positional_encoding_features,
            number_residues=n_node,
        )
        # def vmap_edge_sinusoidal_fn(senders, receivers):
        #     return jax.vmap(edge_sinusoidal_fn)(
        #         indice_i=senders, indice_j=receivers
        #     )
        # edges_positional_embeddings=jax.vmap(vmap_edge_sinusoidal_fn)(
        #     senders=senders, receivers=receivers
        # )
        edges_positional_embeddings = jax.vmap(edge_sinusoidal_fn)(
            indice_i=senders, indice_j=receivers
        )
        return nodes_positional_embeddings, edges_positional_embeddings

    def __repr__(self) -> str:
        return "PositionalEncodingLayer" + str(self.__dict__)
