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

from typing import Optional, Tuple

import haiku as hk
import jax.numpy as jnp
from ml_collections import ConfigDict

from structure_tokenizer.model import prng
from structure_tokenizer.model.modules import GraphNeuralNetwork
from structure_tokenizer.model.positional_encoding_layer import PositionalEncodingLayer
from structure_tokenizer.model.utils import build_initializer
from structure_tokenizer.types import EdgeFeatures, NodeFeatures, ProteinGraph, RNGKey


class StructureEncoder(hk.Module):
    """Structure encoder"""

    def __init__(
        self,
        config: ConfigDict,
        global_config: ConfigDict,
    ):
        """Initializes a structure encoder Network

        Args:
            config (ConfigDict): model hyperparameters
            global_config (ConfigDict): data processing configuration
        """

        super(StructureEncoder, self).__init__()
        self.config = config
        self.global_config = global_config

        if self.config.use_gnn_encoder:
            self.encoder = GraphNeuralNetwork(
                config=self.config.gnn,
                global_config=self.global_config,
            )
        else:
            raise ValueError("Encoder network not recognized")

    def __call__(
        self, graph: ProteinGraph, is_training: bool, safe_key: Optional[RNGKey] = None
    ) -> Tuple[EdgeFeatures, NodeFeatures]:
        """Computes embedding of a protein graph

        Args:
            graph (Dict): graph representing a protein

        Returns:
            graph_embedding (Dict): graph representing protein graph embedding
        """

        if safe_key is None:
            safe_key = prng.SafeKey(hk.next_rng_key())
        safe_key, safe_subkey = safe_key.split()

        w_init = build_initializer(
            distribution=self.global_config.init.distribution,
            scale=self.global_config.init.scale,
            mode=self.global_config.init.mode,
        )

        pos_enc_dim = self.config.positional_encoding_dimension
        (
            node_positional_embedding,
            edge_positional_embedding,
        ) = PositionalEncodingLayer(positional_encoding_dimension=pos_enc_dim)(
            n_node=self.global_config.data.seq_max_size,
            senders=graph.senders,
            receivers=graph.receivers,
            diffusion=False,
            diffusion_time_step=0,
        )

        node_proj = hk.Linear(
            self.config.encoding_dimension, name="init_node_embed", w_init=w_init
        )
        protein_node_features = node_proj(node_positional_embedding)
        # edges features
        protein_edge_features = jnp.concatenate(
            [
                edge_positional_embedding,
                graph.edge_features,
            ],
            axis=-1,
        )

        edge_proj = hk.Linear(
            self.config.encoding_dimension, name="init_edge_embed", w_init=w_init
        )
        protein_edge_features = edge_proj(protein_edge_features)

        if self.config.use_gnn_encoder:
            # GNN encoder expect a graph as input
            graph_protein = ProteinGraph(  # type: ignore
                n_node=graph.n_node,
                n_edge=graph.n_edge,
                nodes_mask=graph.nodes_mask,
                nodes_original_coordinates=graph.nodes_original_coordinates,
                node_features=protein_node_features,
                edge_features=protein_edge_features,
                tokens_mask=graph.tokens_mask,
                senders=graph.senders,
                receivers=graph.receivers,
            )
            return self.encoder(graph_protein, is_training)

        else:
            raise ValueError("Encoder network not recognized")

    def __repr__(self) -> str:
        return "StructureEncoder " + str(self.__dict__)
