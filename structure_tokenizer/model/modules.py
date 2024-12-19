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
from typing import Any, Callable, List, Optional, Tuple, Dict

import haiku as hk
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
from typing_extensions import TypeAlias

from structure_tokenizer.model import common_modules, layer_stack, mapping, prng
from structure_tokenizer.model.gnn_layers import GNNLayer, MPNNLayer
from structure_tokenizer.model.positional_encoding_layer import PositionalEncodingLayer
from structure_tokenizer.model.utils import build_initializer
from structure_tokenizer.types import (
    EdgeFeatures,
    Mask,
    NodeFeatures,
    PairResidueRepresentation,
    ProteinGraph,
    RNGKey,
    SingleResidueRepresentation,
)
from structure_tokenizer.utils.log import get_logger

logger = get_logger(__name__)

Quantized: TypeAlias = jnp.ndarray
StraightThroughQuantized: TypeAlias = jnp.ndarray


class GraphNeuralNetwork(hk.Module):
    """Independent-Equivariant Graph Matching Network"""

    def __init__(
        self,
        config: ConfigDict,
        global_config: ConfigDict,
    ):
        """Initializes an Independent Equivariant Graph Matching Network

        Args:
            hyperparameters (BioClipConfig): model hyperparameters
            data_config (DataProcessingConfig): data processing configuration
        """

        super(GraphNeuralNetwork, self).__init__()
        self.config = config
        self.global_config = global_config

        layer_cls: Callable = {
            "GNNLayer": GNNLayer,
            "MPNNLayer": MPNNLayer,
        }[self.config.gnn_layer.layer_cls]

        self.gnn_layers: List[
            Callable[[ProteinGraph], Tuple[EdgeFeatures, NodeFeatures]]
        ] = [
            layer_cls(
                config=self.config.gnn_layer,
                global_config=self.global_config,
                graph_max_neighbor=self.global_config.data.graph_max_neighbor,
            )
        ]
        if self.config.shared_layers:
            intermediate_layer = layer_cls(  # BG: ProteinMPNN hard coded - to change
                config=self.config.gnn_layer,
                global_config=self.global_config,
                graph_max_neighbor=self.global_config.data.graph_max_neighbor,
            )
            for _ in range(1, self.config.gnn_number_layers):
                self.gnn_layers.append(intermediate_layer)

        else:
            for _ in range(1, self.config.gnn_number_layers):
                self.gnn_layers.append(
                    layer_cls(  # BG: ProteinMPNN hard coded - to change
                        config=self.config.gnn_layer,
                        global_config=self.global_config,
                        graph_max_neighbor=self.global_config.data.graph_max_neighbor,
                    )
                )

    def __call__(
        self, graph: ProteinGraph, is_training: bool = False, safe_key=None
    ) -> Tuple[NodeFeatures, EdgeFeatures]:
        """Computes embedding of a protein graph

        Args:
            graph (Dict):  graph representing a protein
            is_training: patch



        Returns:
            graph (Dict): graph representing protein graph embedding
        """

        if safe_key is None:
            safe_key = prng.SafeKey(hk.next_rng_key())
        safe_key, safe_subkey = safe_key.split()

        graph_protein = graph
        for _, layer in enumerate(self.gnn_layers):
            edge_features, node_features = layer(graph_protein)

            graph_protein = ProteinGraph(  # type: ignore
                n_node=graph.n_node,
                n_edge=graph.n_edge,
                nodes_mask=graph.nodes_mask,
                nodes_original_coordinates=graph.nodes_original_coordinates,
                node_features=node_features,
                edge_features=edge_features,
                tokens_mask=graph.tokens_mask,
                senders=graph.senders,
                receivers=graph.receivers,
            )

        return graph_protein.node_features, graph_protein.edge_features

    def __repr__(self) -> str:
        return "GraphNeuralNetwork " + str(self.__dict__)


def apply_dropout(
    *,
    tensor: jnp.ndarray,
    safe_key: RNGKey,
    rate: float,
    is_training: float,
    broadcast_dim=None,
):
    """Applies dropout to a tensor."""
    if is_training and rate != 0.0:
        shape = list(tensor.shape)
        if broadcast_dim is not None:
            shape[broadcast_dim] = 1
        keep_rate = 1.0 - rate
        keep = jax.random.bernoulli(safe_key.get(), keep_rate, shape=shape)
        # return keep * tensor / keep_rate
        # TB edit: faster and avoids casting a bool array.
        return jnp.where(keep, tensor / keep_rate, 0)
    else:
        return tensor


def dropout_wrapper(
    module: Any,
    input_act: jnp.ndarray,
    mask: Mask,
    safe_key: RNGKey,
    global_config: Any,
    output_act: Optional[jnp.ndarray] = None,
    is_training: bool = True,
    **kwargs,
):
    """Applies module + dropout + residual update."""
    if output_act is None:
        output_act = input_act

    gc = global_config
    residual = module(input_act, mask, is_training=is_training, **kwargs)
    dropout_rate = 0.0 if gc.deterministic else module.config.dropout_rate

    if module.config.shared_dropout:
        if module.config.orientation == "per_row":
            broadcast_dim = 0
        else:
            broadcast_dim = 1
    else:
        broadcast_dim = None

    residual = apply_dropout(
        tensor=residual,
        safe_key=safe_key,
        rate=dropout_rate,
        is_training=is_training,
        broadcast_dim=broadcast_dim,
    )

    new_act = output_act + residual

    return new_act


class Transition(hk.Module):
    """Transition layer.

    Jumper et al. (2021) Suppl. Alg. 9 "MSATransition"
    Jumper et al. (2021) Suppl. Alg. 15 "PairTransition"
    """

    def __init__(self, config, global_config, name="transition_block"):
        super().__init__(name=name)
        self.config = config
        self.global_config = global_config

    def __call__(self, act, mask, is_training=True):
        """Builds Transition module.

        Arguments:
          act: A tensor of queries of size [1, N_res, N_channel].
          mask: A tensor denoting the mask of size [1, N_res].
          is_training: Whether the module is in training mode.

        Returns:
          A float32 tensor of size [1, N_res, N_channel].
        """
        _, _, nc = act.shape

        num_intermediate = int(nc * self.config.num_intermediate_factor)
        # mask = jnp.expand_dims(mask, axis=-1)

        act = hk.LayerNorm(
            axis=[-1], create_scale=True, create_offset=True, name="input_layer_norm"
        )(act)

        w_init = build_initializer(
            distribution=self.global_config.init.distribution,
            scale=self.global_config.init.scale,
            mode=self.global_config.init.mode,
        )

        transition_module = hk.Sequential(
            [
                common_modules.Linear(
                    num_intermediate,
                    initializer="relu",
                    name="transition1",
                    w_init=w_init,
                ),
                jax.nn.relu,
                common_modules.Linear(
                    nc,
                    w_init=w_init,
                    name="transition2",
                ),
            ]
        )

        act = mapping.inference_subbatch(
            transition_module,
            self.config.subbatch_size,
            batched_args=[act],
            nonbatched_args=[],
            low_memory=not is_training,
        )

        return act


def glorot_uniform():
    return hk.initializers.VarianceScaling(
        scale=1.0, mode="fan_avg", distribution="uniform"
    )


class Attention(hk.Module):
    """Multihead attention."""

    def __init__(self, config, global_config, output_dim, name="attention"):
        super().__init__(name=name)

        self.config = config
        self.global_config = global_config
        self.output_dim = output_dim

    def __call__(self, q_data, m_data, bias, nonbatched_bias=None):
        """Builds Attention module.

        Arguments:
          q_data: A tensor of queries, shape [batch_size, N_queries, q_channels].
          m_data: A tensor of memories from which the keys and values are
            projected, shape [batch_size, N_keys, m_channels].
          bias: bias for the attention, shape [batch_size, num_heads, N_queries, N_keys]
          nonbatched_bias: Shared bias, shape [N_queries, N_keys].

        Returns:
          A float32 tensor of shape [batch_size, N_queries, output_dim].
        """
        # Sensible default for when the config keys are missing
        key_dim = self.config.get("key_dim", int(q_data.shape[-1]))
        value_dim = self.config.get("value_dim", int(m_data.shape[-1]))
        num_head = self.config.num_head
        assert key_dim % num_head == 0
        assert value_dim % num_head == 0
        key_dim = key_dim // num_head
        value_dim = value_dim // num_head

        q_weights = hk.get_parameter(
            "query_w",
            shape=(q_data.shape[-1], num_head, key_dim),
            # init=glorot_uniform(),
            init=build_initializer(
                distribution=self.global_config.init.distribution,
                scale=self.global_config.init.scale,
                mode=self.global_config.init.mode,
            ),
        )
        k_weights = hk.get_parameter(
            "key_w",
            shape=(m_data.shape[-1], num_head, key_dim),
            # init=glorot_uniform(),
            init=build_initializer(
                distribution=self.global_config.init.distribution,
                scale=self.global_config.init.scale,
                mode=self.global_config.init.mode,
            ),
        )
        v_weights = hk.get_parameter(
            "value_w",
            shape=(m_data.shape[-1], num_head, value_dim),
            # init=glorot_uniform(),
            init=build_initializer(
                distribution=self.global_config.init.distribution,
                scale=self.global_config.init.scale,
                mode=self.global_config.init.mode,
            ),
        )

        q = jnp.einsum("bqa,ahc->bqhc", q_data, q_weights) * key_dim ** (-0.5)
        k = jnp.einsum("bka,ahc->bkhc", m_data, k_weights)
        v = jnp.einsum("bka,ahc->bkhc", m_data, v_weights)
        logits = jnp.einsum("bqhc,bkhc->bhqk", q, k) + bias
        if nonbatched_bias is not None:
            logits += jnp.expand_dims(nonbatched_bias, axis=0)
        weights = jax.nn.softmax(logits)
        weighted_avg = jnp.einsum("bhqk,bkhc->bqhc", weights, v)

        if self.global_config.zero_init:
            init = hk.initializers.Constant(0.0)
        else:
            # init = glorot_uniform()
            init = build_initializer(
                distribution=self.global_config.init.distribution,
                scale=self.global_config.init.scale,
                mode=self.global_config.init.mode,
            )

        if self.config.gating:
            gating_weights = hk.get_parameter(
                "gating_w",
                shape=(q_data.shape[-1], num_head, value_dim),
                init=hk.initializers.Constant(0.0),
            )
            gating_bias = hk.get_parameter(
                "gating_b",
                shape=(num_head, value_dim),
                init=hk.initializers.Constant(1.0),
            )

            gate_values = (
                jnp.einsum("bqc, chv->bqhv", q_data, gating_weights) + gating_bias
            )

            gate_values = jax.nn.sigmoid(gate_values)

            weighted_avg *= gate_values

        o_weights = hk.get_parameter(
            "output_w", shape=(num_head, value_dim, self.output_dim), init=init
        )
        o_bias = hk.get_parameter(
            "output_b", shape=(self.output_dim,), init=hk.initializers.Constant(0.0)
        )

        output = jnp.einsum("bqhc,hco->bqo", weighted_avg, o_weights) + o_bias

        return output


class CrossAttention(hk.Module):
    """Single rows cross-attention for re sampling seq. length."""

    def __init__(self, config, global_config, name="resampling_cross_attention"):
        super().__init__(name=name)
        self.config = config
        self.global_config = global_config

    def __call__(self, query, attn_mask, data, is_training=False):
        """Builds MSARowAttentionWithPairBias module.

        Arguments:
          query: [1, N_token, c_q] resampled representation.
          data: [1, N_res, c_m] full size representation.
          attn_mask: [N_head, N_token, N_res] attn masks.
          is_training: Whether the module is in training mode.

        Returns:
          Update to query, shape [1, N_token, c_q].
        """
        c = self.config

        bias = 1e9 * (attn_mask - 1.0)

        query = hk.LayerNorm(
            axis=[-1], create_scale=True, create_offset=True, name="query_norm"
        )(query)

        data = hk.LayerNorm(
            axis=[-1], create_scale=True, create_offset=True, name="data_norm"
        )(data)

        attn_mod = Attention(c, self.global_config, query.shape[-1])

        single_act = attn_mod(
            q_data=query,
            m_data=data,
            bias=bias,
        )
        return single_act


class CrossAttentionScaler(hk.Module):
    """Chain scaler using cross attention.

    Produces the single and pair representations.
    """

    def __init__(self, config, global_config, name="cross_attn_scaler"):
        super().__init__(name=name)
        self.config = config
        self.global_config = global_config

    def __call__(
        self,
        single_act: jnp.ndarray,
        attn_mask: Mask,
        output_mask: Mask,
        input_mask: Mask,
        is_training: bool,
        safe_key: Optional[RNGKey] = None,
    ):
        c = self.config
        gc = self.global_config

        if safe_key is None:
            safe_key = prng.SafeKey(hk.next_rng_key())

        #####
        #  Run Scaling CrossAttnetion iterations.
        #####
        # max_len: max_len (with down / upscaling)

        # original features
        original = single_act

        # Addind pos encoding if needed:
        if c.use_original_posenc:
            in_pos_enc = PositionalEncodingLayer(
                self.config.positional_encoding_dimension
            )
            in_sinusoidal_fn = functools.partial(
                in_pos_enc.sinusoidal_positional_encoding_features,
                indice_j=0,
                number_residues=single_act.shape[-2],
            )
            in_pos = jax.vmap(in_sinusoidal_fn)(jnp.arange(single_act.shape[-2]))
            in_pos = jnp.repeat(in_pos[None], single_act.shape[0], axis=0)
            original = jnp.concatenate([in_pos, single_act], axis=-1)

            original = hk.Linear(
                c.out_emb_size,
                name="linear_proj_original",
                w_init=build_initializer(
                    distribution=self.global_config.init.distribution,
                    scale=self.global_config.init.scale,
                    mode=self.global_config.init.mode,
                ),
            )(original)

        # resampled features
        offset = c.use_global_node if c.use_global_node else 0

        resampled_pos_enc = PositionalEncodingLayer(
            positional_encoding_dimension=c.out_emb_size
        )
        resampled_sinusoidal_fn = functools.partial(
            resampled_pos_enc.sinusoidal_positional_encoding_features,
            indice_j=0,
            number_residues=c.max_out_len + offset,
        )
        resampled = jax.vmap(resampled_sinusoidal_fn)(
            jnp.arange(c.max_out_len + offset)
        )

        resampled = jnp.repeat(resampled[None], single_act.shape[0], axis=0)

        scaler_inputs = {
            "original": original,
            "resampled": resampled,
        }
        masks = {
            "attention_mask": attn_mask,
            "output_mask": output_mask,
            "input_mask": input_mask,
        }
        # Init scaler itertion
        sc_cross_attn_iteration = CrossAttentionScalerIteration(
            c, gc, name="cross_attn_scaler_iteration"
        )

        # Main trunk of scaler.
        def scaler_fn(x):
            act, safe_key = x
            safe_key, safe_subkey = safe_key.split()
            scaler_outputs = sc_cross_attn_iteration(
                activations=act,
                masks=masks,
                is_training=is_training,
                safe_key=safe_subkey,
            )
            return (scaler_outputs, safe_key)

        if gc.use_remat:
            scaler_fn = hk.remat(scaler_fn)

        scaler_stack = layer_stack.layer_stack(c.sc_num_block)(scaler_fn)
        scaler_outputs, safe_key = scaler_stack((scaler_inputs, safe_key))

        return scaler_outputs["resampled"]


class CrossAttentionScalerIteration(hk.Module):
    """Single iteration (block) of Scaler stack."""

    def __init__(self, config, global_config, name="cross_attn_scaler_iteration"):
        super().__init__(name=name)
        self.config = config
        self.global_config = global_config

    def __call__(
        self,
        activations: jnp.ndarray,
        masks: Dict[str, Mask],
        is_training: bool = False,
        safe_key: Optional[RNGKey] = None,
    ):
        """Builds CrossAttentionScalerIteration module.

        Arguments:
          activations: Dictionary containing activations:
            * 'original': original sequence, shape [N_res, c_m].
            * 'resampled': downsampled sequence, shape [N_tokens, c_z].
          masks: Dictionary of masks:
            * 'attention_mask': single mask, shape [N_tokens, N_res].
            * 'output_mask': output mask, shape [N_tokens].
            * 'input_mask': output mask, shape [N_res].
          is_training: Whether the module is in training mode.
          safe_key: prng.SafeKey encapsulating rng key.

        Returns:
          Outputs, same type as act.
        """
        c = self.config
        gc = self.global_config

        if safe_key is None:
            safe_key = prng.SafeKey(hk.next_rng_key())

        original, resampled = (
            activations["original"],
            activations["resampled"],
        )
        # print(f"original shape: {original.shape}")
        # print(f"resampled shape: {resampled.shape}")

        (
            attention_mask,
            output_mask,
            input_mask,
        ) = (
            masks["attention_mask"],
            masks["output_mask"],
            masks["input_mask"],
        )
        # print(f"attn_mask shape: {attention_mask.shape}")
        # print(f"output_mask shape: {output_mask.shape}")
        # print(f"input_mask shape: {input_mask.shape}")

        dropout_wrapper_fn = functools.partial(
            dropout_wrapper, is_training=is_training, global_config=gc
        )

        safe_key, *sub_keys = safe_key.split(20)
        sub_keys = iter(sub_keys)

        #####
        #  resampled track
        #####
        # Origin --> resampled communication
        resampled = dropout_wrapper_fn(
            CrossAttention(c.cross_attn, gc, name="cross_attention"),
            resampled,
            mask=attention_mask,
            safe_key=next(sub_keys),
            data=original,
        )
        # resampled --> resampled transition
        resampled = dropout_wrapper_fn(
            Transition(c.resampled_transition, gc, name="resampled_transition"),
            resampled,
            output_mask,
            safe_key=next(sub_keys),
        )

        #####
        #  Original track
        #####
        # Original --> Original transition(s)
        original = dropout_wrapper_fn(
            Transition(c.original_transition, gc, name="original_transition"),
            original,
            input_mask,
            safe_key=next(sub_keys),
        )

        outputs = {
            "resampled": resampled.astype(jnp.float32),
            "original": original.astype(jnp.float32),
        }

        return outputs


class PairwiseRepresentation(hk.Module):
    def __init__(
        self,
        config: Any,  # TODO: this is not to be done for clarity of arguments
        global_config: Any,
        name: Optional[str] = "pairwise_representation",
    ):
        super(PairwiseRepresentation, self).__init__(name=name)

        self.config = config
        self.global_config = global_config
        self.w_init = build_initializer(
            distribution=self.global_config.init.distribution,
            scale=self.global_config.init.scale,
            mode=self.global_config.init.mode,
        )

    def __call__(
        self,
        x: SingleResidueRepresentation,
        mask: Optional[Mask] = None,
    ) -> PairResidueRepresentation:
        """
        The idea is to follow alg.10 of AF2. Adapted from outermeanstack.

        obtain two vectors (N_residues, input_dim) from a single vector
        (N_residues, input_dim)

        Then outer product them --> (N_residues, N_residues, input_dim)

        and scale --> (N_residues, N_residues, outdim)


        :param x:
        :return:
        """
        # reshaping mask to [Nseq, 1]

        if mask is not None:
            mask = mask[..., 0][..., None]
        else:
            mask = jnp.ones(x.shape[:-1] + [1])

        # norm input
        x = hk.LayerNorm([-1], True, True, name="layer_norm_input")(x)
        # left proj
        left_proj = mask * common_modules.Linear(  # type: ignore
            self.config.num_intermediate_factor * x.shape[-1],
            initializer="linear",
            name="left_projection",
            w_init=self.w_init,
        )(x)

        # right proj
        right_proj = mask * common_modules.Linear(  # type: ignore
            self.config.num_intermediate_factor * x.shape[-1],
            initializer="linear",
            name="right_projection",
            w_init=self.w_init,
        )(x)

        # now perform outer product
        init_pairwise_representation = jnp.einsum(
            "...nd, ...kd -> ...nkd ", left_proj, right_proj
        )

        # final proj
        pairwise_representation = hk.Sequential(
            [
                common_modules.Linear(  # type: ignore
                    self.config.num_intermediate_factor * self.config.output_dim,
                    initializer="relu",
                    name="output_projection_layer1",
                    w_init=self.w_init,
                ),
                jax.nn.relu,
                common_modules.Linear(  # type: ignore
                    self.config.output_dim,
                    name="output_projection_layer2",
                    w_init=self.w_init,
                ),  # type: ignore
            ]
        )(init_pairwise_representation)

        # right proj
        init_pairwise_representation = common_modules.Linear(  # type: ignore
            self.config.output_dim,
            initializer="linear",
            name="right_projection",
            w_init=self.w_init,
        )(init_pairwise_representation)

        if self.config.lnormalisation:
            pairwise_representation = hk.LayerNorm(
                [-1], True, True, name="layer_norm_output"
            )(pairwise_representation + init_pairwise_representation)
        else:
            pairwise_representation = (
                pairwise_representation + init_pairwise_representation
            )

        return pairwise_representation
