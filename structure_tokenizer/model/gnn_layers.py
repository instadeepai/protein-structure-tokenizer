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

from typing import Any, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from haiku._src.layer_norm import AxisOrAxes, to_abs_axes, to_axes_or_slice

from structure_tokenizer.model.utils import build_initializer
from structure_tokenizer.types import EdgeFeatures, NodeFeatures, ProteinGraph

ERROR_IF_PARAM_AXIS_NOT_EXPLICIT = False


class MaskedLayerNorm(hk.Module):
    """MaskedLayerNorm module."""

    def __init__(
        self,
        axis: AxisOrAxes,
        create_scale: bool,
        create_offset: bool,
        eps: float = 1e-5,
        scale_init: Optional[hk.initializers.Initializer] = None,
        offset_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
        *,
        param_axis: Optional[AxisOrAxes] = None,
    ):
        """Constructs a LayerNorm module.

        Args:
          axis: Integer, list of integers, or slice indicating which axes to
            normalize over. Note that the shape of the scale/offset parameters are
            controlled by the ``param_axis`` argument.
          create_scale: Bool, defines whether to create a trainable scale
            per channel applied after the normalization.
          create_offset: Bool, defines whether to create a trainable offset
            per channel applied after normalization and scaling.
          eps: Small epsilon to avoid division by zero variance. Defaults ``1e-5``,
            as in the paper and Sonnet.
          scale_init: Optional initializer for gain (aka scale). By default, one.
          offset_init: Optional initializer for bias (aka offset). By default, zero.
          name: The module name.
          param_axis: Axis used to determine the parameter shape of the learnable
            scale/offset. Sonnet sets this to the channel/feature axis (e.g. to
            ``-1`` for ``NHWC``). Other libraries set this to the same as the
            reduction axis (e.g. ``axis=param_axis``).
        """
        super().__init__(name=name)
        if not create_scale and scale_init is not None:
            raise ValueError("Cannot set `scale_init` if `create_scale=False`.")
        if not create_offset and offset_init is not None:
            raise ValueError("Cannot set `offset_init` if `create_offset=False`.")

        self.axis = to_axes_or_slice(axis)
        self.eps = eps
        self.create_scale = create_scale
        self.create_offset = create_offset
        self.scale_init = scale_init or jnp.ones
        self.offset_init = offset_init or jnp.zeros
        self.param_axis = (-1,) if param_axis is None else to_axes_or_slice(param_axis)
        self._param_axis_passed_explicitly = param_axis is not None

    def __call__(
        self,
        inputs: jnp.ndarray,
        mask: jnp.ndarray,
        scale: Optional[jnp.ndarray] = None,
        offset: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Connects the layer norm.

        Args:
          inputs: An array, where the data format is ``[N, ..., C]``.
          mask: An array indicating the mask (in the form [1....1,0...0])
          scale: An array up to n-D. The shape of this tensor must be broadcastable
            to the shape of ``inputs``. This is the scale applied to the normalized
            inputs. This cannot be passed in if the module was constructed with
            ``create_scale=True``.
          offset: An array up to n-D. The shape of this tensor must be broadcastable
            to the shape of ``inputs``. This is the offset applied to the normalized
            inputs. This cannot be passed in if the module was constructed with
            ``create_offset=True``.
        Returns:
          The array, normalized.
        """
        if self.create_scale and scale is not None:
            raise ValueError("Cannot pass `scale` at call time if `create_scale=True`.")
        if self.create_offset and offset is not None:
            raise ValueError(
                "Cannot pass `offset` at call time if `create_offset=True`."
            )
        axis = to_abs_axes(self.axis, inputs.ndim)
        inputs = mask * inputs
        mean = jnp.mean(
            mask * inputs,
            axis=axis,
            keepdims=True,
        )

        variance = jnp.mean(
            mask * ((inputs - mean) * (inputs - mean)),
            axis=axis,
            keepdims=True,
        )

        if (
            self.create_scale or self.create_offset
        ) and not self._param_axis_passed_explicitly:
            if ERROR_IF_PARAM_AXIS_NOT_EXPLICIT and axis != (inputs.ndim - 1,):
                raise ValueError(
                    "When axis is not the final dimension we require "
                    "you to also pass `param_axis` in the ctor."
                    f" axis={axis} ndim={inputs.ndim}"
                )

        # Shape for the learnable scale and offset is the number of channels.
        # See: https://arxiv.org/pdf/1803.08494.pdf around equation 6.
        param_axis = to_abs_axes(self.param_axis, inputs.ndim)
        if param_axis == (inputs.ndim - 1,):
            # For param_axis=-1 we store non-broadcast param shape for compatibility
            # with older checkpoints.
            param_shape: Tuple[int, ...] = (inputs.shape[-1],)
        else:
            param_shape = tuple(
                (inputs.shape[i] if i in param_axis else 1) for i in range(inputs.ndim)
            )

        if self.create_scale:
            scale = hk.get_parameter(
                "scale", param_shape, inputs.dtype, init=self.scale_init
            )
        elif scale is None:
            scale = np.array(1.0, dtype=inputs.dtype)

        if self.create_offset:
            offset = hk.get_parameter(
                "offset", param_shape, inputs.dtype, init=self.offset_init
            )
        elif offset is None:
            offset = np.array(0.0, dtype=inputs.dtype)

        scale = jnp.broadcast_to(scale, inputs.shape)
        offset = jnp.broadcast_to(offset, inputs.shape)
        mean = jnp.broadcast_to(mean, inputs.shape)

        eps = jax.lax.convert_element_type(self.eps, variance.dtype)
        inv = scale * jax.lax.rsqrt(variance + eps)
        return inv * (inputs - mean) + offset


class GNNLayer(hk.Module):
    """Baseline GNN layer"""

    def __init__(
        self,
        config: Any,
        global_config: Any,
        graph_max_neighbor: int,
    ):
        """
        Constructs a baseline GNN Layer

        Args:
            config: config of the layer
            graph_max_neighbor: graph maximum neighbor
        """
        super(GNNLayer, self).__init__()
        self.config = config
        self.global_config = global_config
        self.graph_max_neighbor = graph_max_neighbor

    def __call__(
        self, graph_protein: ProteinGraph
    ) -> Tuple[EdgeFeatures, NodeFeatures]:
        """Computes node/edge embeddings of a protein graph using message passing

        Args:
            graph_protein (ProteinGraph): graph of the protein of interest

        Returns:
            node_features (NodeFeatures): embeddings of nodes
            edge_features (EdgeFeatures): embeddings of edges
        """
        dim = self.config.hidden_dimension

        node_inp = graph_protein.node_features
        node_mask = graph_protein.nodes_mask
        num_res = graph_protein.node_features.shape[0]

        # M_ij
        msg_input = jnp.concatenate(
            [
                node_inp[graph_protein.senders],
                node_inp[graph_protein.receivers],
                graph_protein.edge_features,
            ],
            axis=-1,
        )
        node_messages = hk.nets.MLP(  # BG: check masking
            [2 * dim, dim],
            activation=jax.nn.swish,
            name="node_mlp_0",
            w_init=build_initializer(
                distribution=self.global_config.init.distribution,
                scale=self.global_config.init.scale,
                mode=self.global_config.init.mode,
            ),
        )(msg_input)

        # dV_i_0
        agg_messages = jax.ops.segment_sum(
            node_messages,
            graph_protein.receivers,
            num_segments=len(node_inp),
            indices_are_sorted=True,
            unique_indices=False,
            bucket_size=None,
            mode=None,
        )
        # normalizing
        if self.graph_max_neighbor > 0:
            agg_messages /= self.graph_max_neighbor
        else:
            agg_messages /= jnp.sum(node_mask)

        # V_i_0
        node_features = MaskedLayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="norm_msg"
        )(node_inp + agg_messages, node_mask)

        # dV_i_1
        feedforward_upd = hk.nets.MLP(
            [dim],
            activation=jax.nn.swish,
            name="node_mlp_1",
            w_init=build_initializer(
                distribution=self.global_config.init.distribution,
                scale=self.global_config.init.scale,
                mode=self.global_config.init.mode,
            ),
        )(node_features)

        # V_i_1
        node_features = MaskedLayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="norm_msg"
        )(node_features + feedforward_upd, node_mask)

        # dE_ij
        msg_input = jnp.concatenate(
            [
                node_features[graph_protein.senders],
                node_features[graph_protein.receivers],
                graph_protein.edge_features,
            ],
            axis=-1,
        )
        edge_messages = hk.nets.MLP(  # BG: check masking
            [2 * dim, dim],
            activation=jax.nn.swish,
            name="edge_mlp",
            w_init=build_initializer(
                distribution=self.global_config.init.distribution,
                scale=self.global_config.init.scale,
                mode=self.global_config.init.mode,
            ),
        )(msg_input)

        # E_ij
        if self.graph_max_neighbor > 0:
            edge_messages = (graph_protein.edge_features + edge_messages).reshape(
                -1, self.graph_max_neighbor, dim
            )
        else:
            edge_messages = (graph_protein.edge_features + edge_messages).reshape(
                num_res, num_res, dim
            )

        edge_features = MaskedLayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="norm_msg"
        )(
            edge_messages, jnp.expand_dims(node_mask, axis=-1)
        )  # BG: to check masking and reshaping
        edge_features = edge_features.reshape(-1, dim)

        return edge_features, node_features


class MPNNLayer(hk.Module):
    """GNN Layer of ProteinMPNN"""

    def __init__(
        self,
        config: Any,
        global_config: Any,
        graph_max_neighbor: int,
    ):
        """
        Constructs a GNN Layer from ProteinMPNN

        Args:
            config: config of the layer
            graph_max_neighbor: graph maximum neighbor
        """
        super(MPNNLayer, self).__init__()
        self.config = config
        self.global_config = global_config
        self.graph_max_neighbor = graph_max_neighbor

    def __call__(
        self, graph_protein: ProteinGraph
    ) -> Tuple[EdgeFeatures, NodeFeatures]:
        """Computes node/edge embeddings of a protein graph using message passing

        Args:
            graph_protein (ProteinGraph): graph of the protein of interest

        Returns:
            node_features (NodeFeatures): embeddings of nodes
            edge_features (EdgeFeatures): embeddings of edges
        """
        dim = self.config.hidden_dimension

        node_inp = graph_protein.node_features
        node_mask = graph_protein.nodes_mask
        num_res = graph_protein.node_features.shape[0]

        # M_ij
        msg_input = jnp.concatenate(
            [
                node_inp[graph_protein.senders],
                node_inp[graph_protein.receivers],
                graph_protein.edge_features,
            ],
            axis=-1,
        )
        node_messages = hk.nets.MLP(  # BG: check masking
            [dim, dim, dim],
            activation=jax.nn.gelu,
            name="node_mlp_0",
            w_init=build_initializer(
                distribution=self.global_config.init.distribution,
                scale=self.global_config.init.scale,
                mode=self.global_config.init.mode,
            ),
        )(msg_input)

        # dV_i_0
        agg_messages = jax.ops.segment_sum(
            node_messages,
            graph_protein.receivers,
            num_segments=len(node_inp),
            indices_are_sorted=True,
            unique_indices=False,
            bucket_size=None,
            mode=None,
        )
        # normalizing
        if self.graph_max_neighbor > 0:
            agg_messages /= self.graph_max_neighbor
        else:
            agg_messages /= jnp.sum(node_mask)

        # V_i_0
        node_features = MaskedLayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="norm_msg"
        )(node_inp + agg_messages, node_mask)

        # dV_i_1
        feedforward_upd = hk.nets.MLP(
            [4 * dim, dim],
            activation=jax.nn.gelu,
            name="node_mlp_1",
            w_init=build_initializer(
                distribution=self.global_config.init.distribution,
                scale=self.global_config.init.scale,
                mode=self.global_config.init.mode,
            ),
        )(node_features)

        # V_i_1
        node_features = MaskedLayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="norm_msg"
        )(node_features + feedforward_upd, node_mask)

        # dE_ij
        msg_input = jnp.concatenate(
            [
                node_features[graph_protein.senders],
                node_features[graph_protein.receivers],
                graph_protein.edge_features,
            ],
            axis=-1,
        )
        edge_messages = hk.nets.MLP(  # BG: check masking
            [dim, dim, dim],
            activation=jax.nn.gelu,
            name="edge_mlp",
            w_init=build_initializer(
                distribution=self.global_config.init.distribution,
                scale=self.global_config.init.scale,
                mode=self.global_config.init.mode,
            ),
        )(msg_input)

        # E_ij
        if self.graph_max_neighbor > 0:
            edge_messages = (graph_protein.edge_features + edge_messages).reshape(
                -1, self.graph_max_neighbor, dim
            )
        else:
            edge_messages = (graph_protein.edge_features + edge_messages).reshape(
                num_res, num_res, dim
            )

        edge_features = MaskedLayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="norm_msg"
        )(
            edge_messages, jnp.expand_dims(node_mask, axis=-1)
        )  # BG: to check masking and reshaping
        edge_features = edge_features.reshape(-1, dim)

        return edge_features, node_features
