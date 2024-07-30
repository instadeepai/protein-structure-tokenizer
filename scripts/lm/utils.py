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

import logging
from typing import Callable

import jax
import jax.numpy as jnp

from scripts.lm.lm_types import AttentionMask

SUPPORTED_FFN_ACTIVATIONS = ["gelu", "gelu-no-approx", "relu", "swish", "silu", "sin"]


def get_activation_fn(activation_name: str) -> Callable:
    """
    Return activation fn given its name.
    Args:
        activation_name: Activation name.

    Returns:
        activation function.
    """
    if activation_name not in SUPPORTED_FFN_ACTIVATIONS:
        raise NotImplementedError(
            f"Activation {activation_name} not supported yet. "
            f"Supported activations for feed forward "
            f"block are {SUPPORTED_FFN_ACTIVATIONS}"
        )
    if activation_name == "gelu-no-approx":
        activation_fn = lambda x: jax.nn.gelu(x, approximate=False)  # noqa: E731
    elif activation_name == "sin":
        activation_fn = lambda x: jnp.sin(x)  # noqa: E731
    else:
        activation_fn = getattr(jax.nn, activation_name)
    return activation_fn


def debug_log_tensor(
    tensor_name: str, tensor: jnp.ndarray, logger: logging.Logger
) -> None:
    """
    Logging with debugging level a tensor's name, shape and dtype.

    Args:
        tensor_name: tensor's name.
        tensor: jnp tensor.
        logger: logger to be used.
    """
    shape = tensor.shape
    dtype = tensor.dtype
    debug_msg = f"Tensor, Name = {tensor_name}, Shape = {shape}, Dtype = {dtype}"
    logger.debug(debug_msg, stacklevel=2)


def build_causal_attention_mask(batch_size: int, seq_len: int) -> AttentionMask:
    """
    Builds a batch of causal masks of shape (batch_size, 1, seq_len, seq_len) to feed
    to an attention layer.

    Args:
        batch_size: Batch size.
        seq_len: Length of the sequences.

    Returns:
        Batch of causal masks.
    """
    mask = jnp.ones((batch_size, 1, seq_len, seq_len))
    causal_mask = jnp.tril(mask)
    return causal_mask
