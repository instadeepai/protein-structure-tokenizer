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

from typing import Any, Optional

import haiku as hk
import jax
import jax.numpy as jnp

from structure_tokenizer.model import prng
from structure_tokenizer.types import QuantizerOutput


class NoQuantizer(hk.Module):
    def __init__(self, name="no_quantizer"):
        super().__init__(name)

    def __call__(
        self,
        inputs: jnp.ndarray,
        masks: jnp.ndarray,
        is_training: bool,
        safe_key=None,
    ) -> QuantizerOutput:
        output = QuantizerOutput(
            quantize=inputs,
            continuous_embedding=inputs,
            straight_through_quantized=inputs,
            soft_proba=jnp.zeros(inputs.shape[0], inputs.shape[1], 1),
            perplexity=jnp.zeros((1,)),
            tokens=inputs,
        )

        return output


def scale_and_shift(levels: jnp.ndarray, zhat_normalized: jnp.ndarray) -> jnp.ndarray:
    half_width = levels // 2
    return (zhat_normalized * half_width) + half_width


def scale_and_shift_inverse(levels: jnp.ndarray, zhat: jnp.ndarray) -> jnp.ndarray:
    half_width = levels // 2
    return (zhat - half_width) / half_width


def codes_to_indexes(
    levels: jnp.ndarray, basis: jnp.ndarray, zhat: jnp.ndarray
) -> jnp.ndarray:
    """Converts a ‘code‘ to an index in the codebook.
    index are column style

    """
    assert zhat.shape[-1] == len(levels)
    zhat = scale_and_shift(levels, zhat)
    return (zhat * basis).sum(axis=-1).astype(jnp.uint32)


def indexes_to_codes(levels, indices):
    """Inverse of ‘indexes_to_codes‘."""

    basis = jnp.concatenate((jnp.ones((1,)), jnp.cumprod(levels[:-1]))).astype(
        jnp.uint32
    )

    indices = indices[..., jnp.newaxis]
    codes_non_centered = jnp.mod(jnp.floor_divide(indices, basis), levels)
    return scale_and_shift_inverse(levels, codes_non_centered)


class FiniteScalarCodebook(hk.Module):
    def __init__(
        self,
        config: Any,
        cross_replica_axis: Optional[str] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.config = config

        # define the finite grid (aka implicit codebook)
        self.levels = config.levels
        self.levels_jnp = jnp.asarray(config.levels)
        self.basis = jnp.concatenate(
            (jnp.ones((1,)), jnp.cumprod(self.levels_jnp[:-1]))
        ).astype(jnp.uint32)

        # codebook
        self.num_codes = config.num_codes
        self.codes_dimension = self.levels_jnp.shape[0]
        self.codebook = self.indexes_to_codes(jnp.arange(self.num_codes))
        self.cross_replica_axis = cross_replica_axis

    def _scale_and_shift(self, zhat_normalized):
        half_width = self.levels_jnp // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = self.levels_jnp // 2
        return (zhat - half_width) / half_width

    def codes_to_indexes(self, zhat):
        """Converts a ‘code‘ to an index in the codebook.
        index are column style

        """
        assert zhat.shape[-1] == len(self.levels)
        zhat = self._scale_and_shift(zhat)
        return (zhat * self.basis).sum(axis=-1).astype(jnp.uint32)

    def indexes_to_codes(self, indices):
        """Inverse of ‘indexes_to_codes‘, optionally not renorming"""

        basis = jnp.concatenate(
            (jnp.ones((1,)), jnp.cumprod(self.levels_jnp[:-1]))
        ).astype(jnp.uint32)

        indices = indices[..., jnp.newaxis]
        codes_non_centered = jnp.mod(
            jnp.floor_divide(indices, basis), self.levels_jnp
        )

        scaled = self._scale_and_shift_inverse(codes_non_centered)

        if self.config.renorm:
            return scaled

        return scaled * (self.levels_jnp // 2)

    def __call__(
        self,
        inputs: jnp.ndarray,
        masks: jnp.ndarray,
        is_training: bool,
        safe_key=None,
    ) -> QuantizerOutput:
        """

        inputs:     (batch, num_tokens, D)
        masks:      (batch, num_tokens, D)

        quantize:   (batch, num_tokens, D)

        Need to add mask to remove missing residue in distance
        input: [batch, max_tokens_num, num_heads*key_size]
        mask:  [batch, max_tokens_num, num_heads*key_size]
        Compute dist[i,j,k] = sum(mask[i,j]*(codebook[k] - inputs[i,j])^2, axis=-1)
        encoding_idx = argmin(sqr_dist, axis=-1)
        """

        assert (
            inputs.shape[-1] == self.codebook.shape[-1]
        ), "wrong input embed. dimension"
        assert (
            masks.shape[-1] == self.codebook.shape[-1]
        ), "wrong input embed. dimension"

        dtype = inputs.dtype
        if safe_key is None:
            safe_key = prng.SafeKey(hk.next_rng_key())
        safe_key, safe_subkey = safe_key.split()

        # bound each each dim of the embedding between -L/2 and L/2
        def bound(z):
            """Bound ‘z‘, an array of shape (..., d)."""
            eps = 1e-3
            half_l = (self.levels_jnp - 1) * (1 - eps) / 2
            offset = jnp.where(self.levels_jnp % 2 == 0, 0.5, 0.0)
            shift = jnp.tan(offset / half_l)
            return jnp.tanh(z + shift) * half_l - offset

        bounded_inputs = bound(inputs)  # shape (B,N,D)
        bounded_inputs = masks * bounded_inputs

        # quantize: used for commitment and encoding loss
        # Here quantization is simply the rounding ops
        quantize = jnp.round(bounded_inputs)  # shape (B,N,D)

        # straight-through estimator for reconstruction loss to make gradent flow!
        straight_through_quantized = (
            bounded_inputs
            - jax.lax.stop_gradient(bounded_inputs)
            + jax.lax.stop_gradient(quantize)
        )

        # Renormalize between [-1, 1]
        if self.config.renorm:
            half_width = jnp.expand_dims(self.levels_jnp / 2, axis=[0, 1]).astype(
                quantize.dtype
            )
            quantize = quantize / half_width
            straight_through_quantized = straight_through_quantized / half_width

        # Different metrics
        # perplexity  # shape: (B,N)

        # don't forge the levels / 2 normalization !
        encoding_indices = self.codes_to_indexes(quantize / (self.levels_jnp // 2))
        # one_hots for perplexity: B, N, k, 20
        one_hot_indices = jax.nn.one_hot(encoding_indices, num_classes=self.num_codes)
        # repeat the masking to select good
        masks = jnp.expand_dims(masks[..., 0], -1)
        masks = jnp.repeat(masks, self.num_codes, axis=-1)

        # put 0 when no code from codebook is selected
        one_hot_indices = jnp.where(
            masks, one_hot_indices, jnp.zeros(one_hot_indices.shape)
        )

        one_hot_indices = one_hot_indices.reshape(-1, self.num_codes)
        avg_probs = jnp.sum(one_hot_indices, axis=0) / jnp.sum(one_hot_indices)
        avg_probs = jax.lax.pmean(avg_probs, axis_name="p")
        perplexity = jnp.exp(-jnp.sum(avg_probs * jnp.log(avg_probs + 1e-10)))

        # compute distances for soft proba
        reshaped_masks = jnp.expand_dims(masks, axis=-1)
        sqr_diff = (
            jnp.expand_dims(bounded_inputs, axis=-2)
            - jnp.expand_dims(self.codebook, axis=(0, 1))
        ) ** 2  # shape: (B, N, K, D)
        distances = jnp.sum(sqr_diff, axis=-1)  # shape: (B, N, K)

        output = QuantizerOutput(
            quantize=quantize.astype(dtype),
            straight_through_quantized=straight_through_quantized.astype(dtype),
            perplexity=perplexity,
            soft_proba=jax.nn.softmax(distances, axis=-1),  # shape: (B, N, K)
            distances=jnp.sum(reshaped_masks * sqr_diff, axis=-1),
            continuous_embedding=bounded_inputs,
            tokens=encoding_indices,
        )

        return output
