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
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp

from scripts.lm.lm_types import Tokens
from structure_tokenizer.types import RNGKey


def update_tokens_ids_temperature_sampling(
    tokens_ids: Tokens,
    time_step: jnp.ndarray,
    random_key: RNGKey,
    params: hk.Params,
    apply_fn: hk.Transformed.apply,
    temperature: float = 1.0,
) -> Tuple[Tokens, RNGKey]:
    """
    Update the input sequence of tokens using temperature sampling decoding
    with a decoder model. Typical inputs could be a prompt with end of tokens appended.
    This function can then be called recursively after the prompt to generate the
    rest of the sentence.

    Args:
        tokens_ids: Input tokens ids, shape = (batch_size, sequence_length).
        time_step: Time step at which to decode, shape = (,).
        random_key: Random key.
        params: Decoder parameters.
        apply_fn: Decoder apply fn.
        temperature: temperature coefficient for sampling.

    Returns:
        Tokens ids with decoded token at position time_step + 1 and updated random key.
    """
    logits = apply_fn(params, random_key, tokens_ids)["logits"]
    logits = logits[:, time_step, :]
    rescaled_logits = logits / temperature
    random_key, sub_key = jax.random.split(random_key)
    new_token_id = jax.random.categorical(sub_key, rescaled_logits, axis=-1)
    tokens_ids = tokens_ids.at[:, time_step + 1].set(new_token_id)

    return tokens_ids, random_key


def update_top_k_sampling(
    tokens_ids: Tokens,
    time_step: jnp.ndarray,
    random_key: RNGKey,
    params: hk.Params,
    apply_fn: hk.Transformed.apply,
    k: int = 5,
    temperature: float = 1.0,
) -> Tuple[Tokens, RNGKey]:
    """
        This function implements top_k sampling:
        1. it first select the k highest logits values and indices
        2. it then samples randomly the selected indices using the previously
            estimated logits (optionnaly rescaled by temperature)

    :param tokens_ids:
    :param time_step:
    :param random_key:
    :param params:
    :param apply_fn:
    :param k:
    :param temperature:
    :return:
    """

    top_k = jax.vmap(functools.partial(jax.lax.top_k, k=k))

    # B, T, # voc
    logits = apply_fn(params, random_key, tokens_ids)["logits"]

    # B, # voc
    logits = logits[:, time_step, :]

    # (B, k) (B, k)
    selected_logits, selected_indices = top_k(logits)

    random_key, sub_key = jax.random.split(random_key)

    # (B, ) < k
    selected_sub_index = jax.random.categorical(
        sub_key, selected_logits / temperature, axis=-1
    )

    selected_sub_index = selected_sub_index[..., None]

    new_token_id = jnp.take_along_axis(
        selected_indices, selected_sub_index, axis=-1
    ).squeeze()
    # .squeeze() to remove the trailing dimension that was necessary
    # for the take_along_axis

    tokens_ids = tokens_ids.at[:, time_step + 1].set(new_token_id)

    return tokens_ids, random_key


def update_top_p_sampling(
    tokens_ids: Tokens,
    time_step: jnp.ndarray,
    random_key: RNGKey,
    params: hk.Params,
    apply_fn: hk.Transformed.apply,
    top_p: float = 0.2,
    min_tokens_to_keep: int = 2,
    temperature: float = 1.0,
) -> Tuple[Tokens, RNGKey]:
    """
        This function implements top_p sampling: sampling only the highest
        probability tokens that sums to 'p'.

    :param tokens_ids:
    :param time_step:
    :param random_key:
    :param params:
    :param apply_fn:
    :param top_p:
    :param min_tokens_to_keep:
    :param temperature:
    :return:
    """

    # B, T, # voc
    logits = apply_fn(params, random_key, tokens_ids)["logits"]

    batch_size, t, size_voc = logits.shape
    top_k = jax.vmap(functools.partial(jax.lax.top_k, k=size_voc))

    # B, # voc
    logits = logits[:, time_step, :]

    # sort the logits and the index
    sorted_logits, sorted_indices = top_k(logits)

    # compute the cumulated probability distribution (high value first)
    cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)

    # set token logit to -inf where top_p is exceeded
    new_logits = jnp.where(cumulative_probs <= top_p, sorted_logits, -jnp.inf)

    # Keep at least min_tokens_to_keep
    min_tokens_to_keep_logits = jnp.where(
        jnp.repeat(jnp.arange(size_voc)[None], batch_size, axis=0) < min_tokens_to_keep,
        sorted_logits,
        -jnp.inf,
    )

    # if we don't have at least min_tokens put them !
    logits_to_sample_from = jnp.where(
        (new_logits > -jnp.inf) & (min_tokens_to_keep_logits > -jnp.inf),
        min_tokens_to_keep_logits,
        new_logits,
    )

    random_key, sub_key = jax.random.split(random_key)
    # (B, ) < k
    selected_sub_index = jax.random.categorical(
        sub_key, logits_to_sample_from / temperature, axis=-1
    )
    selected_sub_index = selected_sub_index[..., None]

    # Take the index where it is in the original sorted_index array
    new_token_id = jnp.take_along_axis(
        sorted_indices, selected_sub_index, axis=-1
    ).squeeze()
    # .squeeze() to remove the trailing dimension that was necessary
    # for the take_along_axis

    tokens_ids = tokens_ids.at[:, time_step + 1].set(new_token_id)

    return tokens_ids, random_key


def random_sampling(
    random_key: RNGKey,
    max_len: int,
    num_sequences: int,
    average_num_struct_tokens: int,
    codebook_size: int,
    eos_token_id: int,
    bos_token_id: int,
) -> Tuple[jnp.asarray, RNGKey]:
    """
        Method to randomly sample integers in the codebook

    :param random_key:
    :param max_len:
    :param num_sequences:
    :param average_num_struct_tokens:
    :param codebook_size:
    :param eos_token_id:
    :param bos_token_id:
    :return:
    """

    # (num_sequences, )
    sequence_length = jnp.clip(
        jax.random.poisson(
            key=random_key,
            lam=average_num_struct_tokens,
            shape=(num_sequences,),
        ),
        a_max=max_len,
    )

    # (num_sequences, max_len)
    sequence_length = sequence_length[..., None].repeat(max_len, axis=-1)

    # (num_sequences, max_len): num_sequences times: [0, 1,...., max_len]
    array_index = jnp.arange(max_len)[None].repeat(num_sequences, axis=0)

    # Sample (N, max_len) integers in the range of the codebook size
    random_key, subkey = jax.random.split(random_key)

    generated_sequences = jax.random.randint(
        key=random_key,
        shape=(num_sequences, max_len),
        minval=0,
        maxval=codebook_size,  # excluded
    )

    # if the index < the sequence length: random token otherwise eos_token
    generated_sequences = jnp.where(
        array_index < sequence_length,
        generated_sequences,
        jnp.ones(generated_sequences.shape) * eos_token_id,
    )

    # add bos and eos_token
    generated_sequences = jnp.pad(
        generated_sequences, ((0, 0), (1, 0)), constant_values=bos_token_id
    )

    generated_sequences = jnp.pad(
        generated_sequences, ((0, 0), (0, 1)), constant_values=eos_token_id
    )

    return generated_sequences, subkey
