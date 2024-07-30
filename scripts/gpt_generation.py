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

import argparse
import functools
import os
import time
from typing import Callable, List, Optional

import haiku as hk
import jax
import jax.numpy as jnp
import joblib
import numpy as np
from tqdm import tqdm

from scripts.inference_runner import InferenceRunner
from scripts.lm.generation_utils import (
    update_tokens_ids_temperature_sampling,
    update_top_k_sampling,
    update_top_p_sampling,
)
from scripts.lm.gpt_model import GptConfig, build_gpt_fn
from scripts.lm.lm_types import Tokens
from structure_tokenizer.types import RNGKey
from structure_tokenizer.utils.log import get_logger
from structure_tokenizer.utils.utils import load_config

logger = get_logger(__name__)


def load_params(filename: str) -> hk.Params:
    """
    Load params from a joblib archive.

    Args:
        filename: File name where it is to be saved

    Returns:
        Parameters pytree.
    """
    return joblib.load(filename)


def generate_sequence_batch(
    update_tokens_fn_sampling: Callable,
    model_params: hk.Params,
    random_key: RNGKey,
    bos_token_id: int,
    batch_size_per_device: int,
    num_tokens_to_decode: int,
    devices: List[jax.Device],
) -> Tokens:

    prompt_length = 1
    sentences = jnp.full(
        shape=(batch_size_per_device, num_tokens_to_decode),
        fill_value=bos_token_id,
    )
    tokens_ids = jax.device_put_replicated(sentences, devices=devices)

    for i in range(num_tokens_to_decode):
        time_step = i + prompt_length - 1
        time_step = jax.device_put_replicated(jnp.asarray(time_step), devices=devices)
        tokens_ids, random_key = update_tokens_fn_sampling(
            tokens_ids=tokens_ids,
            random_key=random_key,
            params=model_params,
            time_step=time_step,
        )

    # all gather ?
    sequences = tokens_ids.reshape(-1, num_tokens_to_decode)
    sequences = jax.device_put(sequences, jax.devices("cpu")[0])

    return sequences


def main(
    sequence_save_path: str,
    gpt_params_path: str,
    batch_size_per_device: int,
    backend: str,
    config_overrides: Optional[List[str]] = None,
):

    config = load_config(
        name="generation",
        job_name="generation",
        overrides=config_overrides,
        config_path="../../config/lm/",
    )

    logger.info("Starting sequence generation...")
    start_time = time.time()

    # Sampling config
    sampling_args = config.sampling

    # Seqs dir
    seq_dir = os.path.join(sequence_save_path, "seqs")
    os.makedirs(seq_dir, exist_ok=True)

    # setting random seed
    seed = sampling_args.seed
    key = jax.random.PRNGKey(seed)
    random_key, _ = jax.random.split(key)

    devices, num_devices = InferenceRunner.prepare_devices(backend)

    # loading params from checkpoint
    params = load_params(filename=gpt_params_path)
    model_params = jax.device_put_replicated(params, devices=devices)
    logger.info("params loaded successfully")

    # Model config
    model_args = config.model.gpt_model
    bos_token_id = model_args.bos_token_id
    eos_token_id = model_args.eos_token_id
    block_size = model_args.block_size

    # build model config
    model_config = GptConfig(
        vocab_size=model_args.vocab_size,
        eos_token_id=model_args.eos_token_id,
        embed_dim=model_args.embed_dim,
        ffn_embed_dim=model_args.ffn_embed_dim,
        num_heads=model_args.num_heads,
        num_layers=model_args.num_layers,
        rope_dimensions=model_args.rope_dimensions,
        max_position_embeddings=block_size,
        add_bias_ffn=model_args.add_bias_ffn,
        ffn_activation_name=model_args.ffn_activation_name,
        use_glu_in_ffn=model_args.use_glu_in_ffn,
        add_bias_lm_head=model_args.add_bias_lm_head,
        use_gradient_checkpointing=False,
        norm_type="layer_norm",
        parallel_attention_ff=False,
        dropout_rate=0.0,
    )

    # get haiku functions from model config
    gptj = build_gpt_fn(model_config)
    gptj_fn = hk.transform(gptj)

    # define sampling fn
    if sampling_args.sampling_method == "top_k":
        # We resort to temperature sampling
        sampling_temp = sampling_args.sampling_temp
        top_k = sampling_args.top_k
        update_tokens_fn_sampling = functools.partial(
            update_top_k_sampling,
            apply_fn=gptj_fn.apply,
            k=top_k,
            temperature=sampling_temp,
        )

    elif sampling_args.sampling_method == "top_p":
        sampling_temp = sampling_args.sampling_temp
        top_p = sampling_args.top_p
        update_tokens_fn_sampling = functools.partial(
            update_top_p_sampling,
            apply_fn=gptj_fn.apply,
            top_p=top_p,
            temperature=sampling_temp,
        )

    elif sampling_args.sampling_method == "temperature":
        sampling_temp = sampling_args.sampling_temp
        update_tokens_fn_sampling = functools.partial(
            update_tokens_ids_temperature_sampling,
            apply_fn=gptj_fn.apply,
            temperature=sampling_temp,
        )

    else:
        raise ValueError(f"Sampling method not defined: {sampling_args.sampling_method}")

    # generation config
    num_sequences = sampling_args.num_sequences
    batch_size = batch_size_per_device * num_devices
    num_iterations = int(num_sequences // batch_size)

    # Pmap
    update_tokens_fn_sampling = jax.pmap(
        update_tokens_fn_sampling, axis_name="p", devices=devices
    )

    logger.info("Fwd sampling function build")

    # setting random seed
    sampling_keys = jax.random.split(random_key, num=num_iterations)

    # Generating
    generated_sequences = []
    logger.info("Generating sequences !")

    for j in tqdm(range(num_iterations), total=num_iterations):

        # set a different random key per device to make batch random generation !
        random_key = jax.random.split(sampling_keys[j], num=num_devices)

        sequences = generate_sequence_batch(
            random_key=random_key,
            update_tokens_fn_sampling=update_tokens_fn_sampling,
            model_params=model_params,
            bos_token_id=bos_token_id,
            batch_size_per_device=batch_size_per_device,
            num_tokens_to_decode=block_size,
            devices=devices,
        )
        sequences = sequences.reshape(-1, block_size)
        sequences = jax.device_put(sequences, jax.devices("cpu")[0])

        generated_sequences.append(sequences)

    generated_sequences_arr = jnp.concatenate(generated_sequences, axis=0)

    # Save generated sequences to local
    for seq_id in range(generated_sequences_arr.shape[0]):
        # remove bos, eos and pad tokens
        padded_seq = generated_sequences_arr[seq_id]
        # Unpadding
        is_eos_token = padded_seq == eos_token_id
        is_before_first_eos_token = jnp.where(
            jnp.cumsum(is_eos_token, axis=-1) == 0, True, False
        )
        unpadded_seq = padded_seq[is_before_first_eos_token]
        # remove bos / eos tokens
        # FIXME: check if it is not "ceinture bretelles"
        seq = unpadded_seq[
            ~((unpadded_seq == bos_token_id) | (unpadded_seq == eos_token_id))
        ]
        # create dir for generated sequence and save tokens
        filename = f"tokens_{seq_id}.npy"
        seq_filepath = os.path.join(seq_dir, filename)
        np.save(seq_filepath, seq)

    logger.info(
        f"Sequence generation done."
        f" Took {time.time() - start_time}s to generate "
        f"{generated_sequences_arr.shape[0]}  sequences"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate sequences !")
    parser.add_argument("--sequence_save_path", type=str, required=True)

    parser.add_argument("--model_downsampling", type=int, choices=[1, 2], default=1)
    parser.add_argument("--codebook_size", type=int, default=4096, choices=[4096, 64000])
    parser.add_argument("--batch_size_per_device", type=int, default=8)
    parser.add_argument(
        "--params_path", type=str, default="weights/gpt_4k_df_1/params.joblib"
    )
    parser.add_argument(
        "--backend", type=str, default="gpu", choices=["gpu", "tpu", "cpu"]
    )

    args = parser.parse_args()

    codebook_size = str(args.codebook_size // 1000) + "k"
    df = args.model_downsampling

    model_config_name = f"gpt_{codebook_size}_df_{df}.yaml"
    data_config = f"ablation_df_{df}.yaml"

    # as of now sampling is fixed
    overrides: List[str] = [f"model={model_config_name}", "sampling=sampling"]

    logger.info(f"Using {model_config_name}")

    main(
        sequence_save_path=args.sequence_save_path,
        gpt_params_path=args.params_path,
        batch_size_per_device=args.batch_size_per_device,
        config_overrides=overrides,
        backend=args.backend,
    )
