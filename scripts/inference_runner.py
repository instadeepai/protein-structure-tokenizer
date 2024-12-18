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
import os
import time
from itertools import cycle, islice
from typing import Any, Callable, List, Optional, Union

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from cloudpathlib import AnyPath
from ml_collections import ConfigDict
from tqdm import tqdm

from structure_tokenizer.data.preprocessing import preprocess_sample
from structure_tokenizer.data.protein import Protein, to_pdb
from structure_tokenizer.data.protein_structure_sample import (
    protein_structure_from_pdb_string,
)
from structure_tokenizer.model.model import Vq3D
from structure_tokenizer.model.quantize import indexes_to_codes
from structure_tokenizer.types import ProteinGraph, QuantizerOutput, RNGKey, Structure


def make_graph_from_pdb(
    pdb_file_path: str,
    num_neighbor: int,
    downsampling_ratio: int,
    residue_loc_is_alphac: bool,
    padding_num_residue: int,
) -> ProteinGraph:
    with open(pdb_file_path, "r") as file:
        pdb_content = file.read()

    protein_structure_sample = protein_structure_from_pdb_string(pdb_content)

    if protein_structure_sample.nb_residues > 512:
        raise NotImplementedError(
            "We currently don't support protein with more than 512 residues"
            f"given: {protein_structure_sample.nb_residues}"
        )

    if protein_structure_sample.nb_residues < num_neighbor:
        raise NotImplementedError(
            f"We currently don't support protein with less than {num_neighbor} residues"
            f"given: {protein_structure_sample.nb_residues}"
        )

    protein_graph = preprocess_sample(
        sample=protein_structure_sample,
        num_neighbor=num_neighbor,
        downsampling_ratio=downsampling_ratio,
        residue_loc_is_alphac=residue_loc_is_alphac,
        padding_num_residue=padding_num_residue,
        crop_index=padding_num_residue,
        noise_level=0.0,
    ).graph

    return protein_graph


def batch_collate(
    batch_dims: List[int], batch_of_samples: List[ProteinGraph]
) -> ProteinGraph:
    return jax.tree_map(
        lambda *x: np.stack(x).reshape((*batch_dims, *x[0].shape)),
        *batch_of_samples,
    )


def build_tokens_mask_from_sequence(
    tokens_ids: jnp.ndarray,
    pad_token_id: int,
) -> jnp.ndarray:
    assert len(tokens_ids.shape) >= 2

    is_eos_token = tokens_ids == pad_token_id
    is_before_first_eos_token = jnp.where(jnp.cumsum(is_eos_token, axis=-1) == 0, 1, 0)

    return is_before_first_eos_token


def build_nodes_mask_from_tokens_mask(
    tokens_mask: jnp.ndarray,
    downsampling_ratio: int,
) -> jnp.ndarray:
    batch, seq_len = tokens_mask.shape

    actual_nb_tokens = tokens_mask.sum(axis=-1)

    # B,1
    last_true_node = (downsampling_ratio * actual_nb_tokens).reshape(batch, 1)
    index = jnp.repeat(jnp.arange(downsampling_ratio * seq_len)[None], batch, axis=0)
    nodes_mask = jnp.where(index < last_true_node, 1, 0)

    return nodes_mask


def load_and_build_batch(files_paths, max_seq_len, pad_token_id):
    def pad(seq, max_seq_len):
        return np.pad(
            seq,
            ((0, 0), (0, max_seq_len - seq.shape[-1])),
            mode="constant",
            constant_values=pad_token_id,
        )

    return np.concatenate(
        [
            pad(
                # load up to max len
                np.load(file_path).astype(np.int32).reshape(1, -1)[:, :max_seq_len],
                # load(file_path).reshape(1, -1),
                max_seq_len,
            )
            for file_path in files_paths
        ]
    )


def load_params(filename: str, tree_def: jax.tree_util.PyTreeDef) -> hk.Params:
    """Load params from a .npz file.

    Args:
        filename (str): File name where it is to be saved
        tree_def: Parameters pytree definition.

    Returns:
        hk.Params: Parameters pytree.
    """
    with open(filename, "rb") as f:
        uploaded = jnp.load(f)
        arrays = [jnp.asarray(uploaded[file]) for file in uploaded.files]
        reconstructed_params = jax.tree_util.tree_unflatten(tree_def, arrays)
        return reconstructed_params


def params_keys_conversion(dict_params: hk.Params, key_name: str = "forward_vq3_d/"):
    """
    utils to remove the "first node to use direct .decode_and_qunatize method"
    :param dict_params:
    :return:
    """
    list_key = list(dict_params.keys())
    for key in list_key:
        local_params = dict_params[key]
        new_key = key.split(key_name)[1]
        dict_params[new_key] = local_params
        del dict_params[key]
    return dict_params


class InferenceRunner:
    @staticmethod
    def prepare_devices(backend: str):
        # devices
        num_local_devices = jax.local_device_count(backend=backend)
        local_devices = jax.local_devices(backend=backend)

        if jax.process_index() == 0:
            print("---Devices---\n" + f"\tlocal device count: {num_local_devices}")
        return local_devices, num_local_devices

    @staticmethod
    def prepare_tokenize_fn(
        cfg: Any,
        devices: jax.Device,
    ) -> Callable:
        def fn(graph: ProteinGraph, safe_key=None) -> QuantizerOutput:
            return Vq3D(
                config=cfg.model,
                global_config=cfg.data,
            ).encode_and_quantize(graph, is_training=False, safe_key=safe_key)

        tokenize = hk.transform(fn)
        return jax.pmap(tokenize.apply, devices=devices, axis_name="p")

    @staticmethod
    def prepare_decode_fn(
        cfg: Any,
        devices: jax.Device,
    ) -> Callable:
        def fn(code, nodes_mask, tokens_mask) -> Structure:
            return Vq3D(
                config=cfg.model,
                global_config=cfg.data,
            ).decode_and_make_structure(
                code, nodes_mask, tokens_mask, is_training=False, safe_key=None
            )

        decode = hk.transform(fn)
        return jax.pmap(decode.apply, devices=devices, axis_name="p")

    @staticmethod
    def prepare_ae_fn(
        cfg: Any,
        devices: List[jax.Device],
    ) -> Callable:
        def fn(batch) -> Structure:
            return Vq3D(
                config=cfg.model,
                global_config=cfg.data,
            )(
                batch, is_training=False, safe_key=None
            )
        encode_decode = hk.transform(fn)
        return jax.pmap(encode_decode.apply, devices=devices, axis_name="p")

    @staticmethod
    def prepare_token_to_code_fn(cfg: Any, devices: List[jax.Device]) -> Callable:
        def fn(tokens: jnp.ndarray, safe_key=None) -> QuantizerOutput:
            return Vq3D(
                config=cfg.model,
                global_config=cfg.data,
            ).indexes_to_codes(tokens)

        tokens_to_code = hk.transform(fn)
        return jax.pmap(tokens_to_code.apply, devices=devices, axis_name="p")


    @staticmethod
    def load_params(model_dir: str, local_devices: jax.Device) -> hk.Params:
        information = np.load(
            os.path.join(model_dir, "state_variables.npy"), allow_pickle=True
        ).item()
        params = load_params(
            filename=os.path.join(model_dir, "params.npz"),
            tree_def=information["params_treedef"],
        )
        params = params_keys_conversion(params, key_name="forward_vq3_d/")
        params = jax.device_put_replicated(params, local_devices)

        return params

    @staticmethod
    def tokenize(
        random_key: RNGKey,
        quantize: Callable[[hk.Params, RNGKey, ProteinGraph], QuantizerOutput],
        model_params: hk.Params,
        pdbs: List[str],
        token_save_path: Union[str, AnyPath],
        num_device: int,
        data_config: ConfigDict,
        batch_size_per_device: int = 8,
        logger: Optional[logging.Logger] = None,
    ):
        if logger is not None:
            logger.info(f"Starting tokenization of {pdbs}")

        os.makedirs(token_save_path, exist_ok=False)

        effective_batch_size = batch_size_per_device * num_device
        num_iteration = len(pdbs) // effective_batch_size + int(
            (len(pdbs) % effective_batch_size) > 0
        )

        total = num_iteration * effective_batch_size

        # repeat the list to reach a length of "total"
        pdbs = list(islice(cycle(pdbs), total))

        for it in tqdm(range(num_iteration), total=num_iteration):
            start_index = it * effective_batch_size

            pdb_to_tokenize = [
                pdbs[i] for i in range(start_index, start_index + effective_batch_size)
            ]

            pdb_file_names = [os.path.basename(pdb_file) for pdb_file in pdb_to_tokenize]

            start_time = time.perf_counter()

            graphs = [
                make_graph_from_pdb(
                    pdb_file_path=pdb_file,
                    num_neighbor=data_config.graph_max_neighbor,
                    downsampling_ratio=data_config.downsampling_ratio,
                    residue_loc_is_alphac=data_config.graph_residue_loc_is_alphac,
                    padding_num_residue=data_config.seq_max_size,
                )
                for pdb_file in pdb_to_tokenize
            ]

            batched_graph = batch_collate(
                batch_dims=[num_device, batch_size_per_device], batch_of_samples=graphs
            )

            decoded_embedding = quantize(model_params, random_key, batched_graph)

            tokens = jax.block_until_ready(decoded_embedding["tokens"])
            tokens = jax.device_put(tokens, jax.devices("cpu")[0])
            batch_dim = np.prod(tokens.shape[:2])
            tokens = tokens.reshape(batch_dim, -1)

            # we just extract and save the true structure tokens
            number_structural_tokens = batched_graph.tokens_mask.reshape(
                batch_dim, -1
            ).sum(axis=1)

            for seq_id in range(tokens.shape[0]):
                token_array = tokens[seq_id].reshape(1, -1)
                # subselect only structural tokens
                token_array = token_array[:, : number_structural_tokens[seq_id]]
                filename = pdb_file_names[seq_id].split(".pdb")[0]
                path = os.path.join(token_save_path, filename + "_tokens")
                np.save(path, token_array)

            if logger is not None:
                logger.info(f"Took {time.perf_counter() - start_time}s to tokenize")

    @staticmethod
    def decode_and_save_pdbs(
        random_key: RNGKey,
        decode: Callable,
        indexes_to_codes_fn: Callable,
        sequences: List[str],
        model_params: hk.Params,
        num_device: int,
        structure_save_path: Union[str, AnyPath],
        batch_size_per_device,
        max_seq_len: int,
        downsampling_ratio: int,
        pad_token_id: int,
    ):
        structure_dir = os.path.join(structure_save_path, "structures")
        os.makedirs(structure_dir, exist_ok=False)

        effective_batch_size = batch_size_per_device * num_device
        num_iteration = len(sequences) // effective_batch_size + int(
            (len(sequences) % effective_batch_size) > 0
        )

        total = num_iteration * effective_batch_size

        # repeat the list to reach a length of "total"
        sequences = list(islice(cycle(sequences), total))
        effective_length = max_seq_len // downsampling_ratio
        for it in tqdm(range(num_iteration), total=num_iteration):
            start_index = it * effective_batch_size
            tokens_ids = load_and_build_batch(
                sequences[start_index : start_index + effective_batch_size],
                effective_length,
                pad_token_id,
            )

            tokens_file_name = [
                os.path.basename(sequences[i])
                for i in range(start_index, start_index + effective_batch_size)
            ]

            tokens_mask = build_tokens_mask_from_sequence(
                tokens_ids=tokens_ids, pad_token_id=pad_token_id
            )

            nodes_mask = build_nodes_mask_from_tokens_mask(
                tokens_mask=tokens_mask, downsampling_ratio=downsampling_ratio
            )
            number_of_nodes = nodes_mask.sum(axis=-1)

            tokens_ids = tokens_ids.reshape(
                (num_device, batch_size_per_device, effective_length)
            )

            quantized = indexes_to_codes_fn(
                model_params,  random_key, tokens_ids
            )

            quantized = quantized.reshape(
                (num_device, batch_size_per_device, effective_length, -1)
            )

            # trailing dimension
            tokens_mask = tokens_mask.reshape(num_device, batch_size_per_device, -1, 1)
            nodes_mask = nodes_mask.reshape(num_device, batch_size_per_device, -1, 1)

            # decoding sequences
            decoded_structure = decode(
                model_params,
                random_key,
                quantized,
                nodes_mask,
                tokens_mask,
            )
            decoded_structure = jax.block_until_ready(decoded_structure)
            decoded_structure = jax.device_put(decoded_structure, jax.devices("cpu")[0])

            # extract the necessary data for atom computation
            atom_positions = decoded_structure["final_atom_positions"].reshape(
                effective_batch_size, -1, 37, 3
            )
            mask = decoded_structure["final_atom_mask"].reshape(
                effective_batch_size, -1, 37
            )

            total_padded_res = atom_positions.shape[1]

            # dummy ALA aatype. Needed for pdbs format
            aatype_dummy = np.concatenate(
                [np.ones((total_padded_res, 1)), np.zeros((total_padded_res, 20))],
                axis=-1,
            )  # [num_res, 21]
            # Create protein from predictions
            structures = [
                Protein.from_atom37_rep(
                    atom37_positions=atom_positions[k, : number_of_nodes[k]],
                    atom37_gt_exists=mask[k, : number_of_nodes[k]],
                    atom37_atom_exits=mask[k, : number_of_nodes[k]],
                    aatype=aatype_dummy[: number_of_nodes[k]],
                    chain_id="A",
                )
                for k in range(effective_batch_size)
            ]

            # saving pdbs
            for index, structure in enumerate(structures):
                tokens_name = tokens_file_name[index]
                tokens_name = tokens_name.split("_tokens.npy")[0]
                pdb_filepath = os.path.join(
                    structure_dir, f"structure_{tokens_name}.pdb"
                )
                with open(pdb_filepath, "w") as pdb_file:
                    pdb_file.write(to_pdb(structure))
