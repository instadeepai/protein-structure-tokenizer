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
from typing import Callable, Optional, Tuple, Union

import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from ml_collections import ConfigDict

from structure_tokenizer.model import prng
from structure_tokenizer.model.folding import StructureModule
from structure_tokenizer.model.modules import CrossAttentionScaler
from structure_tokenizer.model.quantize import FiniteScalarCodebook, NoQuantizer
from structure_tokenizer.model.sequence_decoder import SequenceDecoder
from structure_tokenizer.model.structure_encoder import StructureEncoder
from structure_tokenizer.model.utils import build_initializer
from structure_tokenizer.types import (
    BatchDataVQ3D,
    Latents,
    Mask,
    ProteinGraph,
    Quantized,
    QuantizerOutput,
    StraightThroughQuantized,
    Structure,
)


class Vq3D(hk.Module):
    """Structure tokenizer model"""

    def __init__(
        self,
        config: ConfigDict,
        global_config: ConfigDict,
    ):
        """Initializes a VQ-3D module

        Args:
            hyperparameters (BioClipConfig): model hyperparameters
            data_config (DataProcessingConfig): data processing configuration
        """

        super(Vq3D, self).__init__()
        self.config = config
        self.global_config = global_config

        # Structure Encoder
        self.structure_encoder = hk.vmap(
            StructureEncoder(self.config.model.encoder, self.global_config),
            in_axes=(0, None, None),
            split_rng=False,
        )
        # Downscaler
        self.downscaler = CrossAttentionScaler(
            self.config.model.down_sampler,
            self.global_config,
            "cross_attn_downsampling",
        )
        # Post downscaling normalisation
        self.downsampler_norm = self.make_normalization(
            self.config.model.down_sampler.normalization,
            "downsampler_norm",
        )

        # Down linear projection
        self.down_proj = self.make_linproj(
            self.config.model.down_proj,
            name="down_proj",
        )
        self._make_codebook()

        # Up linear projection
        self.up_proj = self.make_linproj(
            self.config.model.up_proj,
            name="up_proj",
        )

        #  Upscaler
        self.upscaler = CrossAttentionScaler(
            self.config.model.up_sampler, self.global_config, "cross_attn_upsampling"
        )

        #  Post upsampling norm
        self.post_upsampler_norm = self.make_normalization(
            self.config.model.up_sampler.normalization,
            "upsampler_norm",
        )

        #  Decoder
        self.seq_decoder = hk.vmap(
            SequenceDecoder(
                self.config.model.decoder,
                self.global_config,
            ),
            in_axes=(0, None, None, 0, None, None),
            split_rng=False,
        )

        #  Structure module
        structure_module = StructureModule(
            self.config.model.structure_module, self.global_config
        )
        self.structure_module = hk.vmap(
            structure_module,
            in_axes=(0, 0, 0),
            # in_axes=({"pair": 0, "single": 0}, {"seq_mask": 0}, 0, None),
            # out_axes=({"traj": 0, "representations": {"structure_module": 0}}),
            split_rng=False,
        )
        self.use_down_causal_attn = self.config.model.down_sampler.causal_attn
        self.use_down_local_attn = self.config.model.down_sampler.use_local_attn
        self.use_up_local_attn = self.config.model.up_sampler.use_local_attn

    @hk.transparent
    def _make_codebook(self) -> None:
        """
        Builds the codebook

        :param cfg:

        :return:
        """
        if self.config.model.codebook.use_codebook:
            self.codebook = FiniteScalarCodebook(
                self.config.model.codebook,
                cross_replica_axis="p",
                name="fsq_codebook",
            )
        else:
            self.codebook = NoQuantizer()

    def make_linproj(self, config, name: Optional[str] = None):
        if config.emb_proj:

            def projection(x):
                return hk.Linear(
                    config.emb_dim,
                    name=name,
                    w_init=build_initializer(
                        distribution=self.global_config.init.distribution,
                        scale=self.global_config.init.scale,
                        mode=self.global_config.init.mode,
                    ),
                )(x).astype(x.dtype)

            return projection

        return lambda x: x

    def make_normalization(
        self, normalization: str, name: Optional[str] = None
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        if normalization == "spherical":

            def normalization_fn(x: jnp.ndarray) -> jnp.ndarray:
                return jnp.divide(
                    x, jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-6
                )

        elif normalization == "layer_norm":

            def normalization_fn(x: jnp.ndarray) -> jnp.ndarray:
                return hk.LayerNorm(
                    axis=-1,
                    create_scale=True,
                    create_offset=True,
                    name=name,
                    param_axis=-1,
                )(x).astype(x.dtype)

        else:

            def normalization_fn(x: jnp.ndarray) -> jnp.ndarray:
                return x

        return normalization_fn

    def __call__(
        self,
        batch: BatchDataVQ3D,
        is_training: bool = False,
        safe_key=None,
    ) -> Tuple[Structure, QuantizerOutput]:
        """Tokenized structure into embedding sequence. Performs
            a full encoder, tokenizer, quantization, decoder,
            structure module update.


        Args:
            batch (BatchDataVQ3D):  batch representing a protein

        Returns:
            coordinates_protein (Coordinates): predicted protein coordinates
            hidden_features (NodeFeatures): protein hidden features
        """

        if safe_key is None:
            safe_key = prng.SafeKey(hk.next_rng_key())
        safe_key, *sub_keys = safe_key.split(30)
        sub_keys = iter(sub_keys)

        nodes_mask = batch.graph.nodes_mask
        tokens_mask = batch.graph.tokens_mask

        # Encode and downsample data
        continuous_embedding, continuous_embedding_pre_proj = self.encode(
            batch.graph,
            is_training,
            next(sub_keys),
        )

        quantized, quantized_emb = self.quantize(
            continuous_embedding,
            tokens_mask,
            is_training,
            next(sub_keys),
        )

        quantized_emb["continuous_embedding_pre_proj"] = continuous_embedding_pre_proj

        # Upsample and decode
        (
            quantized_proj,
            s_i,
            z_ij,
        ) = self.decode(
            quantized,
            nodes_mask,
            tokens_mask,
            is_training,
            next(sub_keys),
        )

        #
        quantized_emb["quantize_post_proj"] = quantized_proj

        # Structure module
        representation = {"single": s_i, "pair": z_ij}
        decoded_structure = self.structure_module(
            representation, batch.features, nodes_mask
        )

        return decoded_structure, quantized_emb

    def indexes_to_codes(self, tokens):
        return self.codebook.indexes_to_codes(tokens)

    def get_downsampling_mask(self, down_attn_mask: Mask) -> Mask:
        """
            Modifies (or not) the downsampling mask (1,..., 1, 0, ..., 0)

        :param down_attn_mask:

        :return:
            Modified the Downsampling for either "local" or "causal" attn
        """

        if self.use_down_local_attn or self.use_down_causal_attn:
            elem_mask = jnp.pad(
                jnp.ones(self.global_config.data.downsampling_ratio),
                (
                    0,
                    self.global_config.data.seq_max_size
                    - self.global_config.data.downsampling_ratio,
                ),
                constant_values=0,
            )

            masker = functools.partial(lambda x, i: jax.numpy.roll(x, i), elem_mask)
            index = jnp.arange(
                0,
                self.global_config.data.seq_max_size,
                self.global_config.data.downsampling_ratio,
            )
            local_attn_downsampling_mask = (
                jax.vmap(masker)(index)[None]
                .repeat(down_attn_mask.shape[0], axis=0)[:, None]
                .repeat(self.config.model.down_sampler.cross_attn.num_head, 1)
            )

            if self.use_down_local_attn:
                down_attn_mask = local_attn_downsampling_mask * down_attn_mask
            else:
                local_attn_downsampling_mask = jnp.cumsum(
                    local_attn_downsampling_mask, axis=-2
                )

                down_attn_mask = local_attn_downsampling_mask * down_attn_mask

        if self.config.model.down_sampler.use_global_node > 0:
            down_attn_mask = jnp.pad(
                down_attn_mask,
                (
                    (0, 0),
                    (0, 0),
                    (self.config.model.down_sampler.use_global_node, 0),
                    (0, 0),
                ),
                constant_values=1,
            )

        return down_attn_mask

    def get_upsampling_mask(self, up_attn_mask: Mask) -> Mask:
        """
        Builds custom upsampling mask for the cross attention upscaler

        :param up_attn_mask : (Mask)
        :return:
            modified up_attn_mask: (Mask)
        """

        if self.use_up_local_attn:
            elem_mask = jnp.pad(
                jnp.ones(self.global_config.data.downsampling_ratio),
                (
                    0,
                    self.global_config.data.seq_max_size
                    - self.global_config.data.downsampling_ratio,
                ),
                constant_values=0,
            )

            masker = functools.partial(lambda x, i: jax.numpy.roll(x, i), elem_mask)
            index = jnp.arange(
                0,
                self.global_config.data.seq_max_size,
                self.global_config.data.downsampling_ratio,
            )
            local_upsampling_mask = (
                jax.vmap(masker)(index)[None]
                .repeat(up_attn_mask.shape[0], axis=0)[:, None]
                .repeat(self.config.model.up_sampler.cross_attn.num_head, 1)
            )
            local_upsampling_mask = jnp.swapaxes(local_upsampling_mask, -2, -1)

            up_attn_mask = up_attn_mask * local_upsampling_mask

        return up_attn_mask

    def encode(
        self, graph: ProteinGraph, is_training: bool = False, safe_key=None
    ) -> Tuple[Latents, Latents]:
        """
        encode structure into continuous embedding sequence.

        Args:

        :param graph:  (ProteinGraph):  batch representing a protein
        :param is_training:  (bool):  mode
        :param safe_key: (RNGKey):

        :return:
            continuous embedded sequence

        """

        if safe_key is None:
            safe_key = prng.SafeKey(hk.next_rng_key())
        safe_key, *sub_keys = safe_key.split(20)
        sub_keys = iter(sub_keys)

        nodes_mask = graph.nodes_mask
        tokens_mask = graph.tokens_mask
        # Create sampler masks if needed:
        base_mask = tokens_mask * jnp.transpose(nodes_mask, axes=(0, -1, -2))

        # Embed structure
        nodes_emd, edges_emb = self.structure_encoder(
            graph,
            is_training,
            next(sub_keys),
        )

        # Downsampling
        down_attn_mask = jnp.repeat(
            base_mask[:, None], self.config.model.down_sampler.cross_attn.num_head, 1
        )
        out_downsampling_mask = jnp.repeat(
            tokens_mask,
            self.config.model.down_sampler.out_emb_size,
            axis=-1,
        )

        down_attn_mask = self.get_downsampling_mask(down_attn_mask)

        # emb of shape (batch, max_tokens_num, num_head*key_size)
        continuous_embedding = self.downscaler(
            nodes_emd,
            down_attn_mask,
            out_downsampling_mask,
            nodes_mask,
            is_training,
            next(sub_keys),
        )

        # Post downsampling normalization
        continuous_embedding_pre_proj = self.downsampler_norm(continuous_embedding)

        # Projecting to low dim for FSQ
        return (
            self.down_proj(continuous_embedding_pre_proj),
            continuous_embedding_pre_proj,
        )

    def quantize(
        self,
        continuous_embedding: Latents,
        tokens_mask: jnp.ndarray,
        is_training: bool,
        safe_key: jnp.ndarray,
    ) -> Tuple[Union[Quantized, StraightThroughQuantized], QuantizerOutput]:
        if safe_key is None:
            safe_key = prng.SafeKey(hk.next_rng_key())
        safe_key, *sub_keys = safe_key.split(20)
        sub_keys = iter(sub_keys)

        # build the mask
        quantized_mask = jnp.repeat(
            tokens_mask,
            self.config.model.codebook.codes_dimension,
            axis=-1,
        )

        # if use_codebook = False: self.codebook is just an output wrapper
        quantized_emb = self.codebook(
            continuous_embedding, quantized_mask, is_training, next(sub_keys)
        )

        if is_training:
            quantized = quantized_emb["straight_through_quantized"]
        else:
            quantized = quantized_emb["quantize"]

        return quantized, quantized_emb

    def encode_and_quantize(
        self, graph: ProteinGraph, is_training: bool = False, safe_key=None
    ) -> QuantizerOutput:
        if safe_key is None:
            safe_key = prng.SafeKey(hk.next_rng_key())
        safe_key, *sub_keys = safe_key.split(20)
        sub_keys = iter(sub_keys)

        tokens_mask = graph.tokens_mask

        # Encode and downsample data
        continuous_embedding, continuous_embedding_pre_proj = self.encode(
            graph,
            is_training,
            next(sub_keys),
        )

        _, quantized_emb = self.quantize(
            continuous_embedding,
            tokens_mask,
            is_training,
            next(sub_keys),
        )

        quantized_emb["continuous_embedding_pre_proj"] = continuous_embedding_pre_proj

        return quantized_emb

    def decode(self, quantized, nodes_mask, tokens_mask, is_training, safe_key):
        if safe_key is None:
            safe_key = prng.SafeKey(hk.next_rng_key())
        safe_key, *sub_keys = safe_key.split(20)
        sub_keys = iter(sub_keys)

        # Projecting back up to emb dim
        quantized_proj = self.up_proj(quantized)

        # Upsampling
        # create masks
        base_mask = tokens_mask * jnp.transpose(nodes_mask, axes=(0, -1, -2))
        up_attn_mask = jnp.repeat(
            base_mask[:, None], self.config.model.up_sampler.cross_attn.num_head, 1
        ).transpose((0, 1, 3, 2))

        up_attn_mask = self.get_upsampling_mask(up_attn_mask)

        out_upsampling_mask = jnp.repeat(
            nodes_mask,
            self.config.model.up_sampler.out_emb_size,
            axis=-1,
        )
        # perform xattn upsampling
        # decoded of shape (batch, max_res_num, num_head*key_size)
        seq_decoded = self.upscaler(
            quantized_proj,
            up_attn_mask,
            out_upsampling_mask,
            tokens_mask,
            is_training,
            next(sub_keys),
        )

        # Post upsampling normalization
        seq_decoded = self.post_upsampler_norm(seq_decoded)

        # sequence decoder
        seq_size = self.global_config.data.seq_max_size
        senders = jnp.repeat(jnp.arange(seq_size)[None], seq_size, axis=0).flatten()
        receivers = jnp.repeat(
            jnp.arange(seq_size)[..., None], seq_size, axis=-1
        ).flatten()

        s_i, z_ij = self.seq_decoder(
            seq_decoded,
            senders,
            receivers,
            nodes_mask,
            is_training,
            next(sub_keys),
        )

        return quantized_proj, s_i, z_ij

    def decode_and_make_structure(
        self, quantized, nodes_mask, tokens_mask, is_training, safe_key
    ) -> Structure:
        _, s_i, z_ij = self.decode(
            quantized, nodes_mask, tokens_mask, is_training, safe_key
        )

        representation = {"single": s_i, "pair": z_ij}

        # for the SM need to build dict:  with dummy aatype & mask atom37_gt_exists
        b, n = s_i.shape[0], s_i.shape[1]
        atom37_gt_exists = jnp.concatenate(  # BG: ugly
            (
                jnp.ones((b, n, 3), dtype=np.int32),  # BG: N, CA, C
                jnp.zeros((b, n, 1), dtype=np.int32),  # BG: CB
                jnp.ones((b, n, 1), dtype=np.int32),  # BG: O
                jnp.zeros((b, n, 37 - 5), dtype=np.int32),  # BG non-backbone atoms
            ),
            axis=-1,
        )
        aatype = jnp.concatenate(  # BG: dummy ALA aatype
            [
                jnp.ones((b, n, 1)),
                jnp.zeros((b, n, 20)),
            ],
            axis=-1,
        )
        features = {
            "atom37_gt_exists": atom37_gt_exists,
            "aatype": aatype,
        }
        decoded_structure = self.structure_module(representation, features, nodes_mask)

        return decoded_structure

    def __repr__(self) -> str:
        return "Vq3D " + str(self.__dict__)


class ForwardVQ3D(hk.Module):
    """Autoencoder module of Vq3D model"""

    def __init__(self, config, global_config):
        """Class to perform Vq3D forward pass

        Args:
            config (BioClipConfig): model hyperparameters
            global_config (DataProcessingConfig): data processing configuration
        """

        super(ForwardVQ3D, self).__init__()
        self.config = config
        self.global_config = global_config

        if self.global_config.mixed_precision:
            # Use mixed precision (only support A100 GPU and TPU for now)
            half = jnp.bfloat16
            full = jnp.float32

            policy = jmp.Policy(compute_dtype=half, param_dtype=full, output_dtype=full)
            hk.mixed_precision.set_policy(Vq3D, policy)

            # Remove it in batch norm to avoid instabilities
            policy = jmp.Policy(compute_dtype=full, param_dtype=full, output_dtype=half)
            hk.mixed_precision.set_policy(hk.BatchNorm, policy)
            hk.mixed_precision.set_policy(hk.LayerNorm, policy)

    def __call__(
        self,
        batch: BatchDataVQ3D,
        is_training: bool = False,
    ) -> Tuple[Structure, QuantizerOutput]:
        """Full forward pass of Vq3D

        Args:
            batch_data (BatchDataVQ3D): batch data

        Returns:
            decoded_structure: reconstructed structure
            seq_emb: quantized representation
            quantized_emb: aux embedding and metrics dict
        """
        decoded_structure, embedding_dict = Vq3D(
            config=self.config, global_config=self.global_config
        )(batch, is_training, None)

        return decoded_structure, embedding_dict

    def __repr__(self) -> str:
        return "ForwardVQ3D" + str(self.__dict__)
