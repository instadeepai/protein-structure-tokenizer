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

from typing import List, Optional

import haiku as hk
import jax.numpy as jnp
from ml_collections import ConfigDict

from structure_tokenizer.model import prng
from structure_tokenizer.model.modules import PairwiseRepresentation, Transition
from structure_tokenizer.model.positional_encoding_layer import PositionalEncodingLayer
from structure_tokenizer.types import Mask, RNGKey, SingleResidueRepresentation


class SequenceDecoder(hk.Module):
    """Sequence decoder"""

    def __init__(
        self,
        config: ConfigDict,
        global_config: ConfigDict,
    ):
        """Initializes a sequence decoder Network

        Args:
            config (ConfigDict): model hyperparameters
            global_config (ConfigDict): data processing configuration
        """

        super(SequenceDecoder, self).__init__()
        self.config = config
        self.global_config = global_config
        self.total_num_residue = self.global_config.data.seq_max_size

    # TODO: typing
    def __call__(
        self,
        single_emb: SingleResidueRepresentation,
        senders: List[int],
        receivers: List[int],
        single_masks: Mask,
        is_training: bool,
        safe_key: Optional[RNGKey] = None,
    ):
        """Computes embedding of a protein graph"""

        if safe_key is None:
            safe_key = prng.SafeKey(hk.next_rng_key())
        safe_key, safe_subkey = safe_key.split()

        # Pair initialisation
        _outerstack_mod = PairwiseRepresentation(
            self.config.pair_representation,
            self.global_config,
            "pairwise_representation",
        )
        pair_features = _outerstack_mod(
            single_emb, jnp.repeat(single_masks, single_emb.shape[-1], -1)
        )

        # Relative positional encoding
        pos_enc_dim = self.config.positional_encoding_dimension
        (
            _,
            pair_relpo_emb,
        ) = PositionalEncodingLayer(positional_encoding_dimension=pos_enc_dim)(
            n_node=self.global_config.data.seq_max_size,
            senders=senders,
            receivers=receivers,
            diffusion=False,
            diffusion_time_step=0,
        )

        # Concatenanting pair init and relpos enc and projecting
        assert pair_relpo_emb.shape[0] == single_emb.shape[0] ** 2
        # reshape edge_features (num_res*num_res, d) ->  (num_res, num_res, d)
        pair_relpo_emb = pair_relpo_emb.reshape(
            (single_emb.shape[0], single_emb.shape[0], pair_relpo_emb.shape[-1]),
        )
        pair_features = jnp.concatenate(
            [
                pair_relpo_emb,
                pair_features,
            ],
            axis=-1,
        )

        # downscale and prepare for potential cross-attention
        pair_features = hk.Linear(pair_relpo_emb.shape[-1])(pair_features)

        # Pair transition
        pair_trans_module = Transition(
            self.config.pair_transition, self.global_config, "pair_transition_init"
        )

        pair_features = pair_trans_module(pair_features, single_masks, is_training)
        return single_emb, pair_features

    def __repr__(self) -> str:
        return "Se " + str(self.__dict__)
