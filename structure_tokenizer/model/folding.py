# Copyright 2021 DeepMind Technologies Limited
#
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

"""Modules and utilities for the structure module."""

import functools
from typing import Callable, Optional

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import TypeAlias

from structure_tokenizer.model import all_atom, common_modules, prng, quat_affine, r3

FapeLoss: TypeAlias = Callable[..., jnp.ndarray]
TrajAffineTransformation: TypeAlias = jnp.ndarray


def squared_difference(x, y):
    return jnp.square(x - y)


class InvariantPointAttention(hk.Module):
    """Invariant Point attention module.

    The high-level idea is that this attention module works over a set of points
    and associated orientations in 3D space (e.g. protein residues).

    Each residue outputs a set of queries and keys as points in their local
    reference frame.  The attention is then defined as the euclidean distance
    between the queries and keys in the global frame.

    Jumper et al. (2021) Suppl. Alg. 22 "InvariantPointAttention"
    """

    def __init__(
        self, config, global_config, dist_epsilon=1e-8, name="invariant_point_attention"
    ):
        """Initialize.

        Args:
          config: Structure Module Config
          global_config: Global Config of Model.
          dist_epsilon: Small value to avoid NaN in distance calculation.
          name: Haiku Module name.
        """
        super().__init__(name=name)

        self._dist_epsilon = dist_epsilon
        self._zero_initialize_last = global_config.zero_init

        self.config = config

        self.global_config = global_config

    def __call__(self, inputs_1d, inputs_2d, mask, affine):
        """Compute geometry-aware attention.

        Given a set of query residues (defined by affines and associated scalar
        features), this function computes geometry-aware attention between the
        query residues and target residues.

        The residues produce points in their local reference frame, which
        are converted into the global frame in order to compute attention via
        euclidean distance.

        Equivalently, the target residues produce points in their local frame to be
        used as attention values, which are converted into the query residues'
        local frames.

        Args:
          inputs_1d: (N, C) 1D input embedding that is the basis for the
            scalar queries.
          inputs_2d: (N, M, C') 2D input embedding, used for biases and values.
          mask: (N, 1) mask to indicate which elements of inputs_1d participate
            in the attention.
          affine: QuatAffine object describing the position and orientation of
            every element in inputs_1d.

        Returns:
          Transformation of the input embedding.
        """
        num_residues, _ = inputs_1d.shape

        # Improve readability by removing a large number of 'self's.
        num_head = self.config.num_head
        num_scalar_qk = self.config.num_scalar_qk
        num_point_qk = self.config.num_point_qk
        num_scalar_v = self.config.num_scalar_v
        num_point_v = self.config.num_point_v
        num_output = self.config.num_channel

        assert num_scalar_qk > 0
        assert num_point_qk > 0
        assert num_point_v > 0

        # Construct scalar queries of shape:
        # [num_query_residues, num_head, num_points]
        q_scalar = common_modules.Linear(num_head * num_scalar_qk, name="q_scalar")(
            inputs_1d
        )
        q_scalar = jnp.reshape(q_scalar, [num_residues, num_head, num_scalar_qk])

        # Construct scalar keys/values of shape:
        # [num_target_residues, num_head, num_points]
        kv_scalar = common_modules.Linear(
            num_head * (num_scalar_v + num_scalar_qk), name="kv_scalar"
        )(inputs_1d)
        kv_scalar = jnp.reshape(
            kv_scalar, [num_residues, num_head, num_scalar_v + num_scalar_qk]
        )
        k_scalar, v_scalar = jnp.split(kv_scalar, [num_scalar_qk], axis=-1)

        # Construct query points of shape:
        # [num_residues, num_head, num_point_qk]

        # First construct query points in local frame.
        q_point_local = common_modules.Linear(
            num_head * 3 * num_point_qk, name="q_point_local"
        )(inputs_1d)
        q_point_local = jnp.split(q_point_local, 3, axis=-1)
        # Project query points into global frame.
        q_point_global = affine.apply_to_point(q_point_local, extra_dims=1)
        # Reshape query point for later use.
        q_point = [
            jnp.reshape(x, [num_residues, num_head, num_point_qk])
            for x in q_point_global
        ]

        # Construct key and value points.
        # Key points have shape [num_residues, num_head, num_point_qk]
        # Value points have shape [num_residues, num_head, num_point_v]

        # Construct key and value points in local frame.
        kv_point_local = common_modules.Linear(
            num_head * 3 * (num_point_qk + num_point_v), name="kv_point_local"
        )(inputs_1d)
        kv_point_local = jnp.split(kv_point_local, 3, axis=-1)
        # Project key and value points into global frame.
        kv_point_global = affine.apply_to_point(kv_point_local, extra_dims=1)
        kv_point_global = [
            jnp.reshape(x, [num_residues, num_head, (num_point_qk + num_point_v)])
            for x in kv_point_global
        ]
        # Split key and value points.
        k_point, v_point = list(
            zip(
                *[
                    jnp.split(
                        x,
                        [
                            num_point_qk,
                        ],
                        axis=-1,
                    )
                    for x in kv_point_global
                ]
            )
        )

        # We assume that all queries and keys come iid from N(0, 1) distribution
        # and compute the variances of the attention logits.
        # Each scalar pair (q, k) contributes Var q*k = 1
        scalar_variance = max(num_scalar_qk, 1) * 1.0
        # Each point pair (q, k) contributes Var [0.5 ||q||^2 - <q, k>] = 9 / 2
        point_variance = max(num_point_qk, 1) * 9.0 / 2

        # Allocate equal variance to scalar, point and attention 2d parts so that
        # the sum is 1.

        num_logit_terms = 3

        scalar_weights = np.sqrt(1.0 / (num_logit_terms * scalar_variance))
        point_weights = np.sqrt(1.0 / (num_logit_terms * point_variance))
        attention_2d_weights = np.sqrt(1.0 / (num_logit_terms))

        # Trainable per-head weights for points.
        trainable_point_weights = jax.nn.softplus(
            hk.get_parameter(
                "trainable_point_weights",
                shape=[num_head],
                # softplus^{-1} (1)
                init=hk.initializers.Constant(np.log(np.exp(1.0) - 1.0)),
            )
        )
        point_weights *= jnp.expand_dims(trainable_point_weights, axis=1)

        v_point = [jnp.swapaxes(x, -2, -3) for x in v_point]

        q_point = [jnp.swapaxes(x, -2, -3) for x in q_point]
        k_point = [jnp.swapaxes(x, -2, -3) for x in k_point]
        dist2 = [
            squared_difference(qx[:, :, None, :], kx[:, None, :, :])
            for qx, kx in zip(q_point, k_point)
        ]
        dist2 = sum(dist2)
        attn_qk_point = -0.5 * jnp.sum(point_weights[:, None, None, :] * dist2, axis=-1)

        v = jnp.swapaxes(v_scalar, -2, -3)
        q = jnp.swapaxes(scalar_weights * q_scalar, -2, -3)
        k = jnp.swapaxes(k_scalar, -2, -3)
        attn_qk_scalar = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
        attn_logits = attn_qk_scalar + attn_qk_point

        attention_2d = common_modules.Linear(num_head, name="attention_2d")(inputs_2d)

        attention_2d = jnp.transpose(attention_2d, [2, 0, 1])
        attention_2d = attention_2d_weights * attention_2d
        attn_logits += attention_2d

        mask_2d = mask * jnp.swapaxes(mask, -1, -2)
        attn_logits -= 1e5 * (1.0 - mask_2d)

        # [num_head, num_query_residues, num_target_residues]
        attn = jax.nn.softmax(attn_logits)

        # [num_head, num_query_residues, num_head * num_scalar_v]
        result_scalar = jnp.matmul(attn, v)

        # For point result, implement matmul manually so that it will be a float32
        # on TPU.  This is equivalent to
        # result_point_global = [jnp.einsum('bhqk,bhkc->bhqc', attn, vx)
        #                        for vx in v_point]
        # but on the TPU, doing the multiply and reduce_sum ensures the
        # computation happens in float32 instead of bfloat16.
        result_point_global = [
            jnp.sum(attn[:, :, :, None] * vx[:, None, :, :], axis=-2) for vx in v_point
        ]

        # [num_query_residues, num_head, num_head * num_(scalar|point)_v]
        result_scalar = jnp.swapaxes(result_scalar, -2, -3)
        result_point_global = [jnp.swapaxes(x, -2, -3) for x in result_point_global]

        # Features used in the linear output projection. Should have the size
        # [num_query_residues, ?]
        output_features = []

        result_scalar = jnp.reshape(
            result_scalar, [num_residues, num_head * num_scalar_v]
        )
        output_features.append(result_scalar)

        result_point_global = [
            jnp.reshape(r, [num_residues, num_head * num_point_v])
            for r in result_point_global
        ]
        result_point_local = affine.invert_point(result_point_global, extra_dims=1)
        output_features.extend(result_point_local)

        output_features.append(
            jnp.sqrt(
                self._dist_epsilon
                + jnp.square(result_point_local[0])
                + jnp.square(result_point_local[1])
                + jnp.square(result_point_local[2])
            )
        )

        # Dimensions: h = heads, i and j = residues,
        # c = inputs_2d channels
        # Contraction happens over the second residue dimension, similarly to how
        # the usual attention is performed.
        result_attention_over_2d = jnp.einsum("hij, ijc->ihc", attn, inputs_2d)
        num_out = num_head * result_attention_over_2d.shape[-1]
        output_features.append(
            jnp.reshape(result_attention_over_2d, [num_residues, num_out])
        )

        final_init = "zeros" if self._zero_initialize_last else "linear"

        final_act = jnp.concatenate(output_features, axis=-1)

        return common_modules.Linear(
            num_output, initializer=final_init, name="output_projection"
        )(final_act)


class FoldIteration(hk.Module):
    """A single iteration of the main structure module loop.

    Jumper et al. (2021) Suppl. Alg. 20 "StructureModule" lines 6-21

    First, each residue attends to all residues using InvariantPointAttention.
    Then, we apply transition layers to update the hidden representations.
    Finally, we use the hidden representations to produce an update to the
    affine of each residue.
    """

    def __init__(self, config, global_config, name="fold_iteration"):
        super().__init__(name=name)
        self.config = config
        self.global_config = global_config

    def __call__(
        self,
        activations,
        sequence_mask,
        update_affine,
        initial_act,
        safe_key=None,
        static_feat_2d=None,
    ):
        c = self.config

        if safe_key is None:
            safe_key = prng.SafeKey(hk.next_rng_key())

        affine = quat_affine.QuatAffine.from_tensor(activations["affine"])

        act = activations["act"]
        attention_module = InvariantPointAttention(self.config, self.global_config)
        # Attention
        attn = attention_module(
            inputs_1d=act, inputs_2d=static_feat_2d, mask=sequence_mask, affine=affine
        )
        act += attn
        safe_key, *sub_keys = safe_key.split(3)
        sub_keys = iter(sub_keys)
        # act = safe_dropout_fn(act, next(sub_keys))
        act = hk.LayerNorm(
            axis=[-1],
            create_scale=True,
            create_offset=True,
            name="attention_layer_norm",
        )(act)

        final_init = "zeros" if self.global_config.zero_init else "linear"

        # Transition
        input_act = act
        for i in range(c.num_layer_in_transition):
            init = "relu" if i < c.num_layer_in_transition - 1 else final_init
            act = common_modules.Linear(
                c.num_channel, initializer=init, name="transition"
            )(act)
            if i < c.num_layer_in_transition - 1:
                act = jax.nn.relu(act)
        act += input_act
        # act = safe_dropout_fn(act, next(sub_keys))
        act = hk.LayerNorm(
            axis=[-1],
            create_scale=True,
            create_offset=True,
            name="transition_layer_norm",
        )(act)

        if update_affine:
            # This block corresponds to
            # Jumper et al. (2021) Alg. 23 "Backbone update"
            affine_update_size = 6

            # Affine update
            affine_update = common_modules.Linear(
                affine_update_size, initializer=final_init, name="affine_update"
            )(act)

            affine = affine.pre_compose(affine_update)

        # The sidechain module predicts position for the full bb atoms
        sc = MultiRigidSidechain(c.sidechain, self.global_config)(
            affine.scale_translation(c.position_scale), [act, initial_act]  # , aatype
        )

        outputs = {"affine": affine.to_tensor(), "sc": sc}

        # outputs = {"affine": affine.to_tensor()}

        # BG: AF stop the gradient for all layers < num_layer, do we want to follow AF?
        affine = affine.apply_rotation_tensor_fn(jax.lax.stop_gradient)

        new_activations = {"act": act, "affine": affine.to_tensor()}
        return new_activations, outputs


def generate_affines(
    representations,
    sequence_mask,
    config,
    global_config,
    safe_key,
):
    """Generate predicted affines for a single chain.

    Jumper et al. (2021) Suppl. Alg. 20 "StructureModule"

    This is the main part of the structure module - it iteratively applies
    folding to produce a set of predicted residue positions.

    Args:
      representations: Representations dictionary.
      mask: res mask.
      config: Config for the structure module.
      global_config: Global config.
      safe_key: A prng.SafeKey object that wraps a PRNG key.

    Returns:
      A dictionary containing residue affines and sidechain positions.
    """
    c = config
    # sequence_mask = batch.graph.nodes_mask

    act = hk.LayerNorm(
        axis=[-1], create_scale=True, create_offset=True, name="single_layer_norm"
    )(representations["single"])

    initial_act = act
    act = common_modules.Linear(c.num_channel, name="initial_projection")(act)

    affine = generate_new_affine(sequence_mask)

    fold_iteration = FoldIteration(c, global_config, name="fold_iteration")

    activations = {
        "act": act,
        "affine": affine.to_tensor(),
    }

    act_2d = hk.LayerNorm(
        axis=[-1], create_scale=True, create_offset=True, name="pair_layer_norm"
    )(representations["pair"])

    outputs = []
    safe_keys = safe_key.split(c.num_layer)
    for sub_key in safe_keys:
        activations, output = fold_iteration(
            activations,
            initial_act=initial_act,
            static_feat_2d=act_2d,
            safe_key=sub_key,
            sequence_mask=sequence_mask,
            update_affine=True,
        )
        outputs.append(output)

    # BG: stacking the SM layers outputs
    output = jax.tree_map(lambda *x: jnp.stack(x), *outputs)

    # Include the activations in the output dict for use by the LDDT-Head.
    output["act"] = activations["act"]

    return output


class StructureModule(hk.Module):
    """StructureModule as a network head.

    Jumper et al. (2021) Suppl. Alg. 20 "StructureModule"
    """

    def __init__(
        self, config, global_config, compute_loss=True, name="structure_module"
    ):
        super().__init__(name=name)
        self.config = config
        self.global_config = global_config
        self.compute_loss = compute_loss

    def __call__(self, representations, features, sequence_mask, safe_key=None):
        c = self.config
        ret = {}

        if safe_key is None:
            safe_key = prng.SafeKey(hk.next_rng_key())

        generate_affines_fn = generate_affines

        output = generate_affines_fn(
            representations=representations,
            sequence_mask=sequence_mask,
            config=self.config,
            global_config=self.global_config,
            safe_key=safe_key,
        )

        # BG: internal representation of SM. Currently not used.
        ret["representations"] = {"structure_module": output["act"]}

        # BG: SM affine outputs for each layers.
        ret["traj"] = output["affine"] * jnp.array([1.0] * 4 + [c.position_scale] * 3)
        # BG: intermdiate SM affine for aux bb fape loss
        ret["int_affines"] = ret["traj"][:-1]
        # BG: final SM affine. Not used anymore
        ret["final_affines"] = ret["traj"][-1]

        # BG: full SC outputs for each layers.
        ret["sidechains"] = output["sc"]

        # BG: Final atom pred as atom14 format.
        atom14_pred_positions = r3.vecs_to_tensor(output["sc"]["atom_pos"])[-1]

        # BG: Final atom pred as atom37 format.
        atom37_pred_positions = all_atom.atom14_to_atom37(
            atom14_pred_positions,
            jnp.argmax(features["aatype"], axis=-1),
        )
        # BG: We directly mask the atom preds here, is it needed?
        atom37_mask = features["atom37_gt_exists"]
        atom37_pred_positions *= atom37_mask[:, :, None]
        ret["final_atom_positions"] = atom37_pred_positions  # (N, 37, 3)
        ret["final_atom_mask"] = atom37_mask  # (N, 37)

        # BG: we always compute the loss live, cqn be removed?
        if self.compute_loss:
            return ret
        else:
            no_loss_features = [
                "final_atom_positions",
                "final_atom_mask",
                "representations",
            ]
            no_loss_ret = {k: ret[k] for k in no_loss_features}
            return no_loss_ret


def make_fape(
    loss_unit_distance: float,
    clamp_distance: Optional[float],
) -> FapeLoss:
    """
        Util to make the Fape functions

    :param clamp_distance:
    :param loss_unit_distance:

    :return:
        A callable loss

    """
    fape_loss_fn = functools.partial(
        all_atom.frame_aligned_point_error,
        l1_clamp_distance=clamp_distance if clamp_distance is not None else None,
        length_scale=loss_unit_distance,
    )

    fape_loss_fn = jax.vmap(fape_loss_fn, (0, None, None, 0, None, None))

    return fape_loss_fn


def backbone_loss(
    clamped_fape_loss_fn, unclamped_fape_loss_fn, pct_unclamped, batch, value, safe_key
):
    """Backbone FAPE Loss.

    Jumper et al. (2021) Suppl. Alg. 20 "StructureModule" line 17

    Args:
      ret: Dictionary to write outputs into, needs to contain 'loss'.
      batch: Batch, needs to contain 'backbone_affine_tensor',
        'backbone_affine_mask'.
      value: Dictionary containing structure module output, needs to contain
        'traj', a trajectory of rigids.
      config: Configuration of loss, should contain 'fape.clamp_distance' and
        'fape.loss_unit_distance'.
    """

    # we only compute the ca fape for the intermdiate SM layers
    affine_trajectory = quat_affine.QuatAffine.from_tensor(value["int_affines"])
    rigid_trajectory = r3.rigids_from_quataffine(affine_trajectory)

    gt_affine = quat_affine.QuatAffine.from_tensor(
        batch.features["backbone_affine_tensor"]
    )
    gt_rigid = r3.rigids_from_quataffine(gt_affine)
    backbone_mask = batch.features["backbone_affine_mask"]
    positions_mask = batch.graph.nodes_mask

    fape_loss = clamped_fape_loss_fn(
        rigid_trajectory,
        gt_rigid,
        backbone_mask,
        rigid_trajectory.trans,
        gt_rigid.trans,
        positions_mask[:, 0],
    )

    fape_loss_unclamped = unclamped_fape_loss_fn(
        rigid_trajectory,
        gt_rigid,
        backbone_mask,
        rigid_trajectory.trans,
        gt_rigid.trans,
        positions_mask[:, 0],
    )

    use_unclamped = jax.random.uniform(safe_key, shape=fape_loss.shape) < pct_unclamped

    # select the loss
    fape_loss = jnp.where(use_unclamped, fape_loss_unclamped, fape_loss)

    return {"fape": fape_loss[-1], "loss": jnp.mean(fape_loss)}


def sidechain_loss(atom_clamp_distance, length_scale, batch, value, nodes_mask):
    """All Atom FAPE Loss using renamed rigids."""

    atom14_gt_positions = batch["atom14_gt_positions"]
    atom14_gt_exists = batch["atom14_gt_exists"]
    gt_frames = batch["rigidgroups_gt_frames"]

    flat_gt_frames = r3.rigids_from_tensor_flat12(jnp.reshape(gt_frames, [-1, 12]))
    flat_frames_mask = jnp.reshape(batch["rigidgroups_gt_exists"], [-1])

    flat_gt_positions = r3.vecs_from_tensor(jnp.reshape(atom14_gt_positions, [-1, 3]))

    num_res = nodes_mask.shape[0]
    all_atoms_mask_from_node = jnp.ones((num_res, 14)) * nodes_mask

    flat_positions_mask = jnp.reshape(atom14_gt_exists, [-1]) * jnp.reshape(
        all_atoms_mask_from_node, (-1)
    )

    # Compute frame_aligned_point_error score for the final layer.
    pred_frames = value["sidechains"]["frames"]
    pred_positions = value["sidechains"]["atom_pos"]

    # This slice the last sc outputs and flatten
    # (compute the full atomistic on the final SM outputs)
    def _slice_last_layer_and_flatten(x):
        return jnp.reshape(x[-1], [-1])

    flat_pred_frames = jax.tree_map(_slice_last_layer_and_flatten, pred_frames)
    flat_pred_positions = jax.tree_map(_slice_last_layer_and_flatten, pred_positions)
    # FAPE Loss on sidechains
    fape = all_atom.frame_aligned_point_error(
        pred_frames=flat_pred_frames,
        target_frames=flat_gt_frames,
        frames_mask=flat_frames_mask,
        pred_positions=flat_pred_positions,
        target_positions=flat_gt_positions,
        positions_mask=flat_positions_mask,
        l1_clamp_distance=atom_clamp_distance,
        length_scale=length_scale,
    )

    return {"fape": fape, "loss": fape}


def generate_new_affine(sequence_mask):
    num_residues, _ = sequence_mask.shape
    quaternion = jnp.tile(
        jnp.reshape(jnp.asarray([1.0, 0.0, 0.0, 0.0]), [1, 4]), [num_residues, 1]
    )

    translation = jnp.zeros([num_residues, 3])
    return quat_affine.QuatAffine(quaternion, translation, unstack_inputs=True)


def l2_normalize(x, axis=-1, epsilon=1e-12):
    return x / jnp.sqrt(jnp.maximum(jnp.sum(x**2, axis=axis, keepdims=True), epsilon))


class MultiRigidSidechain(hk.Module):
    """Class to make side chain atoms."""

    def __init__(self, config, global_config, name="rigid_sidechain"):
        super().__init__(name=name)
        self.config = config
        self.global_config = global_config

    # def __call__(self, affine, representations_list, aatype):
    def __call__(self, affine, representations_list):
        """Predict side chains using multi-rigid representations.

        Args:
          affine: The affines for each residue (translations in angstroms).
          representations_list: A list of activations to predict side chains from.
          aatype: Amino acid types.

        Returns:
          Dict containing atom positions and frames (in angstroms).
        """

        # BG: extra head on the SM act to predict the bb torsion angle
        act = [
            common_modules.Linear(  # pylint: disable=g-complex-comprehension
                self.config.num_channel, name="input_projection"
            )(jax.nn.relu(x))
            for x in representations_list
        ]
        # Sum the activation list (equivalent to concat then Linear).
        act = sum(act)

        final_init = "zeros" if self.global_config.zero_init else "linear"

        # Mapping with some residual blocks.
        for _ in range(self.config.num_residual_block):
            old_act = act
            act = common_modules.Linear(
                self.config.num_channel, initializer="relu", name="resblock1"
            )(jax.nn.relu(act))
            act = common_modules.Linear(
                self.config.num_channel, initializer=final_init, name="resblock2"
            )(jax.nn.relu(act))
            act += old_act

        # Map activations to torsion angles.
        # We only want the 3 backbone torsion angles Shape: (num_res, 3*2).
        num_res = act.shape[0]
        unnormalized_angles = common_modules.Linear(6, name="unnormalized_angles")(
            jax.nn.relu(act)
        )
        unnormalized_angles = jnp.reshape(unnormalized_angles, [num_res, 3, 2])
        angles = l2_normalize(unnormalized_angles, axis=-1)
        outputs = {
            "angles_sin_cos": angles,  # jnp.ndarray (N, 3, 2)
            "unnormalized_angles_sin_cos": unnormalized_angles,  # jnp.ndarray (N, 3, 2)
        }

        # Map torsion angles to frames.
        backb_to_global = r3.rigids_from_quataffine(affine)

        # Jumper et al. (2021) Suppl. Alg. 24 "computeAllAtomCoordinates"

        # r3.Rigids for the ref backbone and the 3 extra backbone frames
        # shape (N, 4)
        all_frames_to_global = all_atom.torsion_angles_to_frames(
            backb_to_global, angles
        )

        # Use frames and literature positions to create the final atom coordinates.
        # r3.Vecs with shape (N, 14).
        pred_positions = all_atom.frames_and_literature_positions_to_atom14_pos(
            all_frames_to_global
        )

        outputs.update(
            {
                "atom_pos": pred_positions,  # r3.Vecs (N, 14)
                "frames": all_frames_to_global,  # r3.Rigids (N, 4)
            }
        )
        return outputs
