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
import os
from multiprocessing import set_start_method
from typing import List, Optional

import jax
import jax.numpy as jnp

from scripts.inference_runner import InferenceRunner
from structure_tokenizer.utils.log import get_logger
from structure_tokenizer.utils.utils import load_config

logger = get_logger(__name__)

jax.config.update("jax_platform_name", "cpu")


def main(
    pdbs: List[str],
    token_save_path: str,
    backend: str,
    batch_size_per_device: int = 8,
    config_name: Optional[str] = None,
    config_overrides: Optional[List[str]] = None,
):
    cfg = load_config(
        name="vq3d_inference",
        job_name="tokenize",
        overrides=config_overrides,
        config_path="../../config/structure_tokenizer",
    )

    runner = InferenceRunner()

    local_devices, n_local_device = runner.prepare_devices(backend=backend)

    # prepare all jax / haiku updates / forward functions
    tokenize = runner.prepare_tokenize_fn(cfg=cfg, devices=local_devices)

    random_key = jax.random.PRNGKey(seed=cfg.random_seed)

    # Set up checkpointer and load state
    model_params = runner.load_params(
        model_dir=cfg.model.weight_paths, local_devices=local_devices
    )
    logger.info("params loaded")

    random_key = jnp.stack([random_key for _ in range(n_local_device)])
    # full training routine
    runner.tokenize(
        random_key=random_key,
        quantize=tokenize,
        model_params=model_params,
        pdbs=pdbs,
        token_save_path=token_save_path,
        data_config=cfg.data.data,
        num_device=len(local_devices),
        batch_size_per_device=batch_size_per_device,
    )


if __name__ == "__main__":
    # Starting child processes
    set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Tokenizer specification !")
    parser.add_argument("--model_downsampling", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--codebook_size",
                        type=int,
                        choices=[432, 1728, 4096, 64000],
                        default=4096
                        )
    parser.add_argument("--token_save_path", type=str, required=True)
    parser.add_argument(
        "--pdb_dir",
        type=str,
        required=True,
        help="folder containing the .pdb files to be tokenized",
    )
    parser.add_argument(
        "--backend", type=str, default="cpu", choices=["gpu", "tpu", "cpu"]
    )

    parser.add_argument("--batch_size_per_device", type=int, default=1)

    args = parser.parse_args()

    codebook_surname = {
        "432": "0.5k", "1728": "1.7k", "4096": "4k",  "64000": "64k"
    }

    df = args.model_downsampling
    pdbs: List[str] = [os.path.join(args.pdb_dir, f) for f in os.listdir(args.pdb_dir)]

    model_config = f"gnn/ablation_{codebook_surname[str(args.codebook_size)]}_df_{df}.yaml"
    data_config = f"ablation_df_{df}.yaml"

    logger.warning(f"Using {model_config}")
    overrides: List[str] = [f"model={model_config}", f"data={data_config}"]

    main(
        pdbs=pdbs,
        token_save_path=args.token_save_path,
        batch_size_per_device=args.batch_size_per_device,
        config_overrides=overrides,
        backend=args.backend,
    )
