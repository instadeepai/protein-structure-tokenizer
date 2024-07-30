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

import os
import sys

import pymol
from pymol import cmd

# Usage: pymol -cq alphafold/validation/pymol_script.py -- filepath_generation

W = 1000
H = 1000
# ROTATIONS = [0, 30, 60, 90, 120]
ROTATIONS = [0,]

# Get filenames and directory name.
filepath_generation = sys.argv[1]
basename_generation = (
    os.path.splitext(os.path.basename(filepath_generation))[0]
)
dirname = os.path.dirname(filepath_generation)
dirpath = os.path.join(dirname, "pymol_plot")
os.makedirs(dirpath, exist_ok=True)

pymol.finish_launching()

# Plot superposed predicted and target structures.
cmd.load(filepath_generation)
cmd.set_name(basename_generation, "sample")
cmd.color("blue", "sample")
cmd.center("sample")
cmd.zoom("sample", complete=0)

# # Print RMSD of alignment to file.
# line = cmd.cealign("target", "prediction")
# with open(f"{dirname}/superposition_rmsd.json", "w") as f:
#     f.write(json.dumps((line["RMSD"], line["alignment_length"])))

for angle in ROTATIONS:
    cmd.rotate("y", angle=angle)
    cmd.ray(W, H)
    cmd.png(f"{dirpath}/angle{int(angle)}.png", W, H)

cmd.delete("all")
