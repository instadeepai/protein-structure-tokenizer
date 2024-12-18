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

# Usage: pymol -cq alphafold/validation/pymol_script.py -- filepath_prediction filepath_target

W = 1000
H = 1000
ROTATIONS = [0, 30, 60, 90, 120]

# Get filenames and directory name.
filepath_prediction, filepath_target = sys.argv[1:]
basename_prediction, basename_target = [
    os.path.splitext(os.path.basename(x))[0] for x in sys.argv[1:]
]
dirname = os.path.dirname(filepath_prediction)
dirpath = os.path.join(dirname, "pymol_plot")
if not os.path.isdir(dirpath):
    os.mkdir(dirpath)

pymol.finish_launching()

# Plot superposed predicted and target structures.
cmd.load(filepath_prediction)
cmd.set_name(basename_prediction, "prediction")
cmd.color("blue", "prediction")
cmd.load(filepath_target)
cmd.set_name(basename_target, "target")
cmd.color("green", "target")
cmd.align("prediction", "target")

# # Print RMSD of alignment to file.
# line = cmd.cealign("target", "prediction")
# with open(f"{dirname}/superposition_rmsd.json", "w") as f:
#     f.write(json.dumps((line["RMSD"], line["alignment_length"])))

for angle in ROTATIONS:
    cmd.rotate("y", angle=angle)
    cmd.ray(W, H)
    cmd.png(f"{dirpath}/superposition_angle{int(angle)}.png", W, H)

cmd.delete("all")
