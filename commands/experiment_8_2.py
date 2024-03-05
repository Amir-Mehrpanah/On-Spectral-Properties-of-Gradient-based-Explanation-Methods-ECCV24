# Experiment 7.2: Integrated gradients + Smooth Grad with different alpha priors

import argparse
import sys
import os

sys.path.append(os.getcwd())
from commands.experiment_8 import (
    experiment_master,
)
import commands.experiment_8


commands.experiment_8.alpha_mask_value = (
    "0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"  # DEBUG
)

# Method args
commands.experiment_8.ig_alpha_priors = {  # DEBUG
    # "og_sg_u_x2_nan": "0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0", #
    # "op_sg_u_x2_nan": "0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0",
    # "ig_vg_u_x_0": "0.0",
    # "ig_vg_u_x2_0": "0.0",
    # "ig_vg_i_u_x_0": "0.0",
    # "ig_vg_i_u_x2_0": "0.0",
    # "ig_sg_u_x_0.1": "0.1",
    # "ig_sg_u_x_0.2": "0.2",
    # "ig_sg_u_x_0.5": "0.5",
    # "ig_sg_u_x_1.0": "1.0",
    "ig_sg_u_x2_0.1": "0.1",
    "ig_sg_u_x2_0.2": "0.3",
    "ig_sg_u_x2_0.5": "0.5",
    "ig_sg_u_x2_0.7": "0.7",
    "ig_sg_u_x2_1.0": "1.0",
    # "ig_sg_u_x_0_0.2": "0.0 0.1 0.2",
    # "ig_sg_u_x_0_0.5": "0.0 0.1 0.2 0.3 0.4 0.5",
    # "ig_sg_u_x_0_1.0": "0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0",
    # "ig_sg_u_x2_0_0.2": "0.0 0.1 0.2",
    # "ig_sg_u_x2_0_0.5": "0.0 0.1 0.2 0.3 0.4 0.5",
    # "ig_sg_u_x2_0_1.0": "0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0",
    # "ig_sg_i_u_x_0_0.2": "0.0 0.1 0.2",
    # "ig_sg_i_u_x_0_0.5": "0.0 0.1 0.2 0.3 0.4 0.5",
    # "ig_sg_i_u_x_0_1.0": "0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0",
    # "ig_sg_i_u_x2_0_0.2": "0.0 0.1 0.2",
    # "ig_sg_i_u_x2_0_0.5": "0.0 0.1 0.2 0.3 0.4 0.5",
    # "ig_sg_i_u_x2_0_1.0": "0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0",
}
commands.experiment_8.combination_fns = [
    "additive",
    "convex",
]
commands.experiment_8.baseline_mask_type = "gaussian-0.4"
commands.experiment_8.projection_type = "prediction"
commands.experiment_8.projection_top_k = "1"
commands.experiment_8.q_baseline_masks = [
    "blur",
]
commands.experiment_8.q_directions = [
    "deletion",
    "insertion",
]
commands.experiment_8.q_job_array = "10-90:20" # "0,100"

if __name__ == "__main__":
    args = commands.experiment_8.parse_args()

    experiment_prefix = (
        os.path.basename(__file__).split(".")[0].replace("experiment_", "")
    )
    experiment_master(args, experiment_prefix=experiment_prefix)
