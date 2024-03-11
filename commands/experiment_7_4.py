# Experiment 7.4: only for visualization

import argparse
import sys
import os

sys.path.append(os.getcwd())
from commands.experiment_7 import (
    experiment_master,
)
import commands.experiment_7

alphas = " ".join([str(x / 100) for x in range(0, 101)])
# alphas = "0.0 0.1 0.2 0.3 0.4 0.5 0.6"

commands.experiment_7.alpha_mask_value = alphas

# Method args
commands.experiment_7.ig_alpha_priors = {  # DEBUG
    # "ig_vg_u_x_0": "0.0",
    # "ig_vg_u_x2_0": "0.0",
    # "ig_vg_i_u_x_0": "0.0",
    # "ig_vg_i_u_x2_0": "0.0",
    # "ig_sg_u_x_0.1": "0.1",
    # "ig_sg_u_x_0.2": "0.2",
    # "ig_sg_u_x_0.3": "0.3",
    # "ig_sg_u_x_0.4": "0.4",
    # "ig_sg_u_x_0.5": "0.5",
    # "ig_sg_u_x_1.0": "1.0",

    # "ig_sg_u_x2_0.1": "0.1",
    # "ig_sg_u_x2_0.2": "0.2",
    # "ig_sg_u_x2_0.3": "0.3",
    # "ig_sg_u_x2_0.4": "0.4",
    "ig_sg_u_x2_0.5": "0.5",
    # "ig_sg_u_x2_0.6": "0.6",
    # "ig_sg_u_x2_1.0": "1.0",

    # "ig_sg_u_x2_1.0": "1.0",
    # "ig_sg_u_x_0_0.2": "0.0 0.1 0.2",
    # "ig_sg_u_x_0_0.5": "0.0 0.1 0.2 0.3 0.4 0.5",
    "ig_sg_u_x2_0_1.0": alphas,
    "al_sg_u_x2_0_1.0": alphas,
    # "sl_sg_u_x2_0_1.0": alphas,
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
commands.experiment_7.combination_fns = [
    "additive",
]

commands.experiment_7.baseline_mask_type = "gaussian-0.3"
commands.experiment_7.projection_type = "prediction"
commands.experiment_7.projection_top_k = "1"
commands.experiment_7.gather_stats_dir_batch_size = 100
commands.experiment_7.gather_stats_take_batch_size = 1
commands.experiment_7.gather_stats_job_array = "7,8,10,13,26"
commands.experiment_7.q_baseline_masks = [
]
commands.experiment_7.q_directions = [
]
commands.experiment_7.q_job_array = ""  #

if __name__ == "__main__":
    args = commands.experiment_7.parse_args()

    experiment_prefix = (
        os.path.basename(__file__).split(".")[0].replace("experiment_", "")
    )
    experiment_master(args, experiment_prefix=experiment_prefix)
