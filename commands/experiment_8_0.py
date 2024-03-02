# Experiment 8.0: Smooth Grad to compute entropy -gQ 5

import argparse
import sys
import os

sys.path.append(os.getcwd())
from commands.experiment_8 import (
    experiment_master,
)
import commands.experiment_8


commands.experiment_8.alpha_mask_value = "0.0"  # DEBUG

# Method args
commands.experiment_8.combination_fns = [
    "additive",
    # "convex",
]
commands.experiment_8.ig_alpha_priors = {
    # "ig_sg_u_x_0": "0.0",
    "none": None,
}
commands.experiment_8.gather_stats_batch_size = 128  # DEBUG
commands.experiment_8.baseline_mask_type = "gaussian"
commands.experiment_8.gather_stats_job_array = "0"
commands.experiment_8.gather_stats_take_batch_size = "1"
commands.experiment_8.q_baseline_masks = [
    "black",
]
commands.experiment_8.q_directions = [
    "deletion",
    # "insertion",
]
commands.experiment_8.q_job_array = "0"

if __name__ == "__main__":
    args = commands.experiment_8.parse_args()

    experiment_prefix = (
        os.path.basename(__file__).split(".")[0].replace("experiment_", "")
    )
    experiment_master(args, experiment_prefix=experiment_prefix)
