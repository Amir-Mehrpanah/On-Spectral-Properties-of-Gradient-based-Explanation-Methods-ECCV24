# Experiment 7.5: Smooth Grad Original paper

import argparse
import sys
import os

sys.path.append(os.getcwd())
from commands.experiment_7 import (
    experiment_master,
)
import commands.experiment_7


# original smoothgrad paper also alters the baseline_mask_type
commands.experiment_7.alpha_mask_type = "max-min"
commands.experiment_7.alpha_mask_value = "nan"
commands.experiment_7._args_pattern_state["alpha_mask"] = [
    "i",
    "dynamic",
]  # same as image so each image gets its own alpha mask value = max-min

# Method args
commands.experiment_7.ig_alpha_priors = {  # DEBUG
    "ig_sg_u_x_nan": "nan",
    # "ig_sg_u_x2_nan": "nan",
}
commands.experiment_7.combination_fns = [
    "additive",
]
commands.experiment_7.baseline_mask_type = "gaussian-0.1"
commands.experiment_7.projection_type = "prediction"
commands.experiment_7.projection_top_k = "1"
commands.experiment_7.q_baseline_masks = [
    "blur",
]
commands.experiment_7.q_directions = [
    "deletion",
    "insertion",
]
commands.experiment_7.q_job_array = "10-90:20" #"0,100"

if __name__ == "__main__":
    args = commands.experiment_7.parse_args()

    experiment_prefix = (
        os.path.basename(__file__).split(".")[0].replace("experiment_", "")
    )
    experiment_master(args, experiment_prefix=experiment_prefix)
