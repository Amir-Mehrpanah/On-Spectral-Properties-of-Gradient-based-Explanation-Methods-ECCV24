# Experiment 7.3: RISE and Occlusion

import sys
import os

sys.path.append(os.getcwd())
from commands.experiment_7 import (
    experiment_master,
)
import commands.experiment_7


commands.experiment_7.alpha_mask_type = "image_bernoulli-7x7"
commands.experiment_7.alpha_mask_value = "0.1"  # DEBUG
commands.experiment_7.gather_stats_job_array = "0-9"  # DEBUG
# Method args
commands.experiment_7.combination_fns = [
    "convex",
]

commands.experiment_7.baseline_mask_type = "static"
commands.experiment_7.baseline_mask_value = "0.0"
commands.experiment_7.projection_type = "prediction"
commands.experiment_7.projection_top_k = "1"
commands.experiment_7.q_baseline_masks = [
    "blur",
    # "black",
]
commands.experiment_7.q_directions = [
    "deletion",
    "insertion",
]
commands.experiment_7.q_job_array = "10-90:20"  # "0,100"

commands.experiment_7._args_pattern_state["projection"] = ["p", "dynamic"]

if __name__ == "__main__":
    args = commands.experiment_7.parse_args()

    experiment_prefix = (
        os.path.basename(__file__).split(".")[0].replace("experiment_", "")
    )
    experiment_master(args, experiment_prefix=experiment_prefix)
