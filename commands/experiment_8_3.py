# Experiment 8.3: RISE and Occlusion

import sys
import os

sys.path.append(os.getcwd())
from commands.experiment_8 import (
    experiment_master,
)
import commands.experiment_8


commands.experiment_8.alpha_mask_type = "image_ohcat-7x7 image_bernoulli-7x7"
commands.experiment_8.alpha_mask_value = "nan 0.1"
commands.experiment_8.ig_alpha_priors = {
    "ig_rise_u_x_0.1": "0.1",
    # "ig_rise_u_x2_0.1": "0.1",
    "ig_occlusion_u_x_nan": "nan",
    # "ig_occlusion_u_x2_nan": "nan",
}
# Method args
commands.experiment_8.combination_fns = [
    "convex",
]
commands.experiment_8.gather_stats_max_batches = (
    8000 // commands.experiment_8.gather_stats_batch_size
)
commands.experiment_8.explainer_fn = "finite_difference"
commands.experiment_8._args_pattern_state = {
    # "key": ["pattern", "compilation state"],
    "image": ["i", "dynamic"],
}
commands.experiment_8.baseline_mask_type = "static"
commands.experiment_8.baseline_mask_value = "0.0"
commands.experiment_8.projection_type = "prediction"
commands.experiment_8.projection_top_k = "1"
commands.experiment_8.q_baseline_masks = [
    "blur",
]
commands.experiment_8.q_directions = [
    "deletion",
    "insertion",
]

if __name__ == "__main__":
    args = commands.experiment_8.parse_args()

    experiment_prefix = (
        os.path.basename(__file__).split(".")[0].replace("experiment_", "")
    )
    experiment_master(args, experiment_prefix=experiment_prefix)
