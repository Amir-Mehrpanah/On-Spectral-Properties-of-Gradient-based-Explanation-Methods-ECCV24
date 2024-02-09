# Experiment 7.1: Integrated gradients with different alpha priors

import argparse
import sys
import os

sys.path.append(os.getcwd())
from commands.experiment_7 import (
    experiment_master,
)
import commands.experiment_7


commands.experiment_7.alpha_mask_value = (
    "0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"  # DEBUG  
)

# Method args
commands.experiment_7.ig_alpha_priors = {  # DEBUG
    "ig_u_x2_0_1.0": "0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0",
    "ig_u_x2_0_0.8": "0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8",
    "ig_u_x2_0_0.6": "0.0 0.1 0.2 0.3 0.4 0.5 0.6",
    "ig_u_x2_0_0.4": "0.0 0.1 0.2 0.3 0.4",
    "ig_u_x2_0_0.2": "0.0 0.1 0.2",

    "ig_u_x_0_1.0": "0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0",
    "ig_u_x_0_0.8": "0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8",
    "ig_u_x_0_0.6": "0.0 0.1 0.2 0.3 0.4 0.5 0.6",
    "ig_u_x_0_0.4": "0.0 0.1 0.2 0.3 0.4",
    "ig_u_x_0_0.2": "0.0 0.1 0.2",
    
    "ig_u_x2_i_0_1.0": "0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0",
    "ig_u_x2_i_0_0.8": "0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8",
    "ig_u_x2_i_0_0.6": "0.0 0.1 0.2 0.3 0.4 0.5 0.6",
    "ig_u_x2_i_0_0.4": "0.0 0.1 0.2 0.3 0.4",
    "ig_u_x2_i_0_0.2": "0.0 0.1 0.2",

    "ig_u_x_i_0_1.0": "0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0",
    "ig_u_x_i_0_0.8": "0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8",
    "ig_u_x_i_0_0.6": "0.0 0.1 0.2 0.3 0.4 0.5 0.6",
    "ig_u_x_i_0_0.4": "0.0 0.1 0.2 0.3 0.4",
    "ig_u_x_i_0_0.2": "0.0 0.1 0.2",
}
commands.experiment_7.combination_fns = [
    "damping",
]

commands.experiment_7.batch_size = 2  # DEBUG
commands.experiment_7.baseline_mask_type = "static"
commands.experiment_7.baseline_mask_value = "0.0"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gather_stats", "-g", action="store_true")
    parser.add_argument("--compute_integrated_grad", "-i", action="store_true")
    parser.add_argument("--compute_accuracy_at_q", "-q", action="store_true")
    parser.add_argument("--compute_raw_data_accuracy_at_q", "-v", action="store_true")
    parser.add_argument("--compute_entropy", "-e", action="store_true")
    parser.add_argument("--remove_batch_data", "-r", action="store_true")
    parser.add_argument("--force_remove_batch_data", "-f", action="store_true")
    parser.add_argument("--num_batches", "-n", type=int, default=1)

    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(1)

    experiment_prefix = (
        os.path.basename(__file__).split(".")[0].replace("experiment_", "")
    )
    experiment_master(args, experiment_prefix=experiment_prefix)
