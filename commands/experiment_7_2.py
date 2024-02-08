# Experiment 7.2: Integrated gradients squared with Smooth Grad different alpha priors


import json
import logging
import argparse
from glob import glob
import sys
import os

sys.path.append(os.getcwd())
from source.utils import Action, Statistics
from commands.experiment_7 import (
    experiment_master,
)
import commands.experiment_7


commands.experiment_7.alpha_mask_value = ""  # DEBUG

# Method args
commands.experiment_7.ig_alpha_priors = {  # DEBUG
    "ig_sg_u_0_1.0": "0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0",
    # "ig_sg_u_0_0.9": "0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9",
    # "ig_sg_u_0_0.7": "0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7",
    # "ig_sg_u_0_0.5": "0.0 0.1 0.2 0.3 0.4 0.5",
    # "ig_sg_u_0_0.3": "0.0 0.1 0.2 0.3",
    # "ig_sg_b_0_1.0": "0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9",
}
commands.experiment_7.ig_stream_statistics = [  # DEBUG
    # Statistics.meanx,
    # Statistics.meanx2,
]
commands.experiment_7.combination_fns = [
    # "additive_combination",
    # "convex_combination",
    # "damping_combination",
]

commands.experiment_7.batch_size = 128  # DEBUG

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gather_stats", "-g", action="store_true")
    parser.add_argument("--compute_integrated_grad", "-i", action="store_true")
    parser.add_argument("--compute_spectral_lens", "-s", action="store_true")
    parser.add_argument("--compute_accuracy_at_q", "-q", action="store_true")
    parser.add_argument("--compute_raw_data_accuracy_at_q", "-v", action="store_true")
    parser.add_argument("--compute_entropy", "-e", action="store_true")
    parser.add_argument("--remove_batch_data", "-r", action="store_true")
    parser.add_argument("--num_batches", "-n", type=int, default=1)

    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(1)

    experiment_master(
        args,
    )
