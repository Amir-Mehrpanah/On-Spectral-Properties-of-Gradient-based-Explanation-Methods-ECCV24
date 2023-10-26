import json

import logging

import sys
import os

sys.path.append(os.getcwd())
from source.utils import Action

from commands.experiment_base import (
    wait_in_queue,
    run_experiment,
    set_logging_level,
    save_raw_data_base_dir,
    save_metadata_base_dir,
)

# Slurm args
job_array_image_index = "3,5,9,11"  # comma separated string
constraint = "gondor"
experiment_name = os.path.basename(__file__).split(".")[0]
number_of_gpus = 4

# Method args
logging_level = logging.DEBUG
set_logging_level(logging_level)
alpha_mask_value = "0.1 0.3 0.5 0.7"  # 4 # space separated string
min_change = 5e-4
batch_size = 16
normalize_sample = True
method = "noise_interpolation"
architecture = "resnet50"
dataset = "imagenet"
baseline_mask_type = "gaussian"
projection_type = "prediction"
projection_top_k = 1
alpha_mask_type = "static"
demo = False
save_raw_data_dir = os.path.join(save_raw_data_base_dir, experiment_name)
save_metadata_dir = os.path.join(save_metadata_base_dir, experiment_name)

_args_pattern_state = {
    # "key": ["pattern", "compilation state"],
    "forward": ["i", "static"],
    "alpha_mask": ["i", "dynamic"],
    "projection": ["i", "static"],
    "image": ["i", "static"],
    "baseline_mask": ["i", "static"],
    "normalize_sample": ["i", "static"],
    "demo": ["i", "static,meta"],
}
args_state = json.dumps(
    {k: v[1] for k, v in _args_pattern_state.items()},
    separators=(";", ":"),  # semi-colon is used to separate args
)
args_pattern = json.dumps(
    {k: v[0] for k, v in _args_pattern_state.items()}, separators=(";", ":")
)

if __name__ == "__main__":
    run_experiment(
        experiment_name=experiment_name,
        job_array_image_index=job_array_image_index,
        constraint=constraint,
        action=Action.gather_stats,
        logging_level=logging_level,
        method=method,
        architecture=architecture,
        dataset=dataset,
        min_change=min_change,
        alpha_mask_type=alpha_mask_type,
        alpha_mask_value=alpha_mask_value,
        projection_type=projection_type,
        projection_top_k=projection_top_k,
        baseline_mask_type=baseline_mask_type,
        demo=demo,
        batch_size=batch_size,
        args_state=args_state,
        args_pattern=args_pattern,
        normalize_sample=normalize_sample,
        save_raw_data_dir=save_raw_data_dir,
        save_metadata_dir=save_metadata_dir,
    )

    wait_in_queue(0)  # wait for all jobs to finish

    run_experiment(
        experiment_name=f"merge_{experiment_name}",
        constraint=constraint,
        action=Action.merge_stats,
        logging_level=logging_level,
        save_metadata_dir=save_metadata_dir,
    )

    wait_in_queue(0)  # wait for all jobs to finish

    run_experiment(
        experiment_name=f"consistency_{experiment_name}",
        constraint=constraint,
        logging_level=logging_level,
        number_of_gpus=number_of_gpus,
        action=Action.compute_consistency,
        save_metadata_dir=save_metadata_dir,
        batch_size=batch_size,
    )
