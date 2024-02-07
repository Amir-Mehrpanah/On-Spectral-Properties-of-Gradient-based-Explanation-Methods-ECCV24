import json
import numpy as np
import logging
import argparse
import sys
import os

sys.path.append(os.getcwd())
from source.utils import Action

from commands.experiment_base import (
    wait_in_queue,
    run_experiment,
    set_logging_level,
    remove_files,
    save_raw_data_base_dir,
    save_metadata_base_dir,
    save_output_base_dir,
)

# Slurm args
job_array = "0-99"
array_process = 'array_process="--image_index $SLURM_ARRAY_TASK_ID 1"'
constraint = "thin"
experiment_name = os.path.basename(__file__).split(".")[0]

# Method args
alpha_mask_value = " ".join([f"{x:.2}" for x in np.linspace(0, 0.5, 20)])
logging_level = logging.DEBUG
set_logging_level(logging_level)
min_change = 5e-4
alpha_prior = ("sl_u_0_0.5", alpha_mask_value)
batch_size = 1
normalize_sample = "False"
method = "noise_interpolation"
combination_fn = "additive_combination" # DEBUG additive_combination
architecture = "resnet50"
dataset = "imagenet"
dataset_dir = "/proj/azizpour-group/datasets/imagenet"
input_shape = (1, 224, 224, 3)
baseline_mask_type = "gaussian-0.3" # DEBUG
projection_type = "prediction"
projection_top_k = 1
alpha_mask_type = "static"
stats_log_level = 1
demo = False
save_raw_data_dir = os.path.join(save_raw_data_base_dir, experiment_name)
save_metadata_dir = os.path.join(save_metadata_base_dir, experiment_name)
save_output_dir = os.path.join(save_output_base_dir, experiment_name)

_args_pattern_state = {
    # "key": ["pattern", "compilation state"],
    "alpha_mask": ["j", "dynamic"],
    "image": ["i", "dynamic"],
}
args_state = json.dumps(
    {k: v[1] for k, v in _args_pattern_state.items()},
    separators=(";", ":"),  # semi-colon is used to separate args
)
args_pattern = json.dumps(
    {k: v[0] for k, v in _args_pattern_state.items()}, separators=(";", ":")
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gather_stats", "-g", action="store_true")
    parser.add_argument("--compute_spectral_lens", "-s", action="store_true")

    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(1)

    if args.gather_stats:
        job_name = experiment_name
        run_experiment(
            experiment_name=job_name,
            job_array=job_array,
            array_process=array_process,
            constraint=constraint,
            number_of_gpus=1,
            action=Action.gather_stats,
            logging_level=logging_level,
            method=method,
            architecture=architecture,
            dataset=dataset,
            min_change=min_change,
            combination_fn=combination_fn,
            alpha_mask_value=alpha_mask_value,
            alpha_mask_type=alpha_mask_type,
            projection_type=projection_type,
            projection_top_k=projection_top_k,
            baseline_mask_type=baseline_mask_type,
            stats_log_level=stats_log_level,
            demo=demo,
            dataset_dir=dataset_dir,
            batch_size=batch_size,
            args_state=args_state,
            args_pattern=args_pattern,
            normalize_sample=normalize_sample,
            save_raw_data_dir=save_raw_data_dir,
            save_metadata_dir=save_metadata_dir,
        )
        wait_in_queue(0, jobnames=job_name)  # wait for all jobs to finish
        job_name = f"merge_{experiment_name}"
        run_experiment(
            experiment_name=job_name,
            constraint=constraint,
            action=Action.merge_stats,
            logging_level=logging_level,
            save_metadata_dir=save_metadata_dir,
        )
        wait_in_queue(0, jobnames=job_name)  # wait for all jobs to finish
        remove_files(save_metadata_dir)

    if args.compute_spectral_lens:
        job_name = experiment_name + "_sl"
        run_experiment(
            experiment_name=job_name,
            constraint=constraint,
            action=Action.compute_spectral_lens,
            logging_level=logging_level,
            save_metadata_dir=save_metadata_dir,
            save_raw_data_dir=save_raw_data_dir,
            stream_statistic="meanx2",
            alpha_mask_name=alpha_prior[0],
            alpha_prior=alpha_prior[1],
        )
        wait_in_queue(0, job_name)
        job_name = f"merge_sl_{experiment_name}"
        run_experiment(
            experiment_name=job_name,
            constraint=constraint,
            action=Action.merge_stats,
            glob_path="sl_*.csv",
            file_name="merged_sl_metadata.csv",
            logging_level=logging_level,
            save_metadata_dir=save_metadata_dir,
        )
        wait_in_queue(0, jobnames=job_name)  # wait for all jobs to finish
