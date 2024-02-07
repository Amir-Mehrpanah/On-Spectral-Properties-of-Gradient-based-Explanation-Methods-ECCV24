from glob import glob
import json
import numpy as np
import logging
import argparse
import sys
import os

sys.path.append(os.getcwd())
from source.utils import Action, StreamNames

from commands.experiment_base import (
    wait_in_queue,
    run_experiment,
    set_logging_level,
    save_raw_data_base_dir,
    save_metadata_base_dir,
    save_output_base_dir,
)

# Slurm args
job_array = "0-990:10"  # DEBUG
constraint = "thin"

# Method args
alpha_mask_value = "0.0 0.1 0.2 0.3"  # DEBUG
logging_level = logging.DEBUG
set_logging_level(logging_level)
min_change = 5e-2
batch_size = 32
normalize_sample = True
method = "noise_interpolation"
architecture = "resnet50"
dataset = "imagenet"
dataset_dir = "/proj/azizpour-group/datasets/imagenet"
input_shape = (1, 224, 224, 3)
baseline_mask_type = "gaussian"
projection_type = "prediction"
projection_top_k = 1
alpha_mask_type = "static"
demo = False
skip_data = " ".join([StreamNames.vanilla_grad_mask, StreamNames.results_at_projection])

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

    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(1)

    for i in range(1):  # DEBUG
        experiment_name = os.path.basename(__file__).split(".")[0] + "_" + str(i)
        save_raw_data_dir = os.path.join(save_raw_data_base_dir, experiment_name)
        save_metadata_dir = os.path.join(save_metadata_base_dir, experiment_name)
        save_output_dir = os.path.join(save_output_base_dir, experiment_name)
        # image_index = "0 100" # skip num_elements (a very bad hack) todo clean up
        array_process = (
            f'array_process="--image_index $((1000*{i} + 10*$SLURM_ARRAY_TASK_ID)) 10"'
        )

        if args.gather_stats:
            run_experiment(
                experiment_name=experiment_name,
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
                alpha_mask_value=alpha_mask_value,
                alpha_mask_type=alpha_mask_type,
                projection_type=projection_type,
                projection_top_k=projection_top_k,
                baseline_mask_type=baseline_mask_type,
                demo=demo,
                dataset_dir=dataset_dir,
                batch_size=batch_size,
                args_state=args_state,
                args_pattern=args_pattern,
                normalize_sample=normalize_sample,
                skip_data=skip_data,
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

            files = glob(os.path.join(save_metadata_dir, "*"))
            for f in files:
                if "merged" in f:
                    continue
                os.remove(f)
