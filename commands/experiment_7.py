import json
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
constraint = "thin"

# Method args
alpha_mask_value = "0.0 0.1 0.2 0.3"  #  FINALE
alpha_mask_type = "static"
logging_level = logging.DEBUG
set_logging_level(logging_level)
min_change = 5e-2
batch_size = 64
normalize_sample = True
input_shape = (1, 224, 224, 3)
method = "noise_interpolation"
architecture = "resnet50"
dataset = "imagenet"
dataset_dir = "/home/x_amime/azizpour-group/datasets/imagenet"
baseline_mask_type = "gaussian"
projection_type = "prediction"
projection_top_k = 1
demo = False
skip_data = " ".join([StreamNames.log_probs, StreamNames.results_at_projection])

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
    parser.add_argument("--merge_stats", "-m", action="store_true")
    parser.add_argument("--compute_spectral_lens", "-s", action="store_true")
    parser.add_argument("--compute_accuracy_at_q", "-q", action="store_true")

    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(1)

    for i in range(1):  # FINALE 
        experiment_name = os.path.basename(__file__).split(".")[0] + "_" + str(i)
        save_raw_data_dir = os.path.join(save_raw_data_base_dir, experiment_name)
        save_metadata_dir = os.path.join(save_metadata_base_dir, experiment_name)
        save_output_dir = os.path.join(save_output_base_dir, experiment_name)
        
        job_array = "0-990:10"  # FINALE
        # image_index = "skip take" # skip num_elements (a very bad hack) todo clean up
        array_process = (
            f'array_process="--image_index $((1000*{i} + $SLURM_ARRAY_TASK_ID)) 10"'
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
                skip_data=skip_data,
                dataset_dir=dataset_dir,
                batch_size=batch_size,
                args_state=args_state,
                args_pattern=args_pattern,
                normalize_sample=normalize_sample,
                save_raw_data_dir=save_raw_data_dir,
                save_metadata_dir=save_metadata_dir,
            )

            wait_in_queue(0)  # wait for all jobs to finish

        if args.merge_stats:
            run_experiment(
                experiment_name=f"merge_{experiment_name}",
                constraint=constraint,
                action=Action.merge_stats,
                logging_level=logging_level,
                save_metadata_dir=save_metadata_dir,
            )

            wait_in_queue(0)  # wait for all jobs to finish

        if args.compute_spectral_lens:
            run_experiment(
                experiment_name=f"sl_{experiment_name}",
                constraint=constraint,
                action=Action.compute_spectral_lens,
                logging_level=logging_level,
                save_metadata_dir=save_metadata_dir,
                save_raw_data_dir=save_raw_data_dir,
            )

            wait_in_queue(0)  # wait for all jobs to finish

        job_array = "10-70:20"  # FINALE
        array_process = (
            f'array_process="--q $SLURM_ARRAY_TASK_ID"'
        )
        if args.compute_accuracy_at_q:
            run_experiment(
                sweeper_name="_sweeper_torch.sbatch",
                job_array=job_array,
                array_process=array_process,
                experiment_name=f"acc_{experiment_name}",
                constraint=constraint,
                number_of_gpus=1,
                action=Action.compute_accuracy_at_q,
                logging_level=logging_level,
                save_metadata_dir=save_metadata_dir,
                batch_size=128,
            )

            # wait_in_queue(0)  # wait for all jobs to finish
