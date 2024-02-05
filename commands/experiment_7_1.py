## Experiment 7.1: Integrated gradients with different alpha priors

import json
import logging
import argparse
from glob import glob
import sys
import os

sys.path.append(os.getcwd())
from source.utils import Action, Statistics, StreamNames

from commands.experiment_base import (
    wait_in_queue,
    remove_files,
    run_experiment,
    set_logging_level,
    save_raw_data_base_dir,
    save_metadata_base_dir,
)

# Slurm args
constraint = "thin"

# Method args
alpha_mask_value = "0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"  #  DEBUG  
alpha_priors = {#  DEBUG 
    "ig_u_0_0.9": "0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9",
    # "ig_u_0_0.7": "0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7",
    # "ig_u_0_0.5": "0.0 0.1 0.2 0.3 0.4 0.5",
    # "ig_u_0_0.3": "0.0 0.1 0.2 0.3",
    # "ig_u_0_0.1": "0.0 0.1",
}
stream_statistics = [#  DEBUG 
    Statistics.meanx,
    Statistics.meanx2,
]
ig_elementwise = ["True", "False"]
alpha_mask_type = "static"
logging_level = logging.DEBUG
set_logging_level(logging_level)
min_change = 5e-2
batch_size = 2
normalize_sample = "False"
input_shape = (1, 224, 224, 3)
method = "noise_interpolation"
combination_fn = "convex_combination"
architecture = "resnet50"
dataset = "imagenet"
dataset_dir = "/home/x_amime/azizpour-group/datasets/imagenet"
baseline_mask_type = "static"
baseline_mask_value = "0.0"
projection_type = "prediction"
projection_top_k = 1
stats_log_level = 1
demo = False

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
    parser.add_argument("--compute_integrated_grad", "-i", action="store_true")
    parser.add_argument("--compute_accuracy_at_q", "-q", action="store_true")
    parser.add_argument("--remove_batch_data", "-r", action="store_true")

    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(1)

    for batch in range(1):  # DEBUG
        experiment_name = os.path.basename(__file__).split(".")[0] + "_" + str(batch)
        save_raw_data_dir = os.path.join(save_raw_data_base_dir, experiment_name)
        save_metadata_dir = os.path.join(save_metadata_base_dir, experiment_name)

        job_array = "0-990:10"  # DEBUG  
        # image_index = "skip take" # skip num_elements (a very bad hack) todo clean up
        array_process = (
            f'array_process="--image_index $((1000*{batch} + $SLURM_ARRAY_TASK_ID)) 10"'
        )

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
                baseline_mask_value=baseline_mask_value,
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

            wait_in_queue(0,jobnames=job_name)  # wait for all jobs to finish
            job_name = f"merge_{experiment_name}"
            run_experiment(
                experiment_name=job_name,
                constraint=constraint,
                action=Action.merge_stats,
                logging_level=logging_level,
                save_metadata_dir=save_metadata_dir,
            )
            wait_in_queue(0,jobnames=job_name)  # wait for all jobs to finish
            remove_files(save_metadata_dir)

        if args.compute_integrated_grad:
            job_name = []
            for r, ig_prod in enumerate(ig_elementwise):
                for k, (alpha_mask_name, alpha_prior) in enumerate(alpha_priors.items()):
                    for j, stream_statistic in enumerate(stream_statistics):
                        temp_name = alpha_mask_name + "_" + stream_statistic
                        job_name.append(f"ig_{experiment_name}_{r}_{k}_{j}")
                        run_experiment(
                            experiment_name=job_name[-1],
                            constraint=constraint,
                            action=Action.compute_integrated_grad,
                            logging_level=logging_level,
                            save_metadata_dir=save_metadata_dir,
                            save_raw_data_dir=save_raw_data_dir,
                            stream_statistic=stream_statistic,
                            alpha_mask_name=temp_name,
                            alpha_prior=alpha_prior,
                            ig_elementwise=ig_prod,
                        )

            wait_in_queue(0,job_name)
            job_name = f"merge_{experiment_name}"
            run_experiment(
                experiment_name=job_name,
                constraint=constraint,
                action=Action.merge_stats,
                glob_path="*.csv",
                file_name="merged_ig_metadata.csv",
                logging_level=logging_level,
                save_metadata_dir=save_metadata_dir,
            )
            wait_in_queue(0,jobnames=job_name)  # wait for all jobs to finish
            remove_files(save_metadata_dir)

        job_array = "10-90:20"  # DEBUG 
        array_process = f'array_process="--q $SLURM_ARRAY_TASK_ID"'
        if args.compute_accuracy_at_q:
            job_name = f"acc_{experiment_name}"
            run_experiment(
                sweeper_name="_sweeper_torch.sbatch",
                job_array=job_array,
                array_process=array_process,
                experiment_name=job_name,
                constraint=constraint,
                number_of_gpus=1,
                glob_path="merged_ig_*.csv",
                save_file_name_prefix="igq",
                action=Action.compute_accuracy_at_q,
                logging_level=logging_level,
                save_metadata_dir=save_metadata_dir,
                batch_size=128,
            )

            wait_in_queue(0,jobnames=job_name)  # wait for all jobs to finish
            job_name = f"merge_{experiment_name}"
            run_experiment(
                experiment_name=job_name,
                constraint=constraint,
                action=Action.merge_stats,
                glob_path="igq_*.csv",
                file_name="merged_igq_metadata.csv",
                logging_level=logging_level,
                save_metadata_dir=save_metadata_dir,
            )
            wait_in_queue(0,jobnames=job_name)  # wait for all jobs to finish
            remove_files(save_metadata_dir)
            
        if args.remove_batch_data and batch != 0:
            remove_files(save_raw_data_dir)
            