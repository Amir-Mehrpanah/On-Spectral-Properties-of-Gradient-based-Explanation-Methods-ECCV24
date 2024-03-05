# Experiment 8: Food101

import json
import logging
import argparse
from glob import glob
import sys
import os

sys.path.append(os.getcwd())
from source.utils import Action
from commands.experiment_base import (
    remove_files,
    wait_in_queue,
    run_experiment,
    set_logging_level,
    save_raw_data_base_dir,
    save_metadata_base_dir,
    save_temp_base_dir,
)

# Slurm args
constraint = "thin"

# Method args
alpha_mask_value = ""
# name must contain ig_ for integrated gradients
# name must contain sl_ for spectral lens
# name must contain _i_ for elementwise multiplication
# name must contain _x_ for meanx
# name must contain _x2_ for meanx2
# name must contain _sg_ for smooth grad
# name must contain _b_ for beta explanation prior
# name must contain _u_ for uniform explanation prior
# other characters are ignored
ig_alpha_priors = {}
combination_fns = []
alpha_mask_type = "static"
logging_level = logging.DEBUG
set_logging_level(logging_level)
min_change = 5e-3
gather_stats_batch_size = 128
method = "noise_interpolation"
architecture = "resnet50"
dataset = "food101"
# data is copied to the node
# see array_process
dataset_dir = "/scratch/local/data/"
baseline_mask_type = None
baseline_mask_value = None
projection_type = "prediction"
explainer_fn = "vanilla_grad"
q_baseline_masks = []
q_directions = []
projection_top_k = "1"
q_job_array = "10-90:20"
num_classes = 101
gather_stats_take_batch_size = 10
gather_stats_dir_batch_size = 1000
gather_stats_max_batches = 8000 // gather_stats_batch_size
stats_log_level = 1
demo = False
q_batch_size = 128
gather_stats_input_shape = "1 256 256 3"
q_input_shape = "256 256 3"
q_prefetch_factor = 16

# https://github.com/google-research/google-research/blob/master/interpretability_benchmark/train_resnet.py#L126
preprocess_mean_rgb = "0.561 0.440 0.312"
preprocess_std_rgb = "0.252 0.256 0.259"


_args_pattern_state = {
    # "key": ["pattern", "compilation state"],
    "alpha_mask": ["j", "dynamic"],
    "image": ["i", "dynamic"],
}

move_data_cmds = (
    'echo "Transferring food101-val.zip!"\n'
    "mkdir -p /scratch/local/data\n"
    "rsync --info=progress2 /proj/azizpour-group/datasets/food101/array_records/food101-val.zip /scratch/local/data/ \n"
    'echo "Extracting food101-val.zip!"\n'
    "unzip /scratch/local/data/food101-val.zip -d /scratch/local/data/\n"
)


def update_dynamic_args():
    global args_state, args_pattern, gather_stats_job_array

    args_state = json.dumps(
        {k: v[1] for k, v in _args_pattern_state.items()},
        separators=(";", ":"),  # semi-colon is used to separate args
    )
    args_pattern = json.dumps(
        {k: v[0] for k, v in _args_pattern_state.items()},
        separators=(";", ":"),
    )

    gather_stats_job_array = f"0-{gather_stats_dir_batch_size-gather_stats_take_batch_size}:{gather_stats_take_batch_size}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gather_stats", "-g", action="store_true")
    parser.add_argument("--compute_integrated_grad", "-t", action="store_true")
    parser.add_argument("--compute_accuracy_at_q", "-q", action="store_true")
    parser.add_argument("--compute_entropy", "-e", action="store_true")
    parser.add_argument("--remove_batch_data", "-r", action="store_true")
    parser.add_argument("--force_remove_batch_data", "-f", action="store_true")
    parser.add_argument("--num_batches", "-n", type=int, default=1)
    parser.add_argument("--start_batches", "-s", type=int, default=0)

    parser.add_argument("--compute_raw_data_accuracy_at_q", "-Q", action="store_true")

    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(1)

    args.num_batches = (
        args.start_batches + 1
        if args.num_batches <= args.start_batches
        else args.num_batches
    )
    update_dynamic_args()

    return args


def experiment_master(
    args,
    experiment_prefix=None,
):
    for batch in range(args.start_batches, args.num_batches):
        for combination_fn in combination_fns:
            experiment_prefix = (
                os.path.basename(__file__).split(".")[0]
                if experiment_prefix is None
                else experiment_prefix
            )
            experiment_name = (
                experiment_prefix + "_" + combination_fn + "_" + str(batch)
            )

            save_raw_data_dir = os.path.join(save_raw_data_base_dir, experiment_name)
            save_metadata_dir = os.path.join(save_metadata_base_dir, experiment_name)

            if args.gather_stats:
                # image_index = "skip take" # skip num_elements (a very bad hack) todo clean up
                array_process = (
                    move_data_cmds
                    + f'array_process="--image_index $(({gather_stats_dir_batch_size}*{batch} + $SLURM_ARRAY_TASK_ID)) {gather_stats_take_batch_size}"'
                )
                job_name = experiment_name
                run_experiment(
                    experiment_name=job_name,
                    save_temp_base_dir=save_temp_base_dir,
                    job_array=gather_stats_job_array,
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
                    explainer_fn=explainer_fn,
                    alpha_mask_value=alpha_mask_value,
                    alpha_mask_type=alpha_mask_type,
                    projection_type=projection_type,
                    projection_top_k=projection_top_k,
                    baseline_mask_type=baseline_mask_type,
                    stats_log_level=stats_log_level,
                    demo=demo,
                    num_classes=num_classes,
                    dataset_dir=dataset_dir,
                    batch_size=gather_stats_batch_size,
                    max_batches=gather_stats_max_batches,
                    args_state=args_state,
                    mean_rgb=preprocess_mean_rgb,
                    std_rgb=preprocess_std_rgb,
                    args_pattern=args_pattern,
                    save_raw_data_dir=save_raw_data_dir,
                    save_metadata_dir=save_metadata_dir,
                    baseline_mask_value=baseline_mask_value,
                    input_shape=gather_stats_input_shape,
                )

                wait_in_queue(0, jobnames=job_name)  # wait for all jobs to finish
                job_name = f"merge_{experiment_name}"
                run_experiment(
                    experiment_name=job_name,
                    constraint=constraint,
                    action=Action.merge_stats,
                    glob_path="????_*_????.csv",
                    file_name="merged_metadata.csv",
                    logging_level=logging_level,
                    save_metadata_dir=save_metadata_dir,
                )
                wait_in_queue(0, jobnames=job_name)  # wait for all jobs to finish
                remove_files(save_metadata_dir)

            if args.compute_integrated_grad:
                job_name = []
                assert projection_type.count(" ") == projection_top_k.count(" "), (
                    "projection_type and projection_top_k must have the same number of "
                    "elements"
                )
                proj_iter = zip(
                    projection_type.split(),
                    projection_top_k.split(),
                )
                for proj_type, proj_top_k in proj_iter:
                    for alpha_mask_name, alpha_prior in ig_alpha_priors.items():
                        # move the data in case of input multiplication
                        if "_i_" in alpha_mask_name:
                            array_process = move_data_cmds
                        else:
                            array_process = ""

                        job_name.append(
                            f"ig_{experiment_name}_{proj_type}_{alpha_mask_name}"
                        )

                        run_experiment(
                            experiment_name=job_name[-1],
                            constraint=constraint,
                            action=Action.compute_integrated_grad,
                            logging_level=logging_level,
                            save_metadata_dir=save_metadata_dir,
                            save_raw_data_dir=save_raw_data_dir,
                            alpha_mask_name=alpha_mask_name,
                            projection_type=proj_type,
                            projection_top_k=proj_top_k,
                            alpha_prior=alpha_prior,
                            input_shape=gather_stats_input_shape,  # batch is ignored by the data source
                            dataset=dataset,
                            mean_rgb=preprocess_mean_rgb,
                            std_rgb=preprocess_std_rgb,
                            dataset_dir=dataset_dir,
                            array_process=array_process,
                        )
                wait_in_queue(0, job_name)
                job_name = []
                job_name.append(f"merge_ig_{experiment_name}")
                run_experiment(
                    experiment_name=job_name[-1],
                    constraint=constraint,
                    action=Action.merge_stats,
                    glob_path="??_*.csv",
                    file_name="merged_ig_metadata.csv",
                    logging_level=logging_level,
                    save_metadata_dir=save_metadata_dir,
                )
                wait_in_queue(0, jobnames=job_name)  # wait for all jobs to finish
                remove_files(save_metadata_dir)

            if args.compute_accuracy_at_q or args.compute_raw_data_accuracy_at_q:
                if args.compute_accuracy_at_q and args.compute_raw_data_accuracy_at_q:
                    glob_file_name = "merged_*metadata.csv"
                elif args.compute_accuracy_at_q:
                    glob_file_name = "merged_??_metadata.csv"
                elif args.compute_raw_data_accuracy_at_q:
                    glob_file_name = "merged_metadata.csv"

                array_process = (
                    move_data_cmds + f'array_process="--q $SLURM_ARRAY_TASK_ID"'
                )
                job_name = []
                files = glob(os.path.join(save_metadata_dir, glob_file_name))
                print("files: ", files)
                for k, file in enumerate(files):
                    for q_direction in q_directions:
                        for q_baseline_mask in q_baseline_masks:
                            for ig_alpha_prior in ig_alpha_priors:
                                glob_path = os.path.basename(file)
                                prefix = glob_path.split("_")[1][:2]
                                job_name.append(
                                    f"acc{k}_{q_direction}_{q_baseline_mask}_{ig_alpha_prior}_{experiment_name}"
                                )
                                run_experiment(
                                    job_array=q_job_array,
                                    num_tasks=16,
                                    array_process=array_process,
                                    experiment_name=job_name[-1],
                                    constraint=constraint,
                                    number_of_gpus=1,
                                    input_shape=q_input_shape,
                                    filter_alpha_prior=ig_alpha_prior,
                                    glob_path=glob_path,
                                    save_temp_base_dir=save_temp_base_dir,
                                    save_file_name_prefix=f"{prefix}q_{ig_alpha_prior}_{q_direction}_{q_baseline_mask}",
                                    action=Action.compute_accuracy_at_q,
                                    q_direction=q_direction,
                                    architecture=architecture,
                                    logging_level=logging_level,
                                    save_metadata_dir=save_metadata_dir,
                                    batch_size=q_batch_size,
                                    prefetch_factor=q_prefetch_factor,
                                    dataset=dataset,
                                    dataset_dir=dataset_dir,
                                    q_baseline_mask=q_baseline_mask,
                                    num_classes=num_classes,
                                )

                wait_in_queue(0, jobnames=job_name)  # wait for all jobs to finish
                job_name = []

                for k, file in enumerate(files):
                    prefix = os.path.basename(file).split("_")[1][:2]
                    print(f"prefix: {prefix}")
                    job_name.append(f"merge_{prefix}_{experiment_name}")
                    run_experiment(
                        experiment_name=job_name[-1],
                        constraint=constraint,
                        action=Action.merge_stats,
                        glob_path=f"{prefix}q_*.csv",
                        file_name=f"merged_{prefix}q_metadata.csv",
                        logging_level=logging_level,
                        save_metadata_dir=save_metadata_dir,
                    )
                wait_in_queue(0, jobnames=job_name)  # wait for all jobs to finish
                remove_files(save_metadata_dir)

            if args.compute_entropy:
                job_name = f"entropy_{experiment_name}"
                run_experiment(
                    experiment_name=job_name,
                    constraint=constraint,
                    action=Action.compute_entropy,
                    save_metadata_dir=save_metadata_dir,
                )
                wait_in_queue(0, jobnames=job_name)

            if args.force_remove_batch_data or (args.remove_batch_data and batch != 0):
                remove_files(save_raw_data_dir)
