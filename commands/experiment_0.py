import os
import numpy as np

num_alphas = 10
job_array_image_index = "1"
alphas = np.linspace(0.1, 1, num_alphas)
min_changes = [1e-2, 1e-3, 1e-4, 1e-5]
method = "noise_interpolation"
architecture = "resnet50"
dataset = "imagenet"
baseline_mask_type = "gaussian"
projection_type = "label"
alpha_mask_type = "static"
save_raw_data_dir = "/local_storage/users/amirme/raw_data/experiment_0"
save_metadata_dir = "/local_storage/users/amirme/metadata/experiment_0"
node = "gondor"
sweeper_cmd = (
    lambda alpha, min_change: "sbatch "
    f"--array={job_array_image_index} "
    f"--constraint={node} "
    "--export "
    f"method={method},"
    f"architecture={architecture},"
    f"dataset={dataset},"
    f'method_args="'
    f"--no_demo "
    f"--normalize_sample "
    f"--min_change={min_change} "
    f"--alpha_mask_type={alpha_mask_type} "
    f"--alpha_mask_value={alpha} "
    f"--projection_type={projection_type} "
    f"--baseline_mask_type={baseline_mask_type} "
    f"--save_raw_data_dir={save_raw_data_dir} "
    f"--save_metadata_dir={save_metadata_dir}"
    f'" '
    "source/commands/_sweeper.sbatch"
)

for alpha in alphas:
    for min_change in min_changes:
        os.system(sweeper_cmd(alpha, min_change))
