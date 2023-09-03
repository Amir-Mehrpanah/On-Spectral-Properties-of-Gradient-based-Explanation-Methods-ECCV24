import os
import numpy as np

num_alphas = 10
job_array_image_index = "3,5,7,9,11,13"
alphas = np.linspace(0.1, 1, num_alphas)
min_change = 1e-3
method = "noise_interpolation"
architecture = "resnet50"
dataset = "imagenet"
baseline_mask_type = "gaussian"
projection_type = "label"
alpha_mask_type = "static"
save_raw_data_dir = "/local_storage/users/amirme/raw_data/noise_interpolation"
save_metadata_dir = "/local_storage/users/amirme/metadata/noise_interpolation"
node = "gondor"
sweeper_cmd = (
    lambda alpha: "sbatch "
    f"--array={job_array_image_index} "
    f"--constraint={node} "
    "--export "
    f"method={method},"
    f"architecture={architecture},"
    f"dataset={dataset},"
    f'method_args="'
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
    os.system(sweeper_cmd(alpha))
