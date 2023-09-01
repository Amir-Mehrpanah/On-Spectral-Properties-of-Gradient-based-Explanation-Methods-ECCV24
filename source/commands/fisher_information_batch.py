import os
import numpy as np

job_array_image_index = "1,3,5,7"
alphas = [0]
min_changes = [1e-5]
method = "fisher_information"
architecture = "resnet50"
dataset = "imagenet"
baseline_mask_type = "static"
baseline_mask_value = 0
projection_type = "static"
projection_indices = range(1000)
alpha_mask_type = "static"
save_raw_data_dir = f"/local_storage/users/amirme/raw_data/{method}"
save_metadata_dir = f"/local_storage/users/amirme/metadata/{method}"
sweeper_cmd = (
    lambda min_change, alpha, projection_index: "sbatch "
    f"--array={job_array_image_index} --export "
    f"method={method},"
    f"architecture={architecture},"
    f"dataset={dataset},"
    f'method_args="'
    f"--min_change={min_change} "
    f"--alpha_mask_type={alpha_mask_type} "
    f"--alpha_mask_value={alpha} "
    f"--projection_type={projection_type} "
    f"--projection_index={projection_index} "
    f"--baseline_mask_type={baseline_mask_type} "
    f"--baseline_mask_value={baseline_mask_value} "
    f"--save_raw_data_dir={save_raw_data_dir} "
    f"--save_metadata_dir={save_metadata_dir}"
    f'" '
    "source/commands/_sweeper.sbatch"
)

for projection_index in projection_indices:
    for alpha in alphas:
        for min_change in min_changes:
            os.system(sweeper_cmd(min_change, alpha, projection_index))
