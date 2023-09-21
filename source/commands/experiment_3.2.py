import os
import subprocess
import time
import numpy as np

job_array_image_index = "5,8,10,12"
alphas = np.linspace(0, 0.6, 7)
min_change = 5e-4
noise_interpolation = "noise_interpolation"
architecture = "resnet50"
dataset = "imagenet"
baseline_mask_type = "gaussian"
projection_type = "prediction"
max_k = 10
projection_top_ks = range(1, max_k + 1)
alpha_mask_type = "static"
save_raw_data_dir = f"/local_storage/users/amirme/raw_data/experiment_3.2"
save_metadata_dir = f"/local_storage/users/amirme/metadata/experiment_3.2"
sweeper_cmd = (
    lambda method, alpha, projection_top_k: "sbatch --constraint=gondor "
    f"--array={job_array_image_index} --export "
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
    f"--projection_top_k={projection_top_k} "
    f"--baseline_mask_type={baseline_mask_type} "
    f"--save_raw_data_dir={save_raw_data_dir} "
    f"--save_metadata_dir={save_metadata_dir}"
    f'" '
    "source/commands/_sweeper.sbatch"
)

for alpha in alphas:
    for projection_top_k in projection_top_ks:
        print(
            "sumbitting job for alpha", alpha, "and projection_top_k", projection_top_k
        )
        os.system(sweeper_cmd(noise_interpolation, alpha, projection_top_k))
        result = 10
        while result > 6:
            result = subprocess.run(["squeue", "-u", "amirme"], stdout=subprocess.PIPE)
            result = result.stdout.decode()
            result = len(result.split("\n")) - 2
            print(
                f"there are {result} number of jobs in the queue, waiting for finishing the jobs"
            )
            time.sleep(5)
