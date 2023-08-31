import os
import numpy as np

num_alphas = 10
alphas = [1.0]  # np.linspace(0.1, 1, num_alphas)
min_changes = [1e-3]  # np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
method = "noise_interpolation"
architecture = "resnet50"
dataset = "imagenet"
sweeper_cmd = (
    lambda min_change, alpha: "sbatch "
    "--export "
    f"method={method},"
    f"architecture={architecture},"
    f"dataset={dataset},"
    f'method_args="'
    f"--min_change={min_change} "
    f"--alpha_mask_type=static "
    f"--alpha_mask_value={alpha} "
    f"--projection_type=label "
    f'--baseline_mask_type=gaussian" '
    "source/commands/_sweeper.sbatch"
)

for alpha in alphas:
    for min_change in min_changes:
        os.system(sweeper_cmd(min_change, alpha))
