import os
import numpy as np

num_alphas = 10
alphas = np.linspace(0.1, 1, num_alphas)
min_changes = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
method = "noise_interpolation"
architecture = "resnet50"

sweeper = (
    lambda min_change, alpha, method, architecture: "sbatch "
    "--export "
    f"min_change={min_change},"
    f"alpha={alpha},"
    f"method={method},"
    f"architecture={architecture} "
    "source/commands/_sweeper.sbatch"
)

for alpha in alphas:
    for min_change in min_changes:
        os.system(sweeper(min_change, alpha, method, architecture))
