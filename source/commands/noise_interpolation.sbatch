#!/usr/bin/env bash

num_alphas=20
linspace="import numpy as np; print(' '.join(map(str, np.linspace(0.1, 1, $num_alphas))))"
alphas=$(python -c $linspace)
min_changes=(1e-2 1e-3 1e-4 1e-05)

for alpha in $alphas; do
    for min_change in $min_changes; do
        sbatch --export min_change=$min_change,alpha=$alpha,method="noise_interpolation",architecture="resnet18" source/commands/_sweeper.sbatch
    done
done