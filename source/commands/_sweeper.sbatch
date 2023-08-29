#!/usr/bin/env bash
# SBATCH --gres gpu:1
#SBATCH --cpus-per-task 4
#SBATCH --time 12:00:00
#SBATCH --constrain "gondor"
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user amirme@kth.se
#SBATCH --output /Midgard/home/amirme/an_explanation_model/outputs/slurm_logs/%J_slurm.out
#SBATCH --error  /Midgard/home/amirme/an_explanation_model/outputs/slurm_logs/%J_slurm.err

echo cwd: $PWD
echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
. ~/miniconda3/etc/profile.d/conda.sh
conda activate xmodel

python driver.py --assert_device --method $method --architecture $architecture --min_change $min_change --alpha $alpha

echo "Job ${SLURM_JOB_ID} on ${SLURMD_NODENAME} finished"
