#!/usr/bin/env bash
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 4
#SBATCH --time 12:00:00
#SBATCH --constrain "gondor"
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user amirme@kth.se
#SBATCH --output /Midgard/home/%u/interp_stats/logs/%J_slurm.out
#SBATCH --error  /Midgard/home/%u/interp_stats/logs/%J_slurm.err

echo cwd: $PWD
echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
. ~/miniconda3/etc/profile.d/conda.sh
conda activate xmodel
python driver.py --max_batches 1e3 --seed ${SEED}

echo "Job ${SLURM_JOB_ID} on ${SLURMD_NODENAME} finished"
