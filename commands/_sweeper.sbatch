#!/usr/bin/env bash
#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 4G
#SBATCH --time 12:00:00
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user %u@kth.se
#SBATCH --output /Midgard/home/%u/an_explanation_model/outputs/slurm_logs/%J_slurm.out
#SBATCH --error  /Midgard/home/%u/an_explanation_model/outputs/slurm_logs/%J_slurm.err

#<array_process>
#MOD_PLACEHOLDER 
#</$array_process>

echo cwd: $PWD
echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
. ~/miniconda3/etc/profile.d/conda.sh
conda activate xmodel

echo SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID
echo sbatch script: method args $method_args
echo sbatch script: array args $array_process

python driver.py --assert_device $array_process $method_args

echo "Job ${SLURM_JOB_ID} on ${SLURMD_NODENAME} finished"
